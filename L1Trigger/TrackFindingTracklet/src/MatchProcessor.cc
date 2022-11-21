#include "L1Trigger/TrackFindingTracklet/interface/MatchProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include <filesystem>

using namespace std;
using namespace trklet;

MatchProcessor::MatchProcessor(string name, Settings const& settings, Globals* global)
    : ProcessBase(name, settings, global),
      phimatchcuttable_(settings),
      zmatchcuttable_(settings),
      rphicutPStable_(settings),
      rphicut2Stable_(settings),
      rcutPStable_(settings),
      rcut2Stable_(settings),
      fullmatches_(12),
      rinvbendlut_(settings),
      luttable_(settings),
      inputProjBuffer_(3) {
  phiregion_ = name[8] - 'A';

  layerdisk_ = initLayerDisk(3);

  barrel_ = layerdisk_ < N_LAYER;

  phishift_ = settings_.nphibitsstub(N_LAYER - 1) - settings_.nphibitsstub(layerdisk_);
  dzshift_ = settings_.nzbitsstub(0) - settings_.nzbitsstub(layerdisk_);

  if (barrel_) {
    icorrshift_ = ilog2(settings_.kphi(layerdisk_) / (settings_.krbarrel() * settings_.kphider()));
    icorzshift_ = ilog2(settings_.kz(layerdisk_) / (settings_.krbarrel() * settings_.kzder()));
  } else {
    icorrshift_ = ilog2(settings_.kphi(layerdisk_) / (settings_.kz() * settings_.kphiderdisk()));
    icorzshift_ = ilog2(settings_.krprojshiftdisk() / (settings_.kz() * settings_.krder()));
  }

  luttable_.initBendMatch(layerdisk_);

  nrbits_ = 5;
  nphiderbits_ = 6;

  if (!barrel_) {
    rinvbendlut_.initProjectionBend(
        global->ITC_L1L2()->der_phiD_final.K(), layerdisk_ - N_LAYER, nrbits_, nphiderbits_);
  }

  nrinv_ = NRINVBITS;

  unsigned int region = getName()[8] - 'A';
  assert(region < settings_.nallstubs(layerdisk_));

  if (barrel_) {
    phimatchcuttable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::barrelphi, region);
    zmatchcuttable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::barrelz, region);
  } else {
    rphicutPStable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::diskPSphi, region);
    rphicut2Stable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::disk2Sphi, region);
    rcutPStable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::diskPSr, region);
    rcut2Stable_.initmatchcut(layerdisk_, TrackletLUT::MatchType::disk2Sr, region);
  }

  for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
    ialphafactinner_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSinner(i) * settings_.rDSSinner(i)) /
                          settings_.kphi();
    ialphafactouter_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSouter(i) * settings_.rDSSouter(i)) /
                          settings_.kphi();
  }

  nvm_ = settings_.nvmme(layerdisk_) * settings_.nallstubs(layerdisk_);
  nvmbins_ = settings_.nvmme(layerdisk_);
  nvmbits_ = settings_.nbitsvmme(layerdisk_) + settings_.nbitsallstubs(layerdisk_);

  nMatchEngines_ = 4;
  for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
    MatchEngineUnit tmpME(settings_, barrel_, layerdisk_, luttable_);
    tmpME.setimeu(iME);
    matchengines_.push_back(tmpME);
  }
}

void MatchProcessor::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output.find("matchout") != std::string::npos) {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    unsigned int iSeed = getISeed(tmp->getName());
    assert(iSeed < fullmatches_.size());
    assert(fullmatches_[iSeed] == nullptr);
    fullmatches_[iSeed] = tmp;
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find output: " << output;
}

void MatchProcessor::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }
  if (input == "allstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    allstubs_ = tmp;
    return;
  }
  if (input == "vmstubin") {
    auto* tmp = dynamic_cast<VMStubsMEMemory*>(memory);
    assert(tmp != nullptr);
    vmstubs_.push_back(tmp);  //to allow more than one stub in?  vmstubs_=tmp;
    return;
  }
  if (input == "projin") {
    auto* tmp = dynamic_cast<TrackletProjectionsMemory*>(memory);
    assert(tmp != nullptr);
    inputprojs_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " could not find input: " << input;
}

void MatchProcessor::execute(unsigned int iSector, double phimin) {
  assert(vmstubs_.size() == 1);

  /*
    The code is organized in three 'steps' corresponding to the PR, ME, and MC functions. The output from
    the PR step is buffered in a 'circular' buffer, and similarly the ME output is put in a circular buffer.     
    The implementation is done in steps, emulating what can be done in firmware. On each step we do:
    
    1) A projection is read and if there is space it is insert into the inputProjBuffer_
    
    2) Process next match in the ME - if there is an idle ME the next projection is inserted
    
    3) Readout match from ME and send to match calculator

    However, for the pipelining to work in HLS these blocks are executed in reverse order
    
  */

  bool print = getName() == "MP_L3PHIC" && iSector == 3;
  print = false;

  phimin_ = phimin;

  Tracklet* oldTracklet = nullptr;

  unsigned int countme = 0;
  unsigned int countall = 0;
  unsigned int countsel = 0;
  unsigned int countinputproj = 0;

  unsigned int iprojmem = 0;
  while (iprojmem < inputprojs_.size() && inputprojs_[iprojmem]->nTracklets() == 0) {
    iprojmem++;
  }

  unsigned int iproj = 0;

  inputProjBuffer_.reset();

  for (const auto& inputproj : inputprojs_) {
    countinputproj += inputproj->nTracklets();
  }

  for (auto& matchengine : matchengines_) {
    matchengine.reset();
  }

  ProjectionTemp tmpProj_, tmpProj__;
  bool good_ = false;
  bool good__ = false;

  for (unsigned int istep = 0; istep < settings_.maxStep("MP"); istep++) {
    // This print statement is useful for detailed comparison with the HLS code
    // It prints out detailed status information for each clock step
    /* 
    if (print) {
      cout << "istep = "<<istep<<" projBuff: "<<inputProjBuffer_.rptr()<<" "<<inputProjBuffer_.wptr()<<" "<<projBuffNearFull;
      unsigned int iMEU = 0;
      for (auto& matchengine : matchengines_) {
	cout <<" MEU"<<iMEU<<": "<<matchengine.rptr()<<" "<<matchengine.wptr()
	     <<" "<<matchengine.idle()<<" "<<matchengine.empty()
	     <<" "<<matchengine.TCID();
	iMEU++;
      }
      cout << std::endl;
    }
    */

    //First do some caching of state at the start of the clock

    bool projdone = false;

    bool projBuffNearFull = inputProjBuffer_.nearfull();

    for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
      matchengines_[iME].setAlmostFull();
    }

    //Step 3
    //Check if we have candidate match to process

    unsigned int iMEbest = 0;
    int bestTCID = matchengines_[0].TCID();
    bool meactive = matchengines_[0].active();
    for (unsigned int iME = 1; iME < nMatchEngines_; iME++) {
      meactive = meactive || matchengines_[iME].active();
      int tcid = matchengines_[iME].TCID();
      if (tcid < bestTCID) {
        bestTCID = tcid;
        iMEbest = iME;
      }
    }

    // check if the matche engine processing the smallest tcid has match

    if (!matchengines_[iMEbest].empty()) {
      std::pair<Tracklet*, const Stub*> candmatch = matchengines_[iMEbest].read();

      const Stub* fpgastub = candmatch.second;
      Tracklet* tracklet = candmatch.first;

      //Consistency check
      if (oldTracklet != nullptr) {
        //allow equal here since we can have more than one cadidate match per tracklet projection
        //cout << "old new : "<<oldTracklet->TCID()<<" "<<tracklet->TCID()<<" "<<iMEbest<<endl;
        assert(oldTracklet->TCID() <= tracklet->TCID());
      }
      oldTracklet = tracklet;

      bool match = matchCalculator(tracklet, fpgastub, print, istep);

      if (settings_.debugTracklet() && match) {
        edm::LogVerbatim("Tracklet") << getName() << " have match";
      }

      countall++;
      if (match)
        countsel++;
    }

    //Step 2
    //Check if we have ME that can process projection

    bool addedProjection = false;
    for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
      if (!matchengines_[iME].idle())
        countme++;
      //if match engine empty and we have queued projections add to match engine
      if ((!addedProjection) && matchengines_[iME].idle() && (!inputProjBuffer_.empty())) {
        ProjectionTemp tmpProj = inputProjBuffer_.read();
        VMStubsMEMemory* stubmem = vmstubs_[0];

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " adding projection to match engine";
        }

        int nbins = (1 << N_RZBITS);
        if (layerdisk_ >= N_LAYER) {
          nbins *= 2;  //twice as many bins in disks (since there are two disks)
        }

        matchengines_[iME].init(stubmem,
                                nbins,
                                tmpProj.slot(),
                                tmpProj.iphi(),
                                tmpProj.shift(),
                                tmpProj.projrinv(),
                                tmpProj.projfinerz(),
                                tmpProj.projfinephi(),
                                tmpProj.use(0, 0),
                                tmpProj.use(0, 1),
                                tmpProj.use(1, 0),
                                tmpProj.use(1, 1),
                                tmpProj.isPSseed(),
                                tmpProj.proj());
        meactive = true;
        addedProjection = true;
      } else {
        matchengines_[iME].step();
      }
      matchengines_[iME].processPipeline();
    }

    //Step 1
    //First step here checks if we have more input projections to put into
    //the input puffer for projections

    if (good__) {
      inputProjBuffer_.store(tmpProj__);
    }

    good__ = good_;
    tmpProj__ = tmpProj_;

    good_ = false;

    if (iprojmem < inputprojs_.size()) {
      TrackletProjectionsMemory* projMem = inputprojs_[iprojmem];
      if (!projBuffNearFull) {
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " have projection in memory : " << projMem->getName();
        }

        Tracklet* proj = projMem->getTracklet(iproj);

        FPGAWord fpgaphi = proj->proj(layerdisk_).fpgaphiproj();

        unsigned int iphi = (fpgaphi.value() >> (fpgaphi.nbits() - nvmbits_)) & (nvmbins_ - 1);

        int nextrabits = 2;
        int overlapbits = nvmbits_ + nextrabits;

        unsigned int extrabits = fpgaphi.bits(fpgaphi.nbits() - overlapbits - nextrabits, nextrabits);

        unsigned int ivmPlus = iphi;

        int shift = 0;

        if (extrabits == ((1U << nextrabits) - 1) && iphi != ((1U << settings_.nbitsvmme(layerdisk_)) - 1)) {
          shift = 1;
          ivmPlus++;
        }
        unsigned int ivmMinus = iphi;
        if (extrabits == 0 && iphi != 0) {
          shift = -1;
          ivmMinus--;
        }

        int projrinv = -1;
        if (barrel_) {
          FPGAWord phider = proj->proj(layerdisk_).fpgaphiprojder();
          projrinv = (1 << (nrinv_ - 1)) - 1 - (phider.value() >> (phider.nbits() - nrinv_));
        } else {
          //The next lines looks up the predicted bend based on:
          // 1 - r projections
          // 2 - phi derivative
          // 3 - the sign - i.e. if track is forward or backward

          int rindex =
              (proj->proj(layerdisk_).fpgarzproj().value() >> (proj->proj(layerdisk_).fpgarzproj().nbits() - nrbits_)) &
              ((1 << nrbits_) - 1);

          int phiprojder = proj->proj(layerdisk_).fpgaphiprojder().value();

          int phiderindex = (phiprojder >> (proj->proj(layerdisk_).fpgaphiprojder().nbits() - nphiderbits_)) &
                            ((1 << nphiderbits_) - 1);

          int signindex = proj->proj(layerdisk_).fpgarzprojder().value() < 0;

          int bendindex = (signindex << (nphiderbits_ + nrbits_)) + (rindex << (nphiderbits_)) + phiderindex;

          projrinv = rinvbendlut_.lookup(bendindex);

          proj->proj(layerdisk_).setBendIndex(projrinv);
        }
        assert(projrinv >= 0);

        unsigned int slot = proj->proj(layerdisk_).fpgarzbin1projvm().value();
        bool second = proj->proj(layerdisk_).fpgarzbin2projvm().value();

        unsigned int projfinephi =
            (fpgaphi.value() >> (fpgaphi.nbits() - (nvmbits_ + NFINEPHIBITS))) & ((1 << NFINEPHIBITS) - 1);
        int projfinerz = proj->proj(layerdisk_).fpgafinerzvm().value();

        bool isPSseed = proj->PSseed();

        int nbins = (1 << N_RZBITS);
        if (layerdisk_ >= N_LAYER) {
          nbins *= 2;  //twice as many bins in disks (since there are two disks)
        }

        VMStubsMEMemory* stubmem = vmstubs_[0];
        bool usefirstMinus = stubmem->nStubsBin(ivmMinus * nbins + slot) != 0;
        bool usesecondMinus = (second && (stubmem->nStubsBin(ivmMinus * nbins + slot + 1) != 0));
        bool usefirstPlus = ivmPlus != ivmMinus && stubmem->nStubsBin(ivmPlus * nbins + slot) != 0;
        bool usesecondPlus = ivmPlus != ivmMinus && (second && (stubmem->nStubsBin(ivmPlus * nbins + slot + 1) != 0));

        good_ = usefirstPlus || usesecondPlus || usefirstMinus || usesecondMinus;

        if (good_) {
          ProjectionTemp tmpProj(proj,
                                 slot,
                                 projrinv,
                                 projfinerz,
                                 projfinephi,
                                 ivmMinus,
                                 shift,
                                 usefirstMinus,
                                 usefirstPlus,
                                 usesecondMinus,
                                 usesecondPlus,
                                 isPSseed);
          tmpProj_ = tmpProj;
        }

        iproj++;
        if (iproj == projMem->nTracklets()) {
          iproj = 0;
          do {
            iprojmem++;
          } while (iprojmem < inputprojs_.size() && inputprojs_[iprojmem]->nTracklets() == 0);
        }

      } else {
        projdone = true && !good_ && !good__;
      }
    }

    //
    //  Check if done
    //
    //
    //

    if ((projdone && !meactive && inputProjBuffer_.rptr() == inputProjBuffer_.wptr()) ||
        (istep == settings_.maxStep("MP") - 1)) {
      if (settings_.writeMonitorData("MP")) {
        globals_->ofstream("matchprocessor.txt") << getName() << " " << istep << " " << countall << " " << countsel
                                                 << " " << countme << " " << countinputproj << endl;
      }
      break;
    }
  }

  if (settings_.writeMonitorData("MC")) {
    globals_->ofstream("matchcalculator.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

bool MatchProcessor::matchCalculator(Tracklet* tracklet, const Stub* fpgastub, bool, unsigned int) {
  const L1TStub* stub = fpgastub->l1tstub();

  if (layerdisk_ < N_LAYER) {
    const Projection& proj = tracklet->proj(layerdisk_);
    int ir = fpgastub->r().value();
    int iphi = proj.fpgaphiproj().value();
    int icorr = (ir * proj.fpgaphiprojder().value()) >> icorrshift_;
    iphi += icorr;

    int iz = proj.fpgarzproj().value();
    int izcor = (ir * proj.fpgarzprojder().value() + (1 << (icorzshift_ - 1))) >> icorzshift_;
    iz += izcor;

    int ideltaz = fpgastub->z().value() - iz;
    int ideltaphi = (fpgastub->phi().value() - iphi) << phishift_;

    //Floating point calculations

    double phi = stub->phi();
    double r = stub->r();
    double z = stub->z();

    if (settings_.useapprox()) {
      double dphi = reco::reduceRange(phi - fpgastub->phiapprox(phimin_, 0.0));
      assert(std::abs(dphi) < 0.001);
      phi = fpgastub->phiapprox(phimin_, 0.0);
      z = fpgastub->zapprox();
      r = fpgastub->rapprox();
    }

    if (phi < 0)
      phi += 2 * M_PI;
    phi -= phimin_;

    double dr = r - settings_.rmean(layerdisk_);
    assert(std::abs(dr) < settings_.drmax());

    double dphi = reco::reduceRange(phi - (proj.phiproj() + dr * proj.phiprojder()));

    double dz = z - (proj.rzproj() + dr * proj.rzprojder());

    double dphiapprox = reco::reduceRange(phi - (proj.phiprojapprox() + dr * proj.phiprojderapprox()));

    double dzapprox = z - (proj.rzprojapprox() + dr * proj.rzprojderapprox());

    int seedindex = tracklet->getISeed();

    assert(phimatchcuttable_.lookup(seedindex) > 0);
    assert(zmatchcuttable_.lookup(seedindex) > 0);

    if (settings_.bookHistos()) {
      bool truthmatch = tracklet->stubtruthmatch(stub);

      HistBase* hists = globals_->histograms();
      hists->FillLayerResidual(layerdisk_ + 1,
                               seedindex,
                               dphiapprox * settings_.rmean(layerdisk_),
                               ideltaphi * settings_.kphi1() * settings_.rmean(layerdisk_),
                               (ideltaz << dzshift_) * settings_.kz(),
                               dz,
                               truthmatch);
    }

    if (settings_.writeMonitorData("Residuals")) {
      double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

      globals_->ofstream("layerresiduals.txt")
          << layerdisk_ + 1 << " " << seedindex << " " << pt << " "
          << ideltaphi * settings_.kphi1() * settings_.rmean(layerdisk_) << " "
          << dphiapprox * settings_.rmean(layerdisk_) << " "
          << phimatchcuttable_.lookup(seedindex) * settings_.kphi1() * settings_.rmean(layerdisk_) << "   "
          << (ideltaz << dzshift_) * settings_.kz() << " " << dz << " "
          << zmatchcuttable_.lookup(seedindex) * settings_.kz() << endl;
    }

    bool imatch = (std::abs(ideltaphi) <= phimatchcuttable_.lookup(seedindex)) &&
                  (ideltaz << dzshift_ < zmatchcuttable_.lookup(seedindex)) &&
                  (ideltaz << dzshift_ >= -zmatchcuttable_.lookup(seedindex));

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " imatch = " << imatch << " ideltaphi cut " << ideltaphi << " "
                                   << phimatchcuttable_.lookup(seedindex) << " ideltaz<<dzshift cut "
                                   << (ideltaz << dzshift_) << " " << zmatchcuttable_.lookup(seedindex);
    }

    //This would catch significant consistency problems in the configuration - helps to debug if there are problems.
    if (std::abs(dphi) > 0.5 * settings_.dphisectorHG() || std::abs(dphiapprox) > 0.5 * settings_.dphisectorHG()) {
      throw cms::Exception("LogicError") << "WARNING dphi and/or dphiapprox too large : " << dphi << " " << dphiapprox
                                         << endl;
    }

    bool keep = true;
    if (!settings_.doKF() || !settings_.doMultipleMatches()) {
      // Case of allowing only one stub per track per layer (or no KF which implies the same).
      if (imatch && tracklet->match(layerdisk_)) {
        // Veto match if is not the best one for this tracklet (in given layer)
        auto res = tracklet->resid(layerdisk_);
        keep = abs(ideltaphi) < abs(res.fpgaphiresid().value());
        imatch = keep;
      }
    }

    if (imatch) {
      tracklet->addMatch(layerdisk_,
                         ideltaphi,
                         ideltaz,
                         dphi,
                         dz,
                         dphiapprox,
                         dzapprox,
                         (phiregion_ << N_BITSMEMADDRESS) + fpgastub->stubindex().value(),
                         fpgastub);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "Accepted full match in layer " << getName() << " " << tracklet;
      }

      int iSeed = tracklet->getISeed();
      assert(fullmatches_[iSeed] != nullptr);
      fullmatches_[iSeed]->addMatch(tracklet, fpgastub);

      return true;
    } else {
      return false;
    }
  } else {  //disk matches

    //check that stubs and projections in same half of detector
    assert(stub->z() * tracklet->t() > 0.0);

    int sign = (tracklet->t() > 0.0) ? 1 : -1;
    int disk = sign * (layerdisk_ - N_LAYER + 1);
    assert(disk != 0);

    //Perform integer calculations here

    int iz = fpgastub->z().value();

    const Projection& proj = tracklet->proj(layerdisk_);

    int iphi = proj.fpgaphiproj().value();
    int iphicorr = (iz * proj.fpgaphiprojder().value()) >> icorrshift_;

    iphi += iphicorr;

    int ir = proj.fpgarzproj().value();
    int ircorr = (iz * proj.fpgarzprojder().value()) >> icorzshift_;
    ir += ircorr;

    int ideltaphi = fpgastub->phi().value() - iphi;

    int irstub = fpgastub->r().value();
    int ialphafact = 0;
    if (!stub->isPSmodule()) {
      assert(irstub < (int)N_DSS_MOD * 2);
      if (layerdisk_ - N_LAYER <= 1) {
        ialphafact = ialphafactinner_[irstub];
        irstub = settings_.rDSSinner(irstub) / settings_.kr();
      } else {
        ialphafact = ialphafactouter_[irstub];
        irstub = settings_.rDSSouter(irstub) / settings_.kr();
      }
    }

    int ideltar = (irstub * settings_.kr()) / settings_.krprojshiftdisk() - ir;

    if (!stub->isPSmodule()) {
      int ialpha = fpgastub->alpha().value();
      int iphialphacor = ((ideltar * ialpha * ialphafact) >> settings_.alphashift());
      ideltaphi += iphialphacor;
    }

    //Perform floating point calculations here

    double phi = stub->phi();
    double z = stub->z();
    double r = stub->r();

    if (settings_.useapprox()) {
      double dphi = reco::reduceRange(phi - fpgastub->phiapprox(phimin_, 0.0));
      assert(std::abs(dphi) < 0.001);
      phi = fpgastub->phiapprox(phimin_, 0.0);
      z = fpgastub->zapprox();
      r = fpgastub->rapprox();
    }

    if (phi < 0)
      phi += 2 * M_PI;
    phi -= phimin_;

    double dz = z - sign * settings_.zmean(layerdisk_ - N_LAYER);

    if (std::abs(dz) > settings_.dzmax()) {
      edm::LogProblem("Tracklet") << __FILE__ << ":" << __LINE__ << " " << name_ << " " << tracklet->getISeed();
      edm::LogProblem("Tracklet") << "stub " << stub->z() << " disk " << disk << " " << dz;
      assert(std::abs(dz) < settings_.dzmax());
    }

    double phiproj = proj.phiproj() + dz * proj.phiprojder();
    double rproj = proj.rzproj() + dz * proj.rzprojder();
    double deltar = r - rproj;

    double dr = stub->r() - rproj;
    double drapprox = stub->r() - (proj.rzprojapprox() + dz * proj.rzprojderapprox());

    double dphi = reco::reduceRange(phi - phiproj);

    double dphiapprox = reco::reduceRange(phi - (proj.phiprojapprox() + dz * proj.phiprojderapprox()));

    double drphi = dphi * stub->r();
    double drphiapprox = dphiapprox * stub->r();

    if (!stub->isPSmodule()) {
      double alphanorm = stub->alphanorm();
      dphi += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r2();
      ;
      dphiapprox += drapprox * alphanorm * settings_.half2SmoduleWidth() / stub->r2();

      drphi += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r();
      drphiapprox += dr * alphanorm * settings_.half2SmoduleWidth() / stub->r();
    }

    int seedindex = tracklet->getISeed();

    int idrphicut = rphicutPStable_.lookup(seedindex);
    int idrcut = rcutPStable_.lookup(seedindex);
    if (!stub->isPSmodule()) {
      idrphicut = rphicut2Stable_.lookup(seedindex);
      idrcut = rcut2Stable_.lookup(seedindex);
    }

    double drphicut = idrphicut * settings_.kphi() * settings_.kr();
    double drcut = idrcut * settings_.krprojshiftdisk();

    if (settings_.writeMonitorData("Residuals")) {
      double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

      globals_->ofstream("diskresiduals.txt")
          << layerdisk_ - N_LAYER + 1 << " " << stub->isPSmodule() << " " << tracklet->layer() << " "
          << abs(tracklet->disk()) << " " << pt << " " << ideltaphi * settings_.kphi() * stub->r() << " " << drphiapprox
          << " " << drphicut << " " << ideltar * settings_.krprojshiftdisk() << " " << deltar << " " << drcut << " "
          << endl;
    }

    bool match = (std::abs(drphi) < drphicut) && (std::abs(deltar) < drcut);
    bool imatch = (std::abs(ideltaphi * irstub) < idrphicut) && (std::abs(ideltar) < idrcut);

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "imatch match disk: " << imatch << " " << match << " " << std::abs(ideltaphi)
                                   << " " << drphicut / (settings_.kphi() * stub->r()) << " " << std::abs(ideltar)
                                   << " " << drcut / settings_.krprojshiftdisk() << " r = " << stub->r();
    }

    bool keep = true;
    if (!settings_.doKF() || !settings_.doMultipleMatches()) {
      // Case of allowing only one stub per track per layer (or no KF which implies the same).
      if (imatch && tracklet->match(layerdisk_)) {
        // Veto match if is not the best one for this tracklet (in given layer)
        auto res = tracklet->resid(layerdisk_);
        keep = abs(ideltaphi) < abs(res.fpgaphiresid().value());
        imatch = keep;
      }
    }

    if (imatch) {
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "MatchCalculator found match in disk " << getName();
      }

      if (std::abs(dphi) >= third * settings_.dphisectorHG()) {
        edm::LogPrint("Tracklet") << "dphi " << dphi << " ISeed " << tracklet->getISeed();
      }
      assert(std::abs(dphi) < third * settings_.dphisectorHG());
      assert(std::abs(dphiapprox) < third * settings_.dphisectorHG());

      tracklet->addMatch(layerdisk_,
                         ideltaphi,
                         ideltar,
                         drphi / stub->r(),
                         dr,
                         drphiapprox / stub->r(),
                         drapprox,
                         (phiregion_ << N_BITSMEMADDRESS) + fpgastub->stubindex().value(),
                         fpgastub);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "Accepted full match in disk " << getName() << " " << tracklet;
      }

      int iSeed = tracklet->getISeed();
      assert(fullmatches_[iSeed] != nullptr);
      fullmatches_[iSeed]->addMatch(tracklet, fpgastub);

      return true;
    } else {
      return false;
    }
  }
}
