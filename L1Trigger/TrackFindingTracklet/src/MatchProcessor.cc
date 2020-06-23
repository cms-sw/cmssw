#include "L1Trigger/TrackFindingTracklet/interface/MatchProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/ProjectionRouterBendTable.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

using namespace std;
using namespace trklet;

MatchProcessor::MatchProcessor(string name, Settings const& settings, Globals* global, unsigned int iSector)
    : ProcessBase(name, settings, global, iSector), fullmatches_(12), inputProjBuffer_(3) {
  phioffset_ = phimin_;

  phiregion_ = name[8] - 'A';

  initLayerDisk(3, layer_, disk_);

  //TODO should sort out constants here
  icorrshift_ = 7;

  if (layer_ <= 3) {
    icorzshift_ = -1 - settings_.PS_zderL_shift();
  } else {
    icorzshift_ = -1 - settings_.SS_zderL_shift();
  }
  phi0shift_ = 3;
  fact_ = 1;
  if (layer_ >= 4) {
    fact_ = (1 << (settings_.nzbitsstub(0) - settings_.nzbitsstub(5)));
    icorrshift_ -= (10 - settings_.nrbitsstub(layer_ - 1));
    icorzshift_ += (settings_.nzbitsstub(0) - settings_.nzbitsstub(5) + settings_.nrbitsstub(layer_ - 1) -
                    settings_.nrbitsstub(0));
    phi0shift_ = 0;
  }

  nrbits_ = 5;
  nphiderbits_ = 6;

  //to adjust globaly the phi and rz matching cuts
  phifact_ = 1.0;
  rzfact_ = 1.0;

  for (unsigned int iSeed = 0; iSeed < 12; iSeed++) {
    if (layer_ > 0) {
      phimatchcut_[iSeed] =
          settings_.rphimatchcut(iSeed, layer_ - 1) / (settings_.kphi1() * settings_.rmean(layer_ - 1));
      zmatchcut_[iSeed] = settings_.zmatchcut(iSeed, layer_ - 1) / settings_.kz();
    }
    if (disk_ != 0) {
      rphicutPS_[iSeed] = settings_.rphicutPS(iSeed, abs(disk_) - 1) / (settings_.kphi() * settings_.kr());
      rphicut2S_[iSeed] = settings_.rphicut2S(iSeed, abs(disk_) - 1) / (settings_.kphi() * settings_.kr());
      rcut2S_[iSeed] = settings_.rcut2S(iSeed, abs(disk_) - 1) / settings_.krprojshiftdisk();
      rcutPS_[iSeed] = settings_.rcutPS(iSeed, abs(disk_) - 1) / settings_.krprojshiftdisk();
    }
  }

  if (iSector_ == 0 && layer_ > 0 && settings_.writeTable()) {
    ofstream outphicut;
    outphicut.open(getName() + "_phicut.tab");
    outphicut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < 12; seedindex++) {
      if (seedindex != 0)
        outphicut << "," << endl;
      outphicut << phimatchcut_[seedindex];
    }
    outphicut << endl << "};" << endl;
    outphicut.close();

    ofstream outzcut;
    outzcut.open(getName() + "_zcut.tab");
    outzcut << "{" << endl;
    for (unsigned int seedindex = 0; seedindex < N_SEED; seedindex++) {
      if (seedindex != 0)
        outzcut << "," << endl;
      outzcut << zmatchcut_[seedindex];
    }
    outzcut << endl << "};" << endl;
    outzcut.close();
  }

  if (layer_ > 0) {
    unsigned int nbits = 3;
    if (layer_ >= 4)
      nbits = 4;

    for (unsigned int irinv = 0; irinv < 32; irinv++) {
      double rinv = (irinv - 15.5) * (1 << (settings_.nbitsrinv() - 5)) * settings_.krinvpars();
      double stripPitch =
          (settings_.rmean(layer_ - 1) < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
      double projbend = bend(settings_.rmean(layer_ - 1), rinv, stripPitch);
      for (unsigned int ibend = 0; ibend < (unsigned int)(1 << nbits); ibend++) {
        double stubbend = benddecode(ibend, layer_ <= (int)N_PSLAYER);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(layer_ - 1);
        table_.push_back(pass);
      }
    }

    if (settings_.writeTable()) {
      ofstream out;
      char layer = '0' + layer_;
      string fname = "METable_L";
      fname += layer;
      fname += ".tab";
      out.open(fname.c_str());
      out << "{" << endl;
      for (unsigned int i = 0; i < table_.size(); i++) {
        if (i != 0) {
          out << "," << endl;
        }
        out << table_[i];
      }
      out << "};" << endl;
      out.close();
    }
  }

  if (disk_ > 0) {
    for (unsigned int iprojbend = 0; iprojbend < 32; iprojbend++) {
      double projbend = 0.5 * (iprojbend - 15.0);
      for (unsigned int ibend = 0; ibend < 8; ibend++) {
        double stubbend = benddecode(ibend, true);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(disk_ + 5);
        tablePS_.push_back(pass);
      }
      for (unsigned int ibend = 0; ibend < 16; ibend++) {
        double stubbend = benddecode(ibend, false);
        bool pass = std::abs(stubbend - projbend) < settings_.bendcutme(disk_ + 5);
        table2S_.push_back(pass);
      }
    }
  }

  for (unsigned int i = 0; i < N_DSS_MOD * 2; i++) {
    ialphafactinner_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSinner(i) * settings_.rDSSinner(i)) /
                          settings_.kphi();
    ialphafactouter_[i] = (1 << settings_.alphashift()) * settings_.krprojshiftdisk() * settings_.half2SmoduleWidth() /
                          (1 << (settings_.nbitsalpha() - 1)) / (settings_.rDSSouter(i) * settings_.rDSSouter(i)) /
                          settings_.kphi();
  }

  barrel_ = layer_ > 0;

  nvm_ = barrel_ ? settings_.nvmme(layer_ - 1) * settings_.nallstubs(layer_ - 1)
                 : settings_.nvmme(disk_ + 5) * settings_.nallstubs(disk_ + 5);
  nvmbins_ = barrel_ ? settings_.nvmme(layer_ - 1) : settings_.nvmme(disk_ + 5);

  if (nvm_ == 32)
    nvmbits_ = 5;
  if (nvm_ == 16)
    nvmbits_ = 4;
  assert(nvmbits_ != -1);

  nMatchEngines_ = 4;
  for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
    MatchEngineUnit tmpME(barrel_, table_, tablePS_, table2S_);
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

void MatchProcessor::execute() {
  if (globals_->projectionRouterBendTable() == nullptr) {  // move to constructor?!
    auto* bendTablePtr = new ProjectionRouterBendTable();
    bendTablePtr->init(settings_, globals_, nrbits_, nphiderbits_);
    globals_->projectionRouterBendTable() = bendTablePtr;
  }

  /*
    The code is organized in three 'steps' corresponding to the PR, ME, and MC functions. The output from
    the PR step is buffered in a 'circular' buffer, and similarly the ME output is put in a circular buffer. 
    
    The implementation is done in steps, emulating what can be done in firmware. One each step we do:
    
    1) A projection is read and if there is space it is insert into the inputProjBuffer_
    
    2) Process next match in the ME - if there is an idle ME the next projection is inserted
    
    3) Readout match from ME and send to match calculator
    
  */

  Tracklet* oldTracklet = nullptr;

  unsigned int countall = 0;
  unsigned int countsel = 0;

  unsigned int iprojmem = 0;
  unsigned int iproj = 0;

  inputProjBuffer_.reset();

  for (unsigned int istep = 0; istep < settings_.maxStep("MP"); istep++) {
    //Step 1
    //First step here checks if we have more input projections to put into
    //the input puffer for projections
    if (iprojmem < inputprojs_.size()) {
      TrackletProjectionsMemory* projMem = inputprojs_[iprojmem];
      if (projMem->nTracklets() == 0) {
        iprojmem++;
      } else if (iproj < projMem->nTracklets()) {
        if (!inputProjBuffer_.almostfull()) {
          Tracklet* proj = projMem->getTracklet(iproj);

          FPGAWord fpgaphi = barrel_ ? proj->fpgaphiproj(layer_) : proj->fpgaphiprojdisk(disk_);

          int iphi = (fpgaphi.value() >> (fpgaphi.nbits() - nvmbits_)) & (nvmbins_ - 1);

          int projrinv = -1;
          if (barrel_) {
            projrinv = 16 + (proj->fpgarinv().value() >> (proj->fpgarinv().nbits() - 5));
          } else {
            //The next lines looks up the predicted bend based on:
            // 1 - r projections
            // 2 - phi derivative
            // 3 - the sign - i.e. if track is forward or backward
            int rindex = (proj->fpgarprojdisk(disk_).value() >> (proj->fpgarprojdisk(disk_).nbits() - nrbits_)) &
                         ((1 << nrbits_) - 1);

            int phiderindex =
                (proj->fpgaphiprojderdisk(disk_).value() >> (proj->fpgaphiprojderdisk(disk_).nbits() - nphiderbits_)) &
                ((1 << nphiderbits_) - 1);

            int signindex = (proj->fpgarprojderdisk(disk_).value() < 0);

            int bendindex = (signindex << (nphiderbits_ + nrbits_)) + (rindex << (nphiderbits_)) + phiderindex;

            projrinv = globals_->projectionRouterBendTable()->bendLoookup(abs(disk_) - 1, bendindex);

            proj->setBendIndex(projrinv, disk_);
          }
          assert(projrinv >= 0);

          unsigned int slot = barrel_ ? proj->zbin1projvm(layer_) : proj->rbin1projvm(disk_);
          bool second = (barrel_ ? proj->zbin2projvm(layer_) : proj->rbin2projvm(disk_)) == 1;

          unsigned int projfinephi = fpgaphi.value() >> (fpgaphi.nbits() - (nvmbits_ + 3)) & 7;
          int projfinerz = barrel_ ? proj->finezvm(layer_) : proj->finervm(disk_);

          bool isPSseed = proj->PSseed() == 1;

          VMStubsMEMemory* stubmem = vmstubs_[iphi];
          if (stubmem->nStubsBin(slot) != 0) {
            ProjectionTemp tmpProj(proj, slot, projrinv, projfinerz, projfinephi, iphi, isPSseed);
            inputProjBuffer_.store(tmpProj);
          }
          if (second && (stubmem->nStubsBin(slot + 1) != 0)) {
            ProjectionTemp tmpProj(proj, slot + 1, projrinv, projfinerz - 8, projfinephi, iphi, isPSseed);
            inputProjBuffer_.store(tmpProj);
          }
          iproj++;
          if (iproj == projMem->nTracklets()) {
            iproj = 0;
            iprojmem++;
          }
        }
      }
    }

    //Step 2
    //Check if we have ME that can process projection

    bool addedProjection = false;
    for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
      matchengines_[iME].step();
      //if match engine empty and we have queued projections add to match engine
      if ((!addedProjection) && matchengines_[iME].idle() && (!inputProjBuffer_.empty())) {
        ProjectionTemp tmpProj = inputProjBuffer_.read();
        VMStubsMEMemory* stubmem = vmstubs_[tmpProj.iphi()];

        matchengines_[iME].init(stubmem,
                                tmpProj.slot(),
                                tmpProj.projrinv(),
                                tmpProj.projfinerz(),
                                tmpProj.projfinephi(),
                                tmpProj.isPSseed(),
                                tmpProj.proj());
        addedProjection = true;
      }
    }

    //Step 3
    //Check if we have candidate match to process

    unsigned int iMEbest = nMatchEngines_;
    int bestTCID = -1;
    bool bestInPipeline = false;
    for (unsigned int iME = 0; iME < nMatchEngines_; iME++) {
      bool empty = matchengines_[iME].empty();
      if (empty && matchengines_[iME].idle())
        continue;
      int currentTCID = empty ? matchengines_[iME].currentProj()->TCID() : matchengines_[iME].peek().first->TCID();
      if ((iMEbest == nMatchEngines_) || (currentTCID < bestTCID)) {
        iMEbest = iME;
        bestTCID = currentTCID;
        bestInPipeline = empty;
      }
    }

    if (iMEbest != nMatchEngines_ && (!bestInPipeline)) {
      std::pair<Tracklet*, const Stub*> candmatch = matchengines_[iMEbest].read();

      const Stub* fpgastub = candmatch.second;
      Tracklet* tracklet = candmatch.first;

      if (oldTracklet != nullptr) {
        //allow equal here since we can have more than one cadidate match per tracklet projection
        assert(oldTracklet->TCID() <= tracklet->TCID());
      }
      oldTracklet = tracklet;

      bool match = matchCalculator(tracklet, fpgastub);

      countall++;
      if (match)
        countsel++;
      ;
    }
  }

  if (settings_.writeMonitorData("MC")) {
    globals_->ofstream("matchcalculator.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

bool MatchProcessor::matchCalculator(Tracklet* tracklet, const Stub* fpgastub) {
  const L1TStub* stub = fpgastub->l1tstub();

  if (layer_ != 0) {
    int ir = fpgastub->r().value();
    int iphi = tracklet->fpgaphiproj(layer_).value();
    int icorr = (ir * tracklet->fpgaphiprojder(layer_).value()) >> icorrshift_;
    iphi += icorr;

    int iz = tracklet->fpgazproj(layer_).value();
    int izcor = (ir * tracklet->fpgazprojder(layer_).value() + (1 << (icorzshift_ - 1))) >> icorzshift_;
    iz += izcor;

    int ideltaz = fpgastub->z().value() - iz;
    int ideltaphi = (fpgastub->phi().value() << phi0shift_) - (iphi << (settings_.phi0bitshift() - 1 + phi0shift_));

    //Floating point calculations

    double phi = stub->phi();
    double r = stub->r();
    double z = stub->z();

    if (settings_.useapprox()) {
      double dphi = reco::reduceRange(phi - fpgastub->phiapprox(phimin_, phimax_));
      assert(std::abs(dphi) < 0.001);
      phi = fpgastub->phiapprox(phimin_, phimax_);
      z = fpgastub->zapprox();
      r = fpgastub->rapprox();
    }

    if (phi < 0)
      phi += 2 * M_PI;
    phi -= phioffset_;

    double dr = r - tracklet->rproj(layer_);
    assert(std::abs(dr) < settings_.drmax());

    double dphi = reco::reduceRange(phi - (tracklet->phiproj(layer_) + dr * tracklet->phiprojder(layer_)));

    double dz = z - (tracklet->zproj(layer_) + dr * tracklet->zprojder(layer_));

    double dphiapprox =
        reco::reduceRange(phi - (tracklet->phiprojapprox(layer_) + dr * tracklet->phiprojderapprox(layer_)));

    double dzapprox = z - (tracklet->zprojapprox(layer_) + dr * tracklet->zprojderapprox(layer_));

    int seedindex = tracklet->getISeed();

    assert(phimatchcut_[seedindex] > 0);
    assert(zmatchcut_[seedindex] > 0);

    if (settings_.bookHistos()) {
      bool truthmatch = tracklet->stubtruthmatch(stub);

      HistBase* hists = globals_->histograms();
      hists->FillLayerResidual(layer_,
                               seedindex,
                               dphiapprox * settings_.rmean(layer_ - 1),
                               ideltaphi * settings_.kphi1() * settings_.rmean(layer_ - 1),
                               ideltaz * fact_ * settings_.kz(),
                               dz,
                               truthmatch);
    }

    if (settings_.writeMonitorData("Residuals")) {
      double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

      globals_->ofstream("layerresiduals.txt")
          << layer_ << " " << seedindex << " " << pt << " "
          << ideltaphi * settings_.kphi1() * settings_.rmean(layer_ - 1) << " "
          << dphiapprox * settings_.rmean(layer_ - 1) << " "
          << phimatchcut_[seedindex] * settings_.kphi1() * settings_.rmean(layer_ - 1) << "   "
          << ideltaz * fact_ * settings_.kz() << " " << dz << " " << zmatchcut_[seedindex] * settings_.kz() << endl;
    }

    bool imatch = (std::abs(ideltaphi) <= phifact_ * phimatchcut_[seedindex]) &&
                  (std::abs(ideltaz * fact_) <= rzfact_ * zmatchcut_[seedindex]);

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " imatch = " << imatch << " ideltaphi cut " << ideltaphi << " "
                                   << phimatchcut_[seedindex] << " ideltaz*fact cut " << ideltaz * fact_ << " "
                                   << zmatchcut_[seedindex];
    }

    if (std::abs(dphi) > 0.2 || std::abs(dphiapprox) > 0.2) {
      edm::LogPrint("Tracklet") << "WARNING dphi and/or dphiapprox too large : " << dphi << " " << dphiapprox;
    }

    assert(std::abs(dphi) < 0.2);
    assert(std::abs(dphiapprox) < 0.2);

    if (imatch) {
      tracklet->addMatch(layer_,
                         ideltaphi,
                         ideltaz,
                         dphi,
                         dz,
                         dphiapprox,
                         dzapprox,
                         (phiregion_ << 7) + fpgastub->stubindex().value(),
                         stub->r(),
                         fpgastub);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "Accepted full match in layer " << getName() << " " << tracklet << " "
                                     << iSector_;
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
    int disk = sign * disk_;
    assert(disk != 0);

    //Perform integer calculations here

    int iz = fpgastub->z().value();
    int iphi = tracklet->fpgaphiprojdisk(disk).value();

    int shifttmp = 6;  //TODO - express in terms of constants
    assert(shifttmp >= 0);
    int iphicorr = (iz * tracklet->fpgaphiprojderdisk(disk).value()) >> shifttmp;

    iphi += iphicorr;

    int ir = tracklet->fpgarprojdisk(disk).value();

    int shifttmp2 = 7;  //TODO - express in terms of constants
    assert(shifttmp2 >= 0);
    int ircorr = (iz * tracklet->fpgarprojderdisk(disk).value()) >> shifttmp2;

    ir += ircorr;

    int ideltaphi = fpgastub->phi().value() * settings_.kphi() / settings_.kphi() - iphi;

    int irstub = fpgastub->r().value();
    int ialphafact = 0;
    if (!stub->isPSmodule()) {
      assert(irstub < (int)N_DSS_MOD * 2);
      if (disk_ <= 2) {
        ialphafact = ialphafactinner_[irstub];
        irstub = settings_.rDSSinner(irstub) / settings_.kr();
      } else {
        ialphafact = ialphafactouter_[irstub];
        irstub = settings_.rDSSouter(irstub) / settings_.kr();
      }
    }

    int ideltar = (irstub * settings_.kr()) / settings_.krprojshiftdisk() - ir;

    if (!stub->isPSmodule()) {
      int ialphanew = fpgastub->alphanew().value();
      int iphialphacor = ((ideltar * ialphanew * ialphafact) >> settings_.alphashift());
      ideltaphi += iphialphacor;
    }

    //Perform floating point calculations here

    double phi = stub->phi();
    double z = stub->z();
    double r = stub->r();

    if (settings_.useapprox()) {
      double dphi = reco::reduceRange(phi - fpgastub->phiapprox(phimin_, phimax_));
      assert(std::abs(dphi) < 0.001);
      phi = fpgastub->phiapprox(phimin_, phimax_);
      z = fpgastub->zapprox();
      r = fpgastub->rapprox();
    }

    if (phi < 0)
      phi += 2 * M_PI;
    phi -= phioffset_;

    double dz = z - sign * settings_.zmean(disk_ - 1);

    if (std::abs(dz) > settings_.dzmax()) {
      edm::LogProblem("Tracklet") << __FILE__ << ":" << __LINE__ << " " << name_ << "_" << iSector_ << " "
                                  << tracklet->getISeed();
      edm::LogProblem("Tracklet") << "stub " << stub->z() << " disk " << disk << " " << dz;
      assert(std::abs(dz) < settings_.dzmax());
    }

    double phiproj = tracklet->phiprojdisk(disk) + dz * tracklet->phiprojderdisk(disk);
    double rproj = tracklet->rprojdisk(disk) + dz * tracklet->rprojderdisk(disk);
    double deltar = r - rproj;

    double dr = stub->r() - rproj;
    double drapprox = stub->r() - (tracklet->rprojapproxdisk(disk) + dz * tracklet->rprojderapproxdisk(disk));

    double dphi = reco::reduceRange(phi - phiproj);
    double dphiapprox =
        reco::reduceRange(phi - (tracklet->phiprojapproxdisk(disk) + dz * tracklet->phiprojderapproxdisk(disk)));

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

    int idrphicut = rphicutPS_[seedindex];
    int idrcut = rcutPS_[seedindex];
    if (!stub->isPSmodule()) {
      idrphicut = rphicut2S_[seedindex];
      idrcut = rcut2S_[seedindex];
    }

    double drphicut = idrphicut * settings_.kphi() * settings_.kr();
    double drcut = idrcut * settings_.krprojshiftdisk();

    if (settings_.writeMonitorData("Residuals")) {
      double pt = 0.01 * settings_.c() * settings_.bfield() / std::abs(tracklet->rinv());

      globals_->ofstream("diskresiduals.txt")
          << disk_ << " " << stub->isPSmodule() << " " << tracklet->layer() << " " << abs(tracklet->disk()) << " " << pt
          << " " << ideltaphi * settings_.kphi() * stub->r() << " " << drphiapprox << " " << drphicut << " "
          << ideltar * settings_.krprojshiftdisk() << " " << deltar << " " << drcut << " " << endl;
    }

    bool match = (std::abs(drphi) < drphicut) && (std::abs(deltar) < drcut);
    bool imatch = (std::abs(ideltaphi * irstub) < idrphicut) && (std::abs(ideltar) < idrcut);

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << "imatch match disk: " << imatch << " " << match << " " << std::abs(ideltaphi)
                                   << " " << drphicut / (settings_.kphi() * stub->r()) << " " << std::abs(ideltar)
                                   << " " << drcut / settings_.krprojshiftdisk() << " r = " << stub->r();
    }

    if (imatch) {
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "MatchCalculator found match in disk " << getName();
      }

      if (std::abs(dphi) >= 0.25) {
        edm::LogPrint("Tracklet") << "dphi " << dphi << " ISeed " << tracklet->getISeed();
      }
      assert(std::abs(dphi) < 0.25);
      assert(std::abs(dphiapprox) < 0.25);

      tracklet->addMatchDisk(disk,
                             ideltaphi,
                             ideltar,
                             drphi / stub->r(),
                             dr,
                             drphiapprox / stub->r(),
                             drapprox,
                             stub->alpha(settings_.stripPitch(stub->isPSmodule())),
                             (phiregion_ << 7) + fpgastub->stubindex().value(),
                             stub->z(),
                             fpgastub);
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "Accepted full match in disk " << getName() << " " << tracklet << " "
                                     << iSector_;
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
