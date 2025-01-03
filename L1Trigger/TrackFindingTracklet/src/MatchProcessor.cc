//////////////////////////////////////////////////////////////////
// MatchProcessor
//
// This module is the combines the old PR+ME+MC modules
// See more in execute()
//
// Variables such as `best_ideltaphi_barrel` store the "global"
// best value for delta phi, r, z, and r*phi, for instances
// where the same tracklet has multiple stub pairs. This allows
// us to find the truly best match
//////////////////////////////////////////////////////////////////

#include "L1Trigger/TrackFindingTracklet/interface/MatchProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/HistBase.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

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
      alphainner_(settings),
      alphaouter_(settings),
      rSSinner_(settings),
      rSSouter_(settings),
      diskRadius_(settings),
      fullmatches_(2),
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

  nrprojbits_ = 8;

  if (!barrel_) {
    rinvbendlut_.initProjectionBend(settings_.kphiderdisk(), layerdisk_ - N_LAYER, nrbits_, nphiderbits_);
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
    alphainner_.initmatchcut(layerdisk_, TrackletLUT::MatchType::alphainner, region);
    alphaouter_.initmatchcut(layerdisk_, TrackletLUT::MatchType::alphaouter, region);
    rSSinner_.initmatchcut(layerdisk_, TrackletLUT::MatchType::rSSinner, region);
    rSSouter_.initmatchcut(layerdisk_, TrackletLUT::MatchType::rSSouter, region);
    diskRadius_.initProjectionDiskRadius(nrprojbits_);
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

  // Pick some initial large values
  best_ideltaphi_barrel = 0xFFFF;
  best_ideltaz_barrel = 0xFFFF;
  best_ideltaphi_disk = 0xFFFF;
  best_ideltar_disk = 0xFFFF;
  curr_tracklet = nullptr;
  next_tracklet = nullptr;
}

void MatchProcessor::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output.find("matchout0") != std::string::npos) {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatches_[0] = tmp;
    return;
  }
  if (output.find("matchout1") != std::string::npos) {
    auto* tmp = dynamic_cast<FullMatchMemory*>(memory);
    assert(tmp != nullptr);
    fullmatches_[1] = tmp;
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

void MatchProcessor::read_input_mems(bool& read_is_valid,
                                     std::vector<bool>& mem_hasdata,
                                     std::vector<int>& nentries,
                                     int& read_addr,
                                     const std::vector<int>& iMem,
                                     const std::vector<int>& iPage,
                                     unsigned int& imem,
                                     unsigned int& ipage) {
  bool any_mem_hasdata = false;

  for (unsigned int i = 0; i < mem_hasdata.size(); i++) {
    if (mem_hasdata[i]) {
      any_mem_hasdata = true;
    }
  };

  int read_addr_next = read_addr + 1;

  // priority encoder
  int read_imem = 0;

  for (unsigned int i = 0; i < mem_hasdata.size(); i++) {
    if (!mem_hasdata[i]) {
      read_imem++;
    } else {
      break;
    }
  }

  imem = iMem[read_imem];
  ipage = iPage[read_imem];

  if (read_is_valid) {
    if (read_addr_next >= nentries[read_imem]) {
      // All entries in the memory[read_imem] have been read out
      // Prepare to move to the next non-empty memory
      read_addr = 0;
      mem_hasdata[any_mem_hasdata ? read_imem : 0] = false;  // set the current lowest 1 bit to 0
    } else {
      read_addr = read_addr_next;
    }
  }

  read_is_valid = read_is_valid && any_mem_hasdata;
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

  bool print = getName() == "MP_L3PHIB_E" && iSector == 3;
  print = false;

  phimin_ = phimin;

  Tracklet* oldTracklet = nullptr;
  candidatematch_ = false;

  unsigned int countme = 0;
  unsigned int countall = 0;
  unsigned int countsel = 0;
  unsigned int countinputproj = 0;

  if (print) {
    for (unsigned int i = 0; i < inputprojs_.size(); i++) {
      for (unsigned int p = 0; p < inputprojs_[i]->nPage(); p++) {
        std::cout << "ProjOcc: " << inputprojs_[i]->getName() << " " << p << " " << inputprojs_[i]->nTracklets(p)
                  << std::endl;
      }
    }
  }

  unsigned int imem = 0;
  unsigned int ipage = 0;

  std::vector<int> iMem, iPage, nentries;
  std::vector<bool> mem_hasdata;

  while (imem < inputprojs_.size()) {
    iMem.push_back(imem);
    iPage.push_back(ipage);
    nentries.push_back(inputprojs_[imem]->nTracklets(ipage));
    mem_hasdata.push_back(inputprojs_[imem]->nTracklets(ipage) > 0);
    ipage++;
    if (ipage >= inputprojs_[imem]->nPage()) {
      ipage = 0;
      imem++;
    }
  }

  imem = 0;
  ipage = 0;

  int read_address = 0;
  int mem_read_addr = 0;

  inputProjBuffer_.reset();

  for (const auto& inputproj : inputprojs_) {
    countinputproj += inputproj->nTracklets();
  }

  for (auto& matchengine : matchengines_) {
    matchengine.reset();
  }

  Tracklet* projdata = 0;
  Tracklet* projdata_ = 0;
  bool validin = false;
  bool validin_ = false;
  bool validmem = false;

  // Reset so events avoids stale pointers
  curr_tracklet = nullptr;
  next_tracklet = nullptr;

  for (unsigned int istep = 0; istep < settings_.maxStep("MP"); istep++) {
    //bool projdone = false;

    bool projBufferNearFull = inputProjBuffer_.nearfull4();

    // This print statement is useful for detailed comparison with the HLS code
    // It prints out detailed status information for each clock step

    if (print) {
      std::cout << "istep = " << istep << " projBuff: " << inputProjBuffer_.rptr() << " " << inputProjBuffer_.wptr()
                << " " << projBufferNearFull;
      std::cout << " " << validin << " " << validin_ << " " << validmem;
      unsigned int iMEU = 0;
      for (auto& matchengine : matchengines_) {
        std::cout << " MEU" << iMEU << ": " << matchengine.rptr() << " " << matchengine.wptr() << " "
                  << matchengine.idle() << " " << matchengine.empty() << " " << matchengine.TCID();
        iMEU++;
      }
      std::cout << std::endl;
    }

    //First do some caching of state at the start of the clock

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

    if (candidatematch_) {
      bool match = matchCalculator(tracklet_, fpgastub_, print, istep);

      if (settings_.debugTracklet() && match) {
        edm::LogVerbatim("Tracklet") << getName() << " have match";
      }

      countall++;
      if (match)
        countsel++;
    }

    if (print) {
      std::cout << "hasMatch: " << (!matchengines_[iMEbest].empty()) << std::endl;
    }

    candidatematch_ = false;
    if (!matchengines_[iMEbest].empty()) {
      std::pair<Tracklet*, const Stub*> candmatch = matchengines_[iMEbest].read();

      candidatematch_ = true;

      fpgastub_ = candmatch.second;
      tracklet_ = candmatch.first;

      //Consistency check
      if (oldTracklet != nullptr) {
        //allow equal here since we can have more than one cadidate match per tracklet projection
        //cout << "old new : "<<oldTracklet->TCID()<<" "<<tracklet->TCID()<<" "<<iMEbest<<endl;
        assert(oldTracklet->TCID() <= tracklet_->TCID());
      }
      oldTracklet = tracklet_;

      /*
      bool match = matchCalculator(tracklet, fpgastub, print, istep);
      
      if (settings_.debugTracklet() && match) {
        edm::LogVerbatim("Tracklet") << getName() << " have match";
      }

      countall++;
      if (match)
        countsel++;
      */
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
                                tmpProj.proj(),
                                print);
        meactive = true;
        addedProjection = true;
      } else {
        matchengines_[iME].step(print);
      }
      matchengines_[iME].processPipeline(print);
    }

    //Step 1
    //First step here checks if we have more input projections to put into
    //the input puffer for projections

    if (validin_) {
      //if (settings_.debugTracklet()) {
      //	edm::LogVerbatim("Tracklet") << getName() << " have projection in memory : " << projMem->getName();
      //}

      FPGAWord fpgaphi = projdata_->proj(layerdisk_).fpgaphiproj();

      unsigned int iphi = (fpgaphi.value() >> (fpgaphi.nbits() - nvmbits_)) & (nvmbins_ - 1);

      constexpr int nextrabits = 2;
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
        FPGAWord phider = projdata_->proj(layerdisk_).fpgaphiprojder();
        projrinv = (1 << (nrinv_ - 1)) - 1 - (phider.value() >> (phider.nbits() - nrinv_));
      } else {
        //The next lines looks up the predicted bend based on:
        // 1 - r projections
        // 2 - phi derivative
        // 3 - the sign - i.e. if track is forward or backward

        int rindex = (projdata_->proj(layerdisk_).fpgarzproj().value() >>
                      (projdata_->proj(layerdisk_).fpgarzproj().nbits() - nrbits_)) &
                     ((1 << nrbits_) - 1);

        int phiprojder = projdata_->proj(layerdisk_).fpgaphiprojder().value();

        int phiderindex = (phiprojder >> (projdata_->proj(layerdisk_).fpgaphiprojder().nbits() - nphiderbits_)) &
                          ((1 << nphiderbits_) - 1);

        int signindex = projdata_->proj(layerdisk_).fpgarzprojder().value() < 0;

        int bendindex = (signindex << (nphiderbits_ + nrbits_)) + (rindex << (nphiderbits_)) + phiderindex;

        projrinv = rinvbendlut_.lookup(bendindex);

        projdata_->proj(layerdisk_).setBendIndex(projrinv);
      }
      assert(projrinv >= 0);

      unsigned int projfinephi =
          (fpgaphi.value() >> (fpgaphi.nbits() - (nvmbits_ + NFINEPHIBITS))) & ((1 << NFINEPHIBITS) - 1);

      unsigned int slot;
      bool second;
      int projfinerz;

      if (barrel_) {
        slot = projdata_->proj(layerdisk_).fpgarzbin1projvm().value();
        second = projdata_->proj(layerdisk_).fpgarzbin2projvm().value();
        projfinerz = projdata_->proj(layerdisk_).fpgafinerzvm().value();
      } else {
        //The -1 here is due to not using the full range of bits. Should be fixed.
        unsigned int ir = projdata_->proj(layerdisk_).fpgarzproj().value() >>
                          (projdata_->proj(layerdisk_).fpgarzproj().nbits() - nrprojbits_ - 1);
        unsigned int word = diskRadius_.lookup(ir);

        slot = (word >> 1) & ((1 << N_RZBITS) - 1);
        if (projdata_->proj(layerdisk_).fpgarzprojder().value() < 0) {
          slot += (1 << N_RZBITS);
        }
        second = word & 1;
        projfinerz = word >> 4;
      }

      bool isPSseed = projdata_->PSseed();

      int nbins = (1 << N_RZBITS);
      if (layerdisk_ >= N_LAYER) {
        nbins *= 2;  //twice as many bins in disks (since there are two disks)
      }

      VMStubsMEMemory* stubmem = vmstubs_[0];
      bool usefirstMinus = stubmem->nStubsBin(ivmMinus * nbins + slot) != 0;
      bool usesecondMinus = (second && (stubmem->nStubsBin(ivmMinus * nbins + slot + 1) != 0));
      bool usefirstPlus = ivmPlus != ivmMinus && (stubmem->nStubsBin(ivmPlus * nbins + slot) != 0);
      bool usesecondPlus = ivmPlus != ivmMinus && (second && (stubmem->nStubsBin(ivmPlus * nbins + slot + 1) != 0));

      bool good = usefirstPlus || usesecondPlus || usefirstMinus || usesecondMinus;

      /*
      int offset = 4;

      int ztemp = proj->proj(layerdisk_).fpgarzproj().value()  >> (proj->proj(layerdisk_).fpgarzproj().nbits() - settings_.MEBinsBits() - NFINERZBITS);
      unsigned int zbin1 = (1 << (settings_.MEBinsBits() - 1)) + ((ztemp - offset) >> NFINERZBITS);
      unsigned int zbin2 = (1 << (settings_.MEBinsBits() - 1)) + ((ztemp + offset) >> NFINERZBITS);
      
      if (zbin1 >= settings_.MEBins()) {
	zbin1 = 0;  //note that zbin1 is unsigned
      }
      if (zbin2 >= settings_.MEBins()) {
	zbin2 = settings_.MEBins() - 1;
      }
      */

      if (good) {
        ProjectionTemp tmpProj(projdata_,
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
        if (print) {
          std::cout << "Add projection to inputProjBuffer istep = " << istep << std::endl;
        }
        inputProjBuffer_.store(tmpProj);
      }
    }

    projdata_ = projdata;
    validin_ = validin;

    validin = validmem;

    if (validin) {
      TrackletProjectionsMemory* projMem = inputprojs_[imem];
      projdata = projMem->getTracklet(read_address, ipage);
      if (print & validin) {
        std::cout << "Reading iprojmem page, readaddress : " << istep << " " << imem << " " << ipage << " "
                  << read_address << std::endl;
      }
    }

    validmem = !projBufferNearFull;

    read_address = mem_read_addr;

    if (validmem) {
      read_input_mems(validmem, mem_hasdata, nentries, mem_read_addr, iMem, iPage, imem, ipage);
    }

    //
    //  Check if done
    //
    //
    //

    /*
    if ((iprojmem!=0 && projdone && !meactive && inputProjBuffer_.rptr() == inputProjBuffer_.wptr()) ||
        (istep == settings_.maxStep("MP") - 1)) {
      if (settings_.writeMonitorData("MP")) {
        globals_->ofstream("matchprocessor.txt") << getName() << " " << istep << " " << countall << " " << countsel
                                                 << " " << countme << " " << countinputproj << endl;
      }
      break;
    }
    */

  }  // end of istep

  if (settings_.writeMonitorData("MC")) {
    globals_->ofstream("matchcalculator.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}

bool MatchProcessor::matchCalculator(Tracklet* tracklet, const Stub* fpgastub, bool print, unsigned int istep) {
  const L1TStub* stub = fpgastub->l1tstub();

  if (layerdisk_ < N_LAYER) {
    const Projection& proj = tracklet->proj(layerdisk_);
    int ir = fpgastub->rvalue();
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
    curr_tracklet = next_tracklet;
    next_tracklet = tracklet;

    // Do we have a new tracklet?
    bool newtracklet = next_tracklet != curr_tracklet;
    if (istep == 0)
      best_ideltar_disk = (1 << (fpgastub->r().nbits() - 1));  // Set to the maximum possible
    // If so, replace the "best" values with the cut tables
    if (newtracklet) {
      best_ideltaphi_barrel = (int)phimatchcuttable_.lookup(seedindex);
      best_ideltaz_barrel = (int)zmatchcuttable_.lookup(seedindex);
    }

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

    bool imatch = (std::abs(ideltaphi) <= best_ideltaphi_barrel && (ideltaz << dzshift_ < best_ideltaz_barrel) &&
                   (ideltaz << dzshift_ >= -best_ideltaz_barrel));

    if (settings_.debugTracklet()) {
      edm::LogVerbatim("Tracklet") << getName() << " imatch = " << imatch << " ideltaphi cut " << ideltaphi << " "
                                   << phimatchcuttable_.lookup(seedindex) << " ideltaz<<dzshift cut "
                                   << (ideltaz << dzshift_) << " " << zmatchcuttable_.lookup(seedindex);
    }

    //This would catch significant consistency problems in the configuration - helps to debug if there are problems.
    if (std::abs(dphi) > 0.5 * settings_.dphisectorHG() || std::abs(dphiapprox) > 0.5 * settings_.dphisectorHG()) {
      //throw cms::Exception("LogicError") << "WARNING dphi and/or dphiapprox too large : " << dphi << " " << dphiapprox
      //                                   << endl;
      std::cout << "WARNING dphi and/or dphiapprox too large : " << dphi << " " << dphiapprox << std::endl;
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
    // Update the "best" values
    if (imatch) {
      best_ideltaphi_barrel = std::abs(ideltaphi);
      best_ideltaz_barrel = std::abs(ideltaz << dzshift_);
    }

    if (imatch) {
      if (print) {
        std::cout << "Adding match on istep = " << istep << std::endl;
      }

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
      int iTB = 0;
      if (iSeed == 2 || iSeed == 4 || iSeed == 5 || iSeed == 6) {
        iTB = 1;
      }
      assert(fullmatches_[iTB] != nullptr);
      fullmatches_[iTB]->addMatch(tracklet, fpgastub);

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

    int irstub = fpgastub->rvalue();
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

    constexpr int diff_bits = 1;
    int ideltar = (irstub >> diff_bits) - ir;

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

    curr_tracklet = next_tracklet;
    next_tracklet = tracklet;
    // Do we have a new tracklet?
    bool newtracklet = next_tracklet != curr_tracklet;
    // If so, replace the "best" values with the cut tables
    if (newtracklet) {
      best_ideltaphi_disk = idrphicut;
      best_ideltar_disk = idrcut;
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
    // Update the "best" values
    if (imatch) {
      best_ideltaphi_disk = std::abs(ideltaphi) * irstub;
      best_ideltar_disk = std::abs(ideltar);
    }

    if (imatch) {
      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "MatchCalculator found match in disk " << getName();
      }

      if (std::abs(dphi) >= third * settings_.dphisectorHG()) {
        edm::LogPrint("Tracklet") << "dphi " << dphi << " ISeed " << tracklet->getISeed();
      }
      //assert(std::abs(dphi) < third * settings_.dphisectorHG());
      //assert(std::abs(dphiapprox) < third * settings_.dphisectorHG());

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
      int iTB = 0;
      if (iSeed == 2 || iSeed == 4 || iSeed == 5 || iSeed == 6) {
        iTB = 1;
      }
      assert(fullmatches_[iTB] != nullptr);
      fullmatches_[iTB]->addMatch(tracklet, fpgastub);

      return true;
    } else {
      return false;
    }
  }
}
