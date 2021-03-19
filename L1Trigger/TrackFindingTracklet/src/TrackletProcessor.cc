
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <utility>
#include <tuple>

using namespace std;
using namespace trklet;

TrackletProcessor::TrackletProcessor(string name, Settings const& settings, Globals* globals, unsigned int iSector)
    : TrackletCalculatorBase(name, settings, globals, iSector), vmrtable_(settings) {
  iAllStub_ = -1;

  for (unsigned int ilayer = 0; ilayer < N_LAYER; ilayer++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(ilayer), nullptr);
    trackletprojlayers_.push_back(tmp);
  }

  for (unsigned int idisk = 0; idisk < N_DISK; idisk++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(idisk + N_LAYER), nullptr);
    trackletprojdisks_.push_back(tmp);
  }

  outervmstubs_ = nullptr;

  initLayerDisksandISeed(layerdisk1_, layerdisk2_, iSeed_);

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk1_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk1_);

  vmrtable_.init(layerdisk1_, getName());

  nbitsrzbin_ = NFINERZBITS;
  if (iSeed_ == 4 || iSeed_ == 5)
    nbitsrzbin_ = 2;

  innerphibits_ = settings_.nfinephi(0, iSeed_);
  outerphibits_ = settings_.nfinephi(1, iSeed_);

  // set TC index
  iTC_ = name_[7] - 'A';
  assert(iTC_ >= 0 && iTC_ < 14);

  TCIndex_ = (iSeed_ << 4) + iTC_;
  assert(TCIndex_ >= 0 && TCIndex_ <= (int)settings_.ntrackletmax());
}

void TrackletProcessor::addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory) {
  outputProj = dynamic_cast<TrackletProjectionsMemory*>(memory);
  assert(outputProj != nullptr);
}

void TrackletProcessor::addOutput(MemoryBase* memory, string output) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding output to " << memory->getName() << " to output "
                                 << output;
  }
  if (output == "trackpar") {
    auto* tmp = dynamic_cast<TrackletParametersMemory*>(memory);
    assert(tmp != nullptr);
    trackletpars_ = tmp;
    return;
  }

  if (output.substr(0, 7) == "projout") {
    //output is on the form 'projoutL2PHIC' or 'projoutD3PHIB'
    auto* tmp = dynamic_cast<TrackletProjectionsMemory*>(memory);
    assert(tmp != nullptr);

    unsigned int layerdisk = output[8] - '1';   //layer or disk counting from 0
    unsigned int phiregion = output[12] - 'A';  //phiregion counting from 0

    if (output[7] == 'L') {
      assert(layerdisk < N_LAYER);
      assert(phiregion < trackletprojlayers_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojlayers_[layerdisk][phiregion] == nullptr);
      trackletprojlayers_[layerdisk][phiregion] = tmp;
      return;
    }

    if (output[7] == 'D') {
      assert(layerdisk < N_DISK);
      assert(phiregion < trackletprojdisks_[layerdisk].size());
      //check that phiregion not already initialized
      assert(trackletprojdisks_[layerdisk][phiregion] == nullptr);
      trackletprojdisks_[layerdisk][phiregion] = tmp;
      return;
    }
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find output : " << output;
}

void TrackletProcessor::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }

  if (input == "outervmstubin") {
    auto* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    outervmstubs_ = tmp;
    iAllStub_ = tmp->getName()[11] - 'A';
    if (iSeed_ == 1)
      iAllStub_ = tmp->getName()[11] - 'I';
    if (iSeed_ == 6 || iSeed_ == 7) {
      if (tmp->getName()[11] == 'X')
        iAllStub_ = 0;
      if (tmp->getName()[11] == 'Y')
        iAllStub_ = 1;
      if (tmp->getName()[11] == 'Z')
        iAllStub_ = 2;
      if (tmp->getName()[11] == 'W')
        iAllStub_ = 3;
    }
    buildLUT();  //need iAllStub_ set before building the table //FIXME should be in initiall

    TrackletEngineUnit teunit(&settings_,
                              nbitsfinephi_,
                              layerdisk2_,
                              iSeed_,
                              nbitsfinephidiff_,
                              iAllStub_,
                              pttableinnernew_,
                              pttableouternew_,
                              outervmstubs_);
    teunits_.resize(6, teunit);

    return;
  }

  if (input == "innerallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    if (innerallstubs_.size() == 2) {  //FIXME this should be done with better logic with reading the input stubs
      innerallstubs_.insert(innerallstubs_.begin(), tmp);
    } else {
      innerallstubs_.push_back(tmp);
    }

    //FIXME should be done once after all inputs are added
    tedatabuffers_.clear();
    CircularBuffer<TEData> tedatabuffertmp(5);
    tedatabuffertmp.reset();

    tedatabuffers_.emplace_back(tedatabuffertmp, 0, 0, 0, (innerallstubs_.size() + 1) / 2);
    tedatabuffers_.emplace_back(
        tedatabuffertmp, 0, (innerallstubs_.size() + 1) / 2, (innerallstubs_.size() + 1) / 2, innerallstubs_.size());

    return;
  }
  if (input == "outerallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    outerallstubs_.push_back(tmp);
    return;
  }
  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletProcessor::execute() {
  if (!settings_.useSeed(iSeed_))
    return;

  //Consistency checks
  assert(iAllStub_ >= 0);
  assert(iAllStub_ < (int)settings_.nallstubs(layerdisk2_));
  assert(outervmstubs_ != nullptr);

  //used to collect performance data
  unsigned int countall = 0;
  unsigned int countsel = 0;

  unsigned int countteall = 0;
  unsigned int stubpairs = 0;

  unsigned int ntedata = 0;

  unsigned int ninnerstubs = 0;

  //Actual implemenation starts here

  //Reset the tebuffers
  for (auto& tebuffer : tedatabuffers_) {
    std::get<0>(tebuffer).reset();
    std::get<1>(tebuffer) = 0;
    std::get<2>(tebuffer) = std::get<3>(tebuffer);
  }

  //Reset the teunits
  for (auto& teunit : teunits_) {
    teunit.reset();
  }

  for (unsigned int istep = 0; istep < settings_.maxStep("TP"); istep++) {
    for (auto& tebuffer : tedatabuffers_) {
      CircularBuffer<TEData>& tedatabuffer = std::get<0>(tebuffer);
      unsigned int& istub = std::get<1>(tebuffer);
      unsigned int& imem = std::get<2>(tebuffer);
      unsigned int imemend = std::get<4>(tebuffer);

      if ((!tedatabuffer.full()) && imem < imemend && istub < innerallstubs_[imem]->nStubs()) {
        ninnerstubs++;

        const Stub* stub = innerallstubs_[imem]->getStub(istub);

        bool negdisk = (stub->disk().value() < 0);  //FIXME stub needs to contain bit for +/- z disk

        FPGAWord phicorr = stub->phicorr();
        int innerfinephi = phicorr.bits(phicorr.nbits() - nbitsfinephi_, nbitsfinephi_);
        FPGAWord innerbend = stub->bend();

        //Take the top nbitszfinebintable_ bits of the z coordinate
        int indexz =
            (((1 << (stub->z().nbits() - 1)) + stub->z().value()) >> (stub->z().nbits() - nbitszfinebintable_));
        int indexr = -1;
        if (layerdisk1_ > (N_LAYER - 1)) {
          if (negdisk) {
            indexz = (1 << nbitszfinebintable_) - indexz;
          }
          indexr = stub->r().value();
          if (stub->isPSmodule()) {
            indexr = stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_);
          }
        } else {
          //Take the top nbitsfinebintable_ bits of the z coordinate
          indexr = (((1 << (stub->r().nbits() - 1)) + stub->r().value()) >> (stub->r().nbits() - nbitsrfinebintable_));
        }

        int lutval = -1;
        if (iSeed_ < 6) {  //FIXME should only be one table...
          lutval = vmrtable_.lookupinner(indexz, indexr);
        } else {
          lutval = vmrtable_.lookupinneroverlap(indexz, indexr);
        }

        if (lutval != -1) {
          unsigned int lutwidth = settings_.lutwidthtab(0, iSeed_);
          FPGAWord lookupbits(lutval, lutwidth, true, __LINE__, __FILE__);

          int rzfinebinfirst = lookupbits.bits(0, 3);  //finephi
          int next = lookupbits.bits(3, 1);            //next r/z bin
          int start = lookupbits.bits(4, nbitsrzbin_);
          int rzdiffmax = lookupbits.bits(lookupbits.nbits() - 3, 3);

          if ((iSeed_ == 4 || iSeed_ == 5) && negdisk) {  //TODO - need to store negative disk
            start += 4;
          }
          int last = start + next;

          int nbins = 8;

          unsigned int useregindex = (innerfinephi << innerbend.nbits()) + innerbend.value();
          if (iSeed_ >= 4) {
            //FIXME If the lookupbits were rationally organized this would be much simpler
            int ir = ((start & 3) << 1) + (rzfinebinfirst >> 2);
            useregindex = (useregindex << 3) + ir;
          }

          assert(useregindex < useregion_.size());
          vector<bool> usereg = useregion_[useregindex];

          TEData tedata;
          tedata.stub_ = stub;
          tedata.rzbinfirst_ = rzfinebinfirst;
          tedata.start_ = start;
          tedata.innerfinephi_ = innerfinephi;
          tedata.rzdiffmax_ = rzdiffmax;
          tedata.innerbend_ = innerbend;

          for (unsigned int ireg = 0; ireg < settings_.nvmte(1, iSeed_); ireg++) {
            assert(ireg < usereg.size());
            if (!usereg[ireg])
              continue;

            for (int ibin = start; ibin <= last; ibin++) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << getName() << " looking for matching stub in bin " << ibin << " with "
                                             << outervmstubs_->nVMStubsBinned(ireg * nbins + ibin) << " stubs";
              }
              assert(ireg * nbins + ibin < outervmstubs_->nBin());
              int nstubs = outervmstubs_->nVMStubsBinned(ireg * nbins + ibin);
              if (nstubs > 0) {
                tedata.regions_.emplace_back(tuple<int, int, int>(ibin - start, ireg, nstubs));
                countteall += nstubs;
              }
            }
          }

          if (!tedata.regions_.empty()) {
            ntedata++;
            //cout << getName() << " adding TE data" << endl;
            tedatabuffer.store(tedata);
          }
        }
        istub++;
        if (istub >= innerallstubs_[imem]->nStubs()) {
          istub = 0;
          imem++;
        }
      }
    }

    bool tedatabuffersdone = true;
    for (auto& tebuffer : tedatabuffers_) {
      CircularBuffer<TEData>& tedatabuffer = std::get<0>(tebuffer);
      tedatabuffersdone &= tedatabuffer.empty();                              //buffer empty
      tedatabuffersdone &= (std::get<2>(tebuffer) == std::get<4>(tebuffer));  //buffer processed all stubs
      if (!tedatabuffer.empty()) {
        for (auto& teunit : teunits_) {
          if (teunit.empty() && teunit.idle()) {
            teunit.init(tedatabuffer.read());
            break;
          }
        }
      }
    }

    bool teunitsdone = true;
    TrackletEngineUnit* teunitptr = nullptr;
    for (auto& teunit : teunits_) {
      if (!teunit.idle()) {
        teunit.step();
        teunitsdone = false;
      }
      if (!teunit.empty()) {
        teunitptr = &teunit;
        teunitsdone = false;
      }
    }

    if (teunitptr != nullptr) {
      auto stubpair = teunitptr->read();
      stubpairs++;

      if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
        edm::LogVerbatim("Tracklet") << "Will break on too many tracklets in " << getName();
        break;
      }
      countall++;
      const Stub* innerFPGAStub = stubpair.first;
      const L1TStub* innerStub = innerFPGAStub->l1tstub();

      const Stub* outerFPGAStub = stubpair.second;
      const L1TStub* outerStub = outerFPGAStub->l1tstub();

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletProcessor execute " << getName() << "[" << iSector_ << "]";
      }

      bool accept = false;

      if (iSeed_ < 4) {
        accept = barrelSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
      } else if (iSeed_ < 6) {
        accept = diskSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
      } else {
        accept = overlapSeeding(outerFPGAStub, outerStub, innerFPGAStub, innerStub);
      }

      if (accept)
        countsel++;

      if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
        edm::LogVerbatim("Tracklet") << "Will break on number of tracklets in " << getName();
        break;
      }

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletProcessor execute done";
      }
    }

    if (settings_.writeMonitorData("TP")) {
      globals_->ofstream("trackletprocessor.txt") << getName() << " " << ninnerstubs   //# inner stubs
                                                  << " " << outervmstubs_->nVMStubs()  //# outer stubs
                                                  << " " << countteall                 //# pairs tried in TE
                                                  << " " << stubpairs                  //# stubs pairs
                                                  << " " << countsel                   //# tracklets found
                                                  << endl;
    }

    if (teunitsdone && tedatabuffersdone) {
      break;
    }
  }
}

void TrackletProcessor::writeTETable() {
  ofstream outstubptinnercut;
  outstubptinnercut.open(getName() + "_stubptinnercut.tab");
  outstubptinnercut << "{" << endl;
  // the below is outcommented as the writing of these tables are currently not implemented, to be adressed
  //for(unsigned int i=0;i<pttableinner_.size();i++){
  //  if (i!=0) outstubptinnercut<<","<<endl;
  //  outstubptinnercut << pttableinner_[i];
  //}
  outstubptinnercut << endl << "};" << endl;
  outstubptinnercut.close();

  ofstream outstubptoutercut;
  outstubptoutercut.open(getName() + "_stubptoutercut.tab");
  outstubptoutercut << "{" << endl;
  // the below is outcommented as the writing of these tables are currently not implemented, to be adressed
  //for(unsigned int i=0;i<pttableouter_.size();i++){
  //  if (i!=0) outstubptoutercut<<","<<endl;
  //  outstubptoutercut << pttableouter_[i];
  //}
  outstubptoutercut << endl << "};" << endl;
  outstubptoutercut.close();
}

void TrackletProcessor::buildLUT() {
  //number of fine phi bins in sector
  int nfinephibins =
      settings_.nallstubs(layerdisk2_) * settings_.nvmte(1, iSeed_) * (1 << settings_.nfinephi(1, iSeed_));
  double dfinephi = settings_.dphisectorHG() / nfinephibins;

  double rmin = -1.0;
  double rmax = -1.0;

  if (iSeed_ < 4) {
    rmin = settings_.rmean(layerdisk1_);
    rmax = settings_.rmean(layerdisk2_);
  } else {
    if (iSeed_ > 5) {
      if (iSeed_ == 6) {
        rmax = settings_.rmaxdiskl1overlapvm();
      } else if (iSeed_ == 7) {
        rmax = settings_.rmaxdiskvm();
      }
      rmin = settings_.rmean(layerdisk1_);
    } else {
      rmax = settings_.rmaxdiskvm();
      rmin = rmax * settings_.zmean(layerdisk2_ - 6 - 1) / settings_.zmean(layerdisk2_ - 6);
    }
  }

  double dphimax = asin(0.5 * settings_.maxrinv() * rmax) - asin(0.5 * settings_.maxrinv() * rmin);

  int nbins = 2.0 * (dphimax / dfinephi + 1.0);

  nbitsfinephidiff_ = log(nbins) / log(2.0) + 1;

  nbitsfinephi_ = settings_.nbitsallstubs(layerdisk2_) + settings_.nbitsvmte(1, iSeed_) + settings_.nfinephi(1, iSeed_);

  int outerrbits = 3;
  if (iSeed_ < 4) {
    outerrbits = 0;
  }

  int outerrbins = (1 << outerrbits);

  double dphi[2];
  double router[2];

  unsigned int nbendbitsinner = 3;
  unsigned int nbendbitsouter = 3;
  if (iSeed_ == 2) {
    nbendbitsouter = 4;
  } else if (iSeed_ == 3) {
    nbendbitsinner = 4;
    nbendbitsouter = 4;
  }

  int nbinsfinephidiff = (1 << nbitsfinephidiff_);

  for (int iphibin = 0; iphibin < nbinsfinephidiff; iphibin++) {
    int iphidiff = iphibin;
    if (iphibin >= nbinsfinephidiff / 2) {
      iphidiff = iphibin - nbinsfinephidiff;
    }
    //min and max dphi
    dphi[0] = (iphidiff - 1.5) * dfinephi;
    dphi[1] = (iphidiff + 1.5) * dfinephi;
    for (int irouterbin = 0; irouterbin < outerrbins; irouterbin++) {
      if (iSeed_ >= 4) {
        router[0] =
            settings_.rmindiskvm() + irouterbin * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
        router[1] =
            settings_.rmindiskvm() + (irouterbin + 1) * (settings_.rmaxdiskvm() - settings_.rmindiskvm()) / outerrbins;
      } else {
        router[0] = settings_.rmean(layerdisk2_);
        router[1] = settings_.rmean(layerdisk2_);
      }

      double bendinnermin = 20.0;
      double bendinnermax = -20.0;
      double bendoutermin = 20.0;
      double bendoutermax = -20.0;
      double rinvmin = 1.0;
      for (int i2 = 0; i2 < 2; i2++) {
        for (int i3 = 0; i3 < 2; i3++) {
          double rinner = 0.0;
          if (iSeed_ == 4 || iSeed_ == 5) {
            rinner = router[i3] * settings_.zmean(layerdisk1_ - N_LAYER) / settings_.zmean(layerdisk2_ - N_LAYER);
          } else {
            rinner = settings_.rmean(layerdisk1_);
          }
          double rinv1 = rinv(0.0, dphi[i2], rinner, router[i3]);
          double pitchinner = (rinner < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double pitchouter =
              (router[i3] < settings_.rcrit()) ? settings_.stripPitch(true) : settings_.stripPitch(false);
          double abendinner = -bend(rinner, rinv1, pitchinner);
          double abendouter = -bend(router[i3], rinv1, pitchouter);
          if (abendinner < bendinnermin)
            bendinnermin = abendinner;
          if (abendinner > bendinnermax)
            bendinnermax = abendinner;
          if (abendouter < bendoutermin)
            bendoutermin = abendouter;
          if (abendouter > bendoutermax)
            bendoutermax = abendouter;
          if (std::abs(rinv1) < rinvmin) {
            rinvmin = std::abs(rinv1);
          }
        }
      }

      bool passptcut = rinvmin < settings_.rinvcutte();

      for (int ibend = 0; ibend < (1 << nbendbitsinner); ibend++) {
        double bend = benddecode(ibend, nbendbitsinner == 3);

        bool passinner = bend - bendinnermin > -settings_.bendcutte(0, iSeed_) &&
                         bend - bendinnermax < settings_.bendcutte(0, iSeed_);
        pttableinnernew_.push_back(passinner && passptcut);
      }

      for (int ibend = 0; ibend < (1 << nbendbitsouter); ibend++) {
        double bend = benddecode(ibend, nbendbitsouter == 3);

        bool passouter = bend - bendoutermin > -settings_.bendcutte(1, iSeed_) &&
                         bend - bendoutermax < settings_.bendcutte(1, iSeed_);
        pttableouternew_.push_back(passouter && passptcut);
      }
    }
  }

  int nirbits = 0;
  if (iSeed_ >= 4)
    nirbits = 3;

  int nregmax = 0;

  for (int innerfinephi = 0; innerfinephi < (1 << nbitsfinephi_); innerfinephi++) {
    for (int innerbend = 0; innerbend < (1 << nbendbitsinner); innerbend++) {
      for (int ir = 0; ir < (1 << nirbits); ir++) {
        int nreg = 0;
        vector<bool> usereg;
        for (unsigned int ireg = 0; ireg < settings_.nvmte(1, iSeed_); ireg++) {
          bool match = false;
          for (int ifinephiouter = 0; ifinephiouter < (1 << settings_.nfinephi(1, iSeed_)); ifinephiouter++) {
            int outerfinephi = iAllStub_ * (1 << (nbitsfinephi_ - settings_.nbitsallstubs(layerdisk2_))) +
                               ireg * (1 << settings_.nfinephi(1, iSeed_)) + ifinephiouter;
            int idphi = outerfinephi - innerfinephi;
            bool inrange = (idphi < (1 << (nbitsfinephidiff_ - 1))) && (idphi >= -(1 << (nbitsfinephidiff_ - 1)));
            if (idphi < 0)
              idphi = idphi + (1 << nbitsfinephidiff_);
            int idphi1 = idphi;
            if (iSeed_ >= 4)
              idphi1 = (idphi << 3) + ir;
            int ptinnerindexnew = (idphi1 << nbendbitsinner) + innerbend;
            match = match || (inrange && pttableinnernew_[ptinnerindexnew]);
          }
          if (match && ir < 3)
            nreg++;
          usereg.push_back(match);
        }

        useregion_.push_back(usereg);
        if (nreg > nregmax)
          nregmax = nreg;
      }
    }
  }
}
