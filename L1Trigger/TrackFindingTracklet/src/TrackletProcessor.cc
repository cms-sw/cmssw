#include "L1Trigger/TrackFindingTracklet/interface/TrackletProcessor.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllInnerStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <utility>
#include <tuple>

using namespace std;
using namespace trklet;

TrackletProcessor::TrackletProcessor(string name, Settings const& settings, Globals* globals)
    : TrackletCalculatorBase(name, settings, globals),
      tebuffer_(CircularBuffer<TEData>(3), 0, 0, 0, 0),
      pttableinner_(settings),
      pttableouter_(settings),
      useregiontable_(settings),
      innerTable_(settings),
      innerOverlapTable_(settings) {
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

  double rmin = -1.0;
  double rmax = -1.0;

  if (iSeed_ == Seed::L1L2 || iSeed_ == Seed::L2L3 || iSeed_ == Seed::L3L4 || iSeed_ == Seed::L5L6) {
    rmin = settings_.rmean(layerdisk1_);
    rmax = settings_.rmean(layerdisk2_);
  } else {
    if (iSeed_ == Seed::L1D1) {
      rmax = settings_.rmaxdiskl1overlapvm();
      rmin = settings_.rmean(layerdisk1_);
    } else if (iSeed_ == Seed::L2D1) {
      rmax = settings_.rmaxdiskvm();
      rmin = settings_.rmean(layerdisk1_);
    } else {
      rmax = settings_.rmaxdiskvm();
      rmin = rmax * settings_.zmean(layerdisk2_ - N_LAYER - 1) / settings_.zmean(layerdisk2_ - N_LAYER);
    }
  }

  double dphimax = asin(0.5 * settings_.maxrinv() * rmax) - asin(0.5 * settings_.maxrinv() * rmin);

  //number of fine phi bins in sector
  int nfinephibins =
      settings_.nallstubs(layerdisk2_) * settings_.nvmte(1, iSeed_) * (1 << settings_.nfinephi(1, iSeed_));
  double dfinephi = settings_.dphisectorHG() / nfinephibins;

  nbitsfinephi_ = settings_.nbitsallstubs(layerdisk2_) + settings_.nbitsvmte(1, iSeed_) + settings_.nfinephi(1, iSeed_);

  int nbins = 2.0 * (dphimax / dfinephi + 1.0);

  nbitsfinephidiff_ = log(nbins) / log(2.0) + 1;

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk1_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk1_);

  nbitsrzbin_ = N_RZBITS;
  if (iSeed_ == Seed::D1D2 || iSeed_ == Seed::D3D4)
    nbitsrzbin_ = 2;

  innerphibits_ = settings_.nfinephi(0, iSeed_);
  outerphibits_ = settings_.nfinephi(1, iSeed_);

  if (layerdisk1_ == LayerDisk::L1 || layerdisk1_ == LayerDisk::L2 || layerdisk1_ == LayerDisk::L3 ||
      layerdisk1_ == LayerDisk::L5 || layerdisk1_ == LayerDisk::D1 || layerdisk1_ == LayerDisk::D3) {
    innerTable_.initVMRTable(layerdisk1_, TrackletLUT::VMRTableType::inner);  //projection to next layer/disk
  }

  if (layerdisk1_ == LayerDisk::L1 || layerdisk1_ == LayerDisk::L2) {
    innerOverlapTable_.initVMRTable(layerdisk1_,
                                    TrackletLUT::VMRTableType::inneroverlap);  //projection to disk from layer
  }

  // set TC index
  iTC_ = name_[7] - 'A';
  assert(iTC_ >= 0 && iTC_ < 14);

  TCIndex_ = (iSeed_ << 4) + iTC_;
  assert(TCIndex_ >= 0 && TCIndex_ <= (int)settings_.ntrackletmax());

  maxStep_ = settings_.maxStep("TP");
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
    if (iSeed_ == Seed::L2L3)
      iAllStub_ = tmp->getName()[11] - 'I';
    if (iSeed_ == Seed::L1D1 || iSeed_ == Seed::L2D1) {
      if (tmp->getName()[11] == 'X')
        iAllStub_ = 0;
      if (tmp->getName()[11] == 'Y')
        iAllStub_ = 1;
      if (tmp->getName()[11] == 'Z')
        iAllStub_ = 2;
      if (tmp->getName()[11] == 'W')
        iAllStub_ = 3;
    }

    unsigned int iTP = getName()[7] - 'A';

    pttableinner_.initTPlut(true, iSeed_, layerdisk1_, layerdisk2_, nbitsfinephidiff_, iTP);
    pttableouter_.initTPlut(false, iSeed_, layerdisk1_, layerdisk2_, nbitsfinephidiff_, iTP);

    //need iAllStub_ set before building the table

    useregiontable_.initTPregionlut(
        iSeed_, layerdisk1_, layerdisk2_, iAllStub_, nbitsfinephidiff_, nbitsfinephi_, pttableinner_, iTP);

    TrackletEngineUnit teunit(&settings_,
                              nbitsfinephi_,
                              layerdisk1_,
                              layerdisk2_,
                              iSeed_,
                              nbitsfinephidiff_,
                              iAllStub_,
                              &pttableinner_,
                              &pttableouter_,
                              outervmstubs_);

    teunits_.resize(settings_.teunits(iSeed_), teunit);

    return;
  }

  if (input == "innerallstubin") {
    auto* tmp = dynamic_cast<AllInnerStubsMemory*>(memory);
    assert(tmp != nullptr);
    if (innerallstubs_.size() == 2) {  //FIXME this should be done with better logic with reading the input stubs
      innerallstubs_.insert(innerallstubs_.begin(), tmp);
    } else {
      innerallstubs_.push_back(tmp);
    }

    //FIXME should be done once after all inputs are added
    tebuffer_ = tuple<CircularBuffer<TEData>, unsigned int, unsigned int, unsigned int, unsigned int>(
        CircularBuffer<TEData>(3), 0, 0, 0, innerallstubs_.size());

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

void TrackletProcessor::execute(unsigned int iSector, double phimin, double phimax) {
  bool print = (iSector == 3) && (getName() == "TP_L1L2D");
  print = false;

  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  if (!settings_.useSeed(iSeed_))
    return;

  //Not most elegant solution; but works
  int donecount = 0;

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

  //Reset the tebuffer
  std::get<0>(tebuffer_).reset();
  std::get<1>(tebuffer_) = 0;
  std::get<2>(tebuffer_) = std::get<3>(tebuffer_);

  //Reset the teunits
  for (auto& teunit : teunits_) {
    teunit.reset();
  }

  TEData tedata;
  TEData tedata__;
  TEData tedata___;
  bool goodtedata = false;
  bool goodtedata__ = false;
  bool goodtedata___ = false;

  bool tebuffernearfull;

  for (unsigned int istep = 0; istep < maxStep_; istep++) {
    // These print statements are not on by defaul but can be enabled for the
    // comparison with HLS code to track differences.
    if (print) {
      CircularBuffer<TEData>& tedatabuffer = std::get<0>(tebuffer_);
      unsigned int& istub = std::get<1>(tebuffer_);
      unsigned int& imem = std::get<2>(tebuffer_);
      cout << "istep=" << istep << " TEBuffer: " << istub << " " << imem << " " << tedatabuffer.rptr() << " "
           << tedatabuffer.wptr();
      int k = -1;
      for (auto& teunit : teunits_) {
        k++;
        cout << " [" << k << " " << teunit.rptr() << " " << teunit.wptr() << " " << teunit.idle() << "]";
      }
      cout << endl;
    }

    CircularBuffer<TEData>& tedatabuffer = std::get<0>(tebuffer_);
    tebuffernearfull = tedatabuffer.nearfull();

    //
    // First block here checks if there is a teunit with data that should should be used
    // to calculate the tracklet parameters
    //

    TrackletEngineUnit* teunitptr = nullptr;

    for (auto& teunit : teunits_) {
      teunit.setNearFull();
      if (!teunit.empty()) {
        teunitptr = &teunit;
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

      if (iSeed_ == Seed::L1L2 || iSeed_ == Seed::L2L3 || iSeed_ == Seed::L3L4 || iSeed_ == Seed::L5L6) {
        accept = barrelSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
      } else if (iSeed_ == Seed::D1D2 || iSeed_ == Seed::D3D4) {
        accept = diskSeeding(innerFPGAStub, innerStub, outerFPGAStub, outerStub);
      } else {
        accept = overlapSeeding(outerFPGAStub, outerStub, innerFPGAStub, innerStub);
      }

      if (accept)
        countsel++;

      if (trackletpars_->nTracklets() >= settings_.ntrackletmax()) {
        edm::LogVerbatim("Tracklet") << "Will break on number of tracklets in " << getName();
        assert(0);
        break;
      }

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletProcessor execute done";
      }
    }

    //
    // The second block fills the teunit if data in buffer and process TEUnit step
    //
    //

    bool notemptytebuffer = !tedatabuffer.empty();

    int ite = -1;
    for (auto& teunit : teunits_) {
      ite++;
      if (teunit.idle()) {
        if (notemptytebuffer) {
          teunit.init(std::get<0>(tebuffer_).read());
          notemptytebuffer = false;  //prevent initialzing another TE unit
        }
      }
      teunit.step(print, istep, ite);
    }

    //
    // The third block here checks if we have input stubs to process
    //
    //

    if (goodtedata___)
      tedatabuffer.store(tedata___);

    goodtedata = false;

    unsigned int& istub = std::get<1>(tebuffer_);
    unsigned int& imem = std::get<2>(tebuffer_);
    unsigned int imemend = std::get<4>(tebuffer_);

    if ((!tebuffernearfull) && imem < imemend && istub < innerallstubs_[imem]->nStubs()) {
      ninnerstubs++;

      const Stub* stub = innerallstubs_[imem]->getStub(istub);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << getName() << " Have stub in " << innerallstubs_[imem]->getName();
      }

      bool negdisk = (stub->disk().value() < 0);  //FIXME stub needs to contain bit for +/- z disk

      FPGAWord phicorr = stub->phicorr();
      int innerfinephi = phicorr.bits(phicorr.nbits() - nbitsfinephi_, nbitsfinephi_);
      FPGAWord innerbend = stub->bend();

      //Take the top nbitszfinebintable_ bits of the z coordinate
      int indexz = (stub->z().value() >> (stub->z().nbits() - nbitszfinebintable_)) & ((1 << nbitszfinebintable_) - 1);
      int indexr = -1;
      if (layerdisk1_ > (N_LAYER - 1)) {
        if (negdisk) {
          indexz = ((1 << nbitszfinebintable_) - 1) - indexz;
        }
        indexr = stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_);
      } else {  //Take the top nbitsfinebintable_ bits of the z coordinate
        indexr = (stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_)) & ((1 << nbitsrfinebintable_) - 1);
      }

      int lutval = -1;
      if (iSeed_ < 6) {  //FIXME should only be one table - but will need coordination with HLS code.
        lutval = innerTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      } else {
        lutval = innerOverlapTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      }

      if (lutval != -1) {
        unsigned int lutwidth = settings_.lutwidthtab(0, iSeed_);
        FPGAWord lookupbits(lutval, lutwidth, true, __LINE__, __FILE__);

        int rzfinebinfirst = lookupbits.bits(0, NFINERZBITS);       //finerz
        int next = lookupbits.bits(NFINERZBITS, 1);                 //use next r/z bin
        int start = lookupbits.bits(NFINERZBITS + 1, nbitsrzbin_);  //rz bin
        int rzdiffmax = lookupbits.bits(NFINERZBITS + 1 + nbitsrzbin_, NFINERZBITS);

        if ((iSeed_ == Seed::D1D2 || iSeed_ == Seed::D3D4) && negdisk) {  //TODO - need to store negative disk
          start += (1 << nbitsrzbin_);
        }
        int last = start + next;

        int nbins = (1 << N_RZBITS);

        unsigned int useregindex = (innerfinephi << innerbend.nbits()) + innerbend.value();
        if (iSeed_ == Seed::D1D2 || iSeed_ == Seed::D3D4 || iSeed_ == Seed::L1D1 || iSeed_ == Seed::L2D1) {
          //FIXME If the lookupbits were rationally organized this would be much simpler
          unsigned int nrbits = 3;
          int ir = ((start & ((1 << (nrbits - 1)) - 1)) << 1) + (rzfinebinfirst >> (NFINERZBITS - 1));
          useregindex = (useregindex << nrbits) + ir;
        }

        unsigned int usereg = useregiontable_.lookup(useregindex);

        tedata.regions_.clear();
        tedata.stub_ = stub;
        tedata.rzbinfirst_ = rzfinebinfirst;
        tedata.start_ = start;
        tedata.innerfinephi_ = innerfinephi;
        tedata.rzdiffmax_ = rzdiffmax;
        tedata.innerbend_ = innerbend;

        std::string mask = "";

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int ireg = 0; ireg < settings_.nvmte(1, iSeed_); ireg++) {
            if (!(usereg & (1 << ireg))) {
              mask = "0" + mask;
              continue;
            }

            if (settings_.debugTracklet()) {
              edm::LogVerbatim("Tracklet") << getName() << " looking for matching stub in bin " << ibin << " with "
                                           << outervmstubs_->nVMStubsBinned(ireg * nbins + ibin) << " stubs";
            }
            assert(ireg * nbins + ibin < outervmstubs_->nBin());
            int nstubs = outervmstubs_->nVMStubsBinned(ireg * nbins + ibin);

            if (nstubs > 0) {
              mask = "1" + mask;
              tedata.regions_.emplace_back(tuple<int, int, int>(ibin - start, ireg, nstubs));
              countteall += nstubs;
            } else {
              mask = "0" + mask;
            }
          }
        }

        if (!tedata.regions_.empty()) {
          ntedata++;
          goodtedata = true;
        }
      }
      istub++;
      if (istub >= innerallstubs_[imem]->nStubs()) {
        istub = 0;
        imem++;
      }
    } else if ((!tebuffernearfull) && imem < imemend && istub == 0) {
      imem++;
    }

    goodtedata___ = goodtedata__;
    goodtedata__ = goodtedata;

    tedata___ = tedata__;
    tedata__ = tedata;

    //
    // stop looping over istep if done
    //

    bool done = true;

    if (imem < imemend || (!tedatabuffer.empty())) {
      done = false;
    }

    for (auto& teunit : teunits_) {
      if (!(teunit.idle() && teunit.empty()))
        done = false;
    }

    if (done) {
      donecount++;
    }

    //FIXME This should be done cleaner... Not too hard, but need to check fully the TEBuffer and TEUnit buffer.
    if (donecount > 4) {
      break;
    }
  }

  //
  // Done with processing - collect performance statistics
  //

  if (settings_.writeMonitorData("TP")) {
    globals_->ofstream("trackletprocessor.txt") << getName() << " " << ninnerstubs   //# inner stubs
                                                << " " << outervmstubs_->nVMStubs()  //# outer stubs
                                                << " " << countteall                 //# pairs tried in TE
                                                << " " << stubpairs                  //# stubs pairs
                                                << " " << countsel                   //# tracklets found
                                                << endl;
  }
}
