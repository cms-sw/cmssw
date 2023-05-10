#include "L1Trigger/TrackFindingTracklet/interface/TrackletProcessorDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/Settings.h"
#include "L1Trigger/TrackFindingTracklet/interface/Globals.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllInnerStubsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/Util.h"
#include "L1Trigger/TrackFindingTracklet/interface/IMATH_TrackletCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Math/interface/deltaPhi.h"

#include <utility>
#include <tuple>

using namespace std;
using namespace trklet;

TrackletProcessorDisplaced::TrackletProcessorDisplaced(string name, Settings const& settings, Globals* globals)
    : TrackletCalculatorDisplaced(name, settings, globals),
      innerTable_(settings),
      innerOverlapTable_(settings),
      innerThirdTable_(settings) {
  innerallstubs_.clear();
  middleallstubs_.clear();
  outerallstubs_.clear();
  stubpairs_.clear();
  innervmstubs_.clear();
  outervmstubs_.clear();

  const unsigned layerdiskPosInName = 4;
  const unsigned regionPosInName1 = 9;

  // iAllStub_ = -1;
  layerdisk_ = initLayerDisk(layerdiskPosInName);

  unsigned int region = name.substr(1)[regionPosInName1] - 'A';
  // assert(region < settings_.nallstubs(layerdisk_));

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 ||
      layerdisk_ == LayerDisk::L5 || layerdisk_ == LayerDisk::D1 || layerdisk_ == LayerDisk::D3) {
    innerTable_.initVMRTable(layerdisk_, TrackletLUT::VMRTableType::inner, region);  //projection to next layer/disk
  }

  if (layerdisk_ == LayerDisk::L1 || layerdisk_ == LayerDisk::L2) {
    innerOverlapTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::inneroverlap, region);  //projection to disk from layer
  }

  if (layerdisk_ == LayerDisk::L2 || layerdisk_ == LayerDisk::L3 || layerdisk_ == LayerDisk::L5 ||
      layerdisk_ == LayerDisk::D1) {
    innerThirdTable_.initVMRTable(
        layerdisk_, TrackletLUT::VMRTableType::innerthird, region);  //projection to third layer/disk
  }

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk_);

  for (unsigned int ilayer = 0; ilayer < N_LAYER; ilayer++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(ilayer), nullptr);
    trackletprojlayers_.push_back(tmp);
  }

  for (unsigned int idisk = 0; idisk < N_DISK; idisk++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(idisk + N_LAYER), nullptr);
    trackletprojdisks_.push_back(tmp);
  }

  // initLayerDisksandISeed(layerdisk1_, layerdisk2_, iSeed_);

  layer_ = 0;
  disk_ = 0;
  layer1_ = 0;
  layer2_ = 0;
  layer3_ = 0;
  disk1_ = 0;
  disk2_ = 0;
  disk3_ = 0;

  constexpr unsigned layerPosInName1 = 4;
  constexpr unsigned diskPosInName1 = 4;
  constexpr unsigned layer1PosInName1 = 4;
  constexpr unsigned disk1PosInName1 = 4;
  constexpr unsigned layer2PosInName1 = 6;
  constexpr unsigned disk2PosInName1 = 6;

  string name1 = name.substr(1);  //this is to correct for "TPD" having one more letter then "TP"
  if (name1[3] == 'L')
    layer_ = name1[layerPosInName1] - '0';
  if (name1[3] == 'D')
    disk_ = name1[diskPosInName1] - '0';

  if (name1[3] == 'L')
    layer1_ = name1[layer1PosInName1] - '0';
  if (name1[3] == 'D')
    disk1_ = name1[disk1PosInName1] - '0';
  if (name1[5] == 'L')
    layer2_ = name1[layer2PosInName1] - '0';
  if (name1[5] == 'D')
    disk2_ = name1[disk2PosInName1] - '0';

  // set TC index
  iSeed_ = 0;

  int iTC = name1[regionPosInName1] - 'A';

  if (name1.substr(3, 6) == "L3L4L2") {
    iSeed_ = 8;
    layer3_ = 2;
  } else if (name1.substr(3, 6) == "L5L6L4") {
    iSeed_ = 9;
    layer3_ = 4;
  } else if (name1.substr(3, 6) == "L2L3D1") {
    iSeed_ = 10;
    disk3_ = 1;
  } else if (name1.substr(3, 6) == "D1D2L2") {
    iSeed_ = 11;
    layer3_ = 2;
  }
  assert(iSeed_ != 0);

  constexpr int TCIndexMin = 128;
  constexpr int TCIndexMax = 191;

  TCIndex_ = (iSeed_ << 4) + iTC;
  assert(TCIndex_ >= TCIndexMin && TCIndex_ < TCIndexMax);

  assert((layer_ != 0) || (disk_ != 0));
}

void TrackletProcessorDisplaced::addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory) {
  outputProj = dynamic_cast<TrackletProjectionsMemory*>(memory);
  assert(outputProj != nullptr);
}

void TrackletProcessorDisplaced::addOutput(MemoryBase* memory, string output) {
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

    constexpr unsigned layerdiskPosInprojout = 8;
    constexpr unsigned phiPosInprojout = 12;

    unsigned int layerdisk = output[layerdiskPosInprojout] - '1';  //layer or disk counting from 0
    unsigned int phiregion = output[phiPosInprojout] - 'A';        //phiregion counting from 0

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

void TrackletProcessorDisplaced::addInput(MemoryBase* memory, string input) {
  if (settings_.writetrace()) {
    edm::LogVerbatim("Tracklet") << "In " << name_ << " adding input from " << memory->getName() << " to input "
                                 << input;
  }

  if (input == "thirdallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    innerallstubs_.push_back(tmp);
    return;
  }
  if (input == "firstallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    middleallstubs_.push_back(tmp);
    return;
  }
  if (input == "secondallstubin") {
    auto* tmp = dynamic_cast<AllStubsMemory*>(memory);
    assert(tmp != nullptr);
    outerallstubs_.push_back(tmp);
    return;
  }
  if (input.substr(0, 8) == "stubpair") {
    auto* tmp = dynamic_cast<StubPairsMemory*>(memory);
    assert(tmp != nullptr);
    stubpairs_.push_back(tmp);
    return;
  }

  if (input == "thirdvmstubin") {
    auto* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    innervmstubs_.push_back(tmp);
    return;
  }
  if (input == "secondvmstubin") {
    auto* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    // outervmstubs_ = tmp;
    outervmstubs_.push_back(tmp);
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletProcessorDisplaced::execute(unsigned int iSector, double phimin, double phimax) {
  // unsigned int nThirdStubs = 0;
  // unsigned int nOuterStubs = 0;
  count_ = 0;

  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  assert(!innerallstubs_.empty());
  assert(!middleallstubs_.empty());
  assert(!outerallstubs_.empty());
  assert(!innervmstubs_.empty());
  assert(!outervmstubs_.empty());
  assert(stubpairs_.empty());

  for (auto& iInnerMem : middleallstubs_) {
    assert(iInnerMem->nStubs() == iInnerMem->nStubs());
    for (unsigned int j = 0; j < iInnerMem->nStubs(); j++) {
      const Stub* firstallstub = iInnerMem->getStub(j);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "In " << getName() << " have first stub\n";
      }

      int inner = 0;
      bool negdisk = (firstallstub->disk().value() < 0);
      int indexz = (((1 << (firstallstub->z().nbits() - 1)) + firstallstub->z().value()) >>
                    (firstallstub->z().nbits() - nbitszfinebintable_));
      int indexr = -1;
      if (layerdisk_ > (N_LAYER - 1)) {
        if (negdisk) {
          indexz = (1 << nbitszfinebintable_) - indexz;
        }
        indexr = firstallstub->r().value();
        if (firstallstub->isPSmodule()) {
          indexr = firstallstub->r().value() >> (firstallstub->r().nbits() - nbitsrfinebintable_);
        }
      } else {
        //Take the top nbitsfinebintable_ bits of the z coordinate. The & is to handle the negative z values.
        indexr = (((1 << (firstallstub->r().nbits() - 1)) + firstallstub->r().value()) >>
                  (firstallstub->r().nbits() - nbitsrfinebintable_));
      }

      assert(indexz >= 0);
      assert(indexr >= 0);
      assert(indexz < (1 << nbitszfinebintable_));
      assert(indexr < (1 << nbitsrfinebintable_));

      // int melut = meTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      // assert(melut >= 0);

      unsigned int lutwidth = settings_.lutwidthtab(inner, iSeed_);
      if (settings_.extended()) {
        lutwidth = settings_.lutwidthtabextended(inner, iSeed_);
      }

      int lutval = -999;

      if (iSeed_ < Seed::L1D1 || iSeed_ > Seed::L2D1) {
        lutval = innerTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      } else {
        lutval = innerOverlapTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      }

      if (lutval == -1)
        continue;
      if (settings_.extended() &&
          (iSeed_ == Seed::L2L3L4 || iSeed_ == Seed::L4L5L6 || iSeed_ == Seed::D1D2L2 || iSeed_ == Seed::L2L3D1)) {
        int lutval2 = innerThirdTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
        if (lutval2 != -1)
          lutval += (lutval2 << 10);
      }

      assert(lutval >= 0);
      // assert(lutwidth > 0);

      FPGAWord binlookup(lutval, lutwidth, true, __LINE__, __FILE__);

      if ((layer1_ == 3 && layer2_ == 4) || (layer1_ == 5 && layer2_ == 6)) {
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " Layer-layer pair\n";
        }

        constexpr int andlookupbits = 1023;
        constexpr int shiftzdiffmax = 7;
        constexpr int andnewbin = 127;
        constexpr int divbin = 8;
        constexpr int andzbinfirst = 7;
        constexpr int shiftstart = 1;
        constexpr int andlast = 1;
        constexpr int maxlast = 8;

        int lookupbits = binlookup.value() & andlookupbits;
        int zdiffmax = (lookupbits >> shiftzdiffmax);
        int newbin = (lookupbits & andnewbin);
        int bin = newbin / divbin;

        int zbinfirst = newbin & andzbinfirst;

        int start = (bin >> shiftstart);
        int last = start + (bin & andlast);

        assert(last < maxlast);

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Will look in zbins " << start << " to " << last << endl;
        }

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int m = 0; m < outervmstubs_.size(); m++) {
            for (unsigned int j = 0; j < outervmstubs_.at(m)->nVMStubsBinned(ibin); j++) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet")
                    << "In " << getName() << " have second stub(1) " << ibin << " " << j << endl;
              }

              const VMStubTE& secondvmstub = outervmstubs_.at(m)->getVMStubTEBinned(ibin, j);

              int zbin = (secondvmstub.vmbits().value() & 7);
              if (start != ibin)
                zbin += 8;
              if (zbin < zbinfirst || zbin - zbinfirst > zdiffmax) {
                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet") << "Stubpair rejected because of wrong zbin";
                }
                continue;
              }

              if ((layer2_ == 4 && layer3_ == 2) || (layer2_ == 6 && layer3_ == 4)) {
                constexpr int vmbitshift = 10;
                constexpr int andlookupbits_ = 1023;
                constexpr int andnewbin_ = 127;
                constexpr int divbin_ = 8;
                constexpr int shiftstart_ = 1;
                constexpr int andlast_ = 1;

                int lookupbits_ = (int)((binlookup.value() >> vmbitshift) & andlookupbits_);
                int newbin_ = (lookupbits_ & andnewbin_);
                int bin_ = newbin_ / divbin_;

                int start_ = (bin_ >> shiftstart_);
                int last_ = start_ + (bin_ & andlast_);

                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet")
                      << "Will look in zbins for third stub" << start_ << " to " << last_ << endl;
                }

                for (int ibin_ = start_; ibin_ <= last_; ibin_++) {
                  for (unsigned int k = 0; k < innervmstubs_.size(); k++) {
                    for (unsigned int l = 0; l < innervmstubs_.at(k)->nVMStubsBinned(ibin_); l++) {
                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "In " << getName() << " have third stub\n";
                      }

                      const VMStubTE& thirdvmstub = innervmstubs_.at(k)->getVMStubTEBinned(ibin_, l);

                      const Stub* innerFPGAStub = firstallstub;
                      const Stub* middleFPGAStub = secondvmstub.stub();
                      const Stub* outerFPGAStub = thirdvmstub.stub();

                      const L1TStub* innerStub = innerFPGAStub->l1tstub();
                      const L1TStub* middleStub = middleFPGAStub->l1tstub();
                      const L1TStub* outerStub = outerFPGAStub->l1tstub();

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "LLL seeding\n"
                            << innerFPGAStub->strbare() << middleFPGAStub->strbare() << outerFPGAStub->strbare()
                            << innerStub->stubword() << middleStub->stubword() << outerStub->stubword()
                            << innerFPGAStub->layerdisk() << middleFPGAStub->layerdisk() << outerFPGAStub->layerdisk();
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "TrackletCalculatorDisplaced execute " << getName() << "[" << iSector_ << "]";
                      }

                      if (innerFPGAStub->layerdisk() >= N_LAYER && middleFPGAStub->layerdisk() >= N_LAYER &&
                                 outerFPGAStub->layerdisk() >= N_LAYER) {
                        throw cms::Exception("LogicError") << __FILE__ << " " << __LINE__ << " Invalid seeding!";
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute done";
                      }
                    }
                  }
                }
              }
            }
          }
        }

      } else if (layer1_ == 2 && layer2_ == 3) {
        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << getName() << " Layer-layer pair";
        }

        constexpr int andlookupbits = 1023;
        constexpr int shiftzdiffmax = 7;
        constexpr int andnewbin = 127;
        constexpr int divbin = 8;
        constexpr int andzbinfirst = 7;
        constexpr int shiftstart = 1;
        constexpr int andlast = 1;
        constexpr int maxlast = 8;

        int lookupbits = binlookup.value() & andlookupbits;
        int zdiffmax = (lookupbits >> shiftzdiffmax);
        int newbin = (lookupbits & andnewbin);
        int bin = newbin / divbin;

        int zbinfirst = newbin & andzbinfirst;

        int start = (bin >> shiftstart);
        int last = start + (bin & andlast);

        assert(last < maxlast);

        if (settings_.debugTracklet()) {
          edm::LogVerbatim("Tracklet") << "Will look in zbins " << start << " to " << last;
        }

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int m = 0; m < outervmstubs_.size(); m++) {
            for (unsigned int j = 0; j < outervmstubs_.at(m)->nVMStubsBinned(ibin); j++) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << "In " << getName() << " have second stub(1) " << ibin << " " << j;
              }

              const VMStubTE& secondvmstub = outervmstubs_.at(m)->getVMStubTEBinned(ibin, j);

              int zbin = (secondvmstub.vmbits().value() & 7);
              if (start != ibin)
                zbin += 8;
              if (zbin < zbinfirst || zbin - zbinfirst > zdiffmax) {
                if (settings_.debugTracklet()) {
                  edm::LogVerbatim("Tracklet") << "Stubpair rejected because of wrong zbin";
                }
                continue;
              }

              if (layer2_ == 3 && disk3_ == 1) {
                constexpr int vmbitshift = 10;
                constexpr int andlookupbits_ = 1023;
                constexpr int andnewbin_ = 127;
                constexpr int divbin_ = 8;
                constexpr int shiftstart_ = 1;
                constexpr int andlast_ = 1;

                int lookupbits_ = (int)((binlookup.value() >> vmbitshift) & andlookupbits_);
                int newbin_ = (lookupbits_ & andnewbin_);
                int bin_ = newbin_ / divbin_;

                int start_ = (bin_ >> shiftstart_);
                int last_ = start_ + (bin_ & andlast_);

                for (int ibin_ = start_; ibin_ <= last_; ibin_++) {
                  for (unsigned int k = 0; k < innervmstubs_.size(); k++) {
                    for (unsigned int l = 0; l < innervmstubs_.at(k)->nVMStubsBinned(ibin_); l++) {
                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "In " << getName() << " have third stub";
                      }

                      const VMStubTE& thirdvmstub = innervmstubs_.at(k)->getVMStubTEBinned(ibin_, l);

                      const Stub* innerFPGAStub = firstallstub;
                      const Stub* middleFPGAStub = secondvmstub.stub();
                      const Stub* outerFPGAStub = thirdvmstub.stub();

                      const L1TStub* innerStub = innerFPGAStub->l1tstub();
                      const L1TStub* middleStub = middleFPGAStub->l1tstub();
                      const L1TStub* outerStub = outerFPGAStub->l1tstub();

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "LLD seeding\n"
                            << innerFPGAStub->strbare() << middleFPGAStub->strbare() << outerFPGAStub->strbare()
                            << innerStub->stubword() << middleStub->stubword() << outerStub->stubword()
                            << innerFPGAStub->layerdisk() << middleFPGAStub->layerdisk() << outerFPGAStub->layerdisk();
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "TrackletCalculatorDisplaced execute " << getName() << "[" << iSector_ << "]";
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute done";
                      }
                    }
                  }
                }
              }
            }
          }
        }

      } else if (disk1_ == 1 && disk2_ == 2) {
        if (settings_.debugTracklet())
          edm::LogVerbatim("Tracklet") << getName() << " Disk-disk pair";

        constexpr int andlookupbits = 511;
        constexpr int shiftrdiffmax = 6;
        constexpr int andnewbin = 63;
        constexpr int divbin = 8;
        constexpr int andrbinfirst = 7;
        constexpr int shiftstart = 1;
        constexpr int andlast = 1;
        constexpr int maxlast = 8;

        int lookupbits = binlookup.value() & andlookupbits;
        bool negdisk = firstallstub->disk().value() < 0;
        int rdiffmax = (lookupbits >> shiftrdiffmax);
        int newbin = (lookupbits & andnewbin);
        int bin = newbin / divbin;

        int rbinfirst = newbin & andrbinfirst;

        int start = (bin >> shiftstart);
        if (negdisk)
          start += 4;
        int last = start + (bin & andlast);
        assert(last < maxlast);

        for (int ibin = start; ibin <= last; ibin++) {
          for (unsigned int m = 0; m < outervmstubs_.size(); m++) {
            if (settings_.debugTracklet()) {
              edm::LogVerbatim("Tracklet")
                  << getName() << " looking for matching stub in " << outervmstubs_.at(m)->getName()
                  << " in bin = " << ibin << " with " << outervmstubs_.at(m)->nVMStubsBinned(ibin) << " stubs";
            }

            for (unsigned int j = 0; j < outervmstubs_.at(m)->nVMStubsBinned(ibin); j++) {
              const VMStubTE& secondvmstub = outervmstubs_.at(m)->getVMStubTEBinned(ibin, j);
              int rbin = (secondvmstub.vmbits().value() & 7);
              if (start != ibin)
                rbin += 8;
              if (rbin < rbinfirst)
                continue;
              if (rbin - rbinfirst > rdiffmax)
                continue;

              if (disk2_ == 2 && layer3_ == 2) {
                constexpr int vmbitshift = 10;
                constexpr int andlookupbits_ = 1023;
                constexpr int andnewbin_ = 127;
                constexpr int divbin_ = 8;
                constexpr int shiftstart_ = 1;
                constexpr int andlast_ = 1;

                int lookupbits_ = (int)((binlookup.value() >> vmbitshift) & andlookupbits_);
                int newbin_ = (lookupbits_ & andnewbin_);
                int bin_ = newbin_ / divbin_;

                int start_ = (bin_ >> shiftstart_);
                int last_ = start_ + (bin_ & andlast_);

                if (firstallstub->disk().value() < 0) {  //TODO - negative disk should come from memory
                  start_ = settings_.NLONGVMBINS() - last_ - 1;
                  last_ = settings_.NLONGVMBINS() - start_ - 1;
                }

                for (int ibin_ = start_; ibin_ <= last_; ibin_++) {
                  for (unsigned int k = 0; k < innervmstubs_.size(); k++) {
                    for (unsigned int l = 0; l < innervmstubs_.at(k)->nVMStubsBinned(ibin_); l++) {
                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "In " << getName() << " have third stub";
                      }

                      const VMStubTE& thirdvmstub = innervmstubs_.at(k)->getVMStubTEBinned(ibin_, l);

                      const Stub* innerFPGAStub = firstallstub;
                      const Stub* middleFPGAStub = secondvmstub.stub();
                      const Stub* outerFPGAStub = thirdvmstub.stub();

                      const L1TStub* innerStub = innerFPGAStub->l1tstub();
                      const L1TStub* middleStub = middleFPGAStub->l1tstub();
                      const L1TStub* outerStub = outerFPGAStub->l1tstub();

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "DDL seeding\n"
                            << innerFPGAStub->strbare() << middleFPGAStub->strbare() << outerFPGAStub->strbare()
                            << innerStub->stubword() << middleStub->stubword() << outerStub->stubword()
                            << innerFPGAStub->layerdisk() << middleFPGAStub->layerdisk() << outerFPGAStub->layerdisk();
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet")
                            << "TrackletCalculatorDisplaced execute " << getName() << "[" << iSector_ << "]";
                      }

                      if (settings_.debugTracklet()) {
                        edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute done";
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
