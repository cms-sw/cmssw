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

// TrackletProcessorDisplaced
//
// This module takes in collections of stubs within a phi region and a
// displaced seed name and tries to create that displaced seed out of the stubs
//
// Update: Claire Savard, Oct. 2024

TrackletProcessorDisplaced::TrackletProcessorDisplaced(string name, Settings const& settings, Globals* globals)
    : TrackletCalculatorDisplaced(name, settings, globals), innerTable_(settings), innerThirdTable_(settings) {
  innerallstubs_.clear();
  middleallstubs_.clear();
  outerallstubs_.clear();
  innervmstubs_.clear();
  outervmstubs_.clear();

  // set layer/disk types based on input seed name
  initLayerDisksandISeedDisp(layerdisk1_, layerdisk2_, layerdisk3_, iSeed_);

  // get projection tables
  unsigned int region = name.back() - 'A';
  innerTable_.initVMRTable(
      layerdisk1_, TrackletLUT::VMRTableType::inner, region, false);  //projection to next layer/disk
  innerThirdTable_.initVMRTable(
      layerdisk1_, TrackletLUT::VMRTableType::innerthird, region, false);  //projection to third layer/disk

  nbitszfinebintable_ = settings_.vmrlutzbits(layerdisk1_);
  nbitsrfinebintable_ = settings_.vmrlutrbits(layerdisk1_);

  for (unsigned int ilayer = 0; ilayer < N_LAYER; ilayer++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(ilayer), nullptr);
    trackletprojlayers_.push_back(tmp);
  }

  for (unsigned int idisk = 0; idisk < N_DISK; idisk++) {
    vector<TrackletProjectionsMemory*> tmp(settings_.nallstubs(idisk + N_LAYER), nullptr);
    trackletprojdisks_.push_back(tmp);
  }

  // set TC index
  iTC_ = region;
  constexpr int TCIndexMin = 128;
  constexpr int TCIndexMax = 191;
  TCIndex_ = (iSeed_ << 4) + iTC_;
  assert(TCIndex_ >= TCIndexMin && TCIndex_ < TCIndexMax);
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
  if (input == "thirdvmstubin") {
    auto* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    innervmstubs_.push_back(tmp);
    return;
  }
  if (input == "secondvmstubin") {
    auto* tmp = dynamic_cast<VMStubsTEMemory*>(memory);
    assert(tmp != nullptr);
    outervmstubs_.push_back(tmp);
    return;
  }

  throw cms::Exception("BadConfig") << __FILE__ << " " << __LINE__ << " Could not find input : " << input;
}

void TrackletProcessorDisplaced::execute(unsigned int iSector, double phimin, double phimax) {
  unsigned int countall = 0;
  unsigned int countsel = 0;

  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  // loop over the middle stubs in the potential seed
  for (unsigned int midmem = 0; midmem < middleallstubs_.size(); midmem++) {
    for (unsigned int i = 0; i < middleallstubs_[midmem]->nStubs(); i++) {
      const Stub* midallstub = middleallstubs_[midmem]->getStub(i);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "In " << getName() << " have middle stub";
      }

      // get r/z index of the middle stub
      int indexz = (((1 << (midallstub->z().nbits() - 1)) + midallstub->z().value()) >>
                    (midallstub->z().nbits() - nbitszfinebintable_));
      int indexr = -1;
      bool negdisk = (midallstub->disk().value() < 0);  // check if disk in negative z region
      if (layerdisk1_ >= LayerDisk::D1) {               // if a disk
        if (negdisk)
          indexz = (1 << nbitszfinebintable_) - indexz;
        indexr = midallstub->r().value();
        if (midallstub->isPSmodule()) {
          indexr = midallstub->r().value() >> (midallstub->r().nbits() - nbitsrfinebintable_);
        }
      } else {  // else a layer
        indexr = (((1 << (midallstub->r().nbits() - 1)) + midallstub->r().value()) >>
                  (midallstub->r().nbits() - nbitsrfinebintable_));
      }

      assert(indexz >= 0);
      assert(indexr >= 0);
      assert(indexz < (1 << nbitszfinebintable_));
      assert(indexr < (1 << nbitsrfinebintable_));

      // create lookupbits that define projections from middle stub
      unsigned int lutwidth = settings_.lutwidthtabextended(0, iSeed_);
      int lutval = -1;
      const auto& lutshift = innerTable_.nbits();
      lutval = innerTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      int lutval2 = innerThirdTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      if (lutval != -1 && lutval2 != -1)
        lutval += (lutval2 << lutshift);
      if (lutval == -1)
        continue;
      FPGAWord lookupbits(lutval, lutwidth, true, __LINE__, __FILE__);

      // get r/z bins for projection into outer layer/disk
      int nbitsrzbin = N_RZBITS;
      if (iSeed_ == Seed::D1D2L2)
        nbitsrzbin--;
      int rzbinfirst = lookupbits.bits(0, NFINERZBITS);
      int next = lookupbits.bits(NFINERZBITS, 1);
      int rzdiffmax = lookupbits.bits(NFINERZBITS + 1 + nbitsrzbin, NFINERZBITS);

      int start = lookupbits.bits(NFINERZBITS + 1, nbitsrzbin);  // first rz bin projection
      if (iSeed_ == Seed::D1D2L2 && negdisk)                     // if projecting into disk
        start += (1 << nbitsrzbin);
      int last = start + next;  // last rz bin projection

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "Will look in r/z bins for outer stub " << start << " to " << last << endl;
      }

      // loop over outer stubs that the middle stub can project to
      for (int ibin = start; ibin <= last; ibin++) {
        for (unsigned int outmem = 0; outmem < outervmstubs_.size(); outmem++) {
          for (unsigned int j = 0; j < outervmstubs_[outmem]->nVMStubsBinned(ibin); j++) {
            if (settings_.debugTracklet())
              edm::LogVerbatim("Tracklet") << "In " << getName() << " have outer stub" << endl;

            const VMStubTE& outvmstub = outervmstubs_[outmem]->getVMStubTEBinned(ibin, j);

            // check if r/z of outer stub is within projection range
            int rzbin = (outvmstub.vmbits().value() & (settings_.NLONGVMBINS() - 1));
            if (start != ibin)
              rzbin += 8;
            if (rzbin < rzbinfirst || rzbin - rzbinfirst > rzdiffmax) {
              if (settings_.debugTracklet()) {
                edm::LogVerbatim("Tracklet") << "Outer stub rejected because of wrong r/z bin";
              }
              continue;
            }

            // get r/z bins for projection into third layer/disk
            int nbitsrzbin_ = N_RZBITS;
            int next_ = lookupbits.bits(lutshift + NFINERZBITS, 1);

            int start_ = lookupbits.bits(lutshift + NFINERZBITS + 1, nbitsrzbin_);  // first rz bin projection
            if (iSeed_ == Seed::D1D2L2 && negdisk)  // if projecting from disk into layer
              start_ = settings_.NLONGVMBINS() - 1 - start_ - next_;
            int last_ = start_ + next_;  // last rz bin projection

            if (settings_.debugTracklet()) {
              edm::LogVerbatim("Tracklet")
                  << "Will look in rz bins for inner stub " << start_ << " to " << last_ << endl;
            }

            // loop over inner stubs that the middle stub can project to
            for (int ibin_ = start_; ibin_ <= last_; ibin_++) {
              for (unsigned int inmem = 0; inmem < innervmstubs_.size(); inmem++) {
                for (unsigned int k = 0; k < innervmstubs_[inmem]->nVMStubsBinned(ibin_); k++) {
                  if (settings_.debugTracklet())
                    edm::LogVerbatim("Tracklet") << "In " << getName() << " have inner stub" << endl;

                  const VMStubTE& invmstub = innervmstubs_[inmem]->getVMStubTEBinned(ibin_, k);

                  countall++;

                  const Stub* innerFPGAStub = invmstub.stub();
                  const Stub* middleFPGAStub = midallstub;
                  const Stub* outerFPGAStub = outvmstub.stub();

                  const L1TStub* innerStub = innerFPGAStub->l1tstub();
                  const L1TStub* middleStub = middleFPGAStub->l1tstub();
                  const L1TStub* outerStub = outerFPGAStub->l1tstub();

                  if (settings_.debugTracklet()) {
                    edm::LogVerbatim("Tracklet")
                        << "triplet seeding\n"
                        << innerFPGAStub->strbare() << middleFPGAStub->strbare() << outerFPGAStub->strbare()
                        << innerStub->stubword() << middleStub->stubword() << outerStub->stubword()
                        << innerFPGAStub->layerdisk() << middleFPGAStub->layerdisk() << outerFPGAStub->layerdisk();
                    edm::LogVerbatim("Tracklet")
                        << "TrackletCalculatorDisplaced execute " << getName() << "[" << iSector_ << "]";
                  }

                  // check if the seed made from the 3 stubs is valid
                  bool accept = false;
                  if (iSeed_ == Seed::L2L3L4 || iSeed_ == Seed::L4L5L6)
                    accept = LLLSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);
                  else if (iSeed_ == Seed::L2L3D1)
                    accept = LLDSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);
                  else if (iSeed_ == Seed::D1D2L2)
                    accept = DDLSeeding(innerFPGAStub, innerStub, middleFPGAStub, middleStub, outerFPGAStub, outerStub);

                  if (accept)
                    countsel++;

                  if (settings_.debugTracklet()) {
                    edm::LogVerbatim("Tracklet") << "TrackletCalculatorDisplaced execute done";
                  }
                  if (countall >= settings_.maxStep("TPD"))
                    break;
                }
                if (countall >= settings_.maxStep("TPD"))
                  break;
              }
              if (countall >= settings_.maxStep("TPD"))
                break;
            }
            if (countall >= settings_.maxStep("TPD"))
              break;
          }
          if (countall >= settings_.maxStep("TPD"))
            break;
        }
        if (countall >= settings_.maxStep("TPD"))
          break;
      }
      if (countall >= settings_.maxStep("TPD"))
        break;
    }
    if (countall >= settings_.maxStep("TPD"))
      break;
  }

  if (settings_.writeMonitorData("TPD")) {
    globals_->ofstream("trackletprocessordisplaced.txt") << getName() << " " << countall << " " << countsel << endl;
  }
}
