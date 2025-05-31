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
// Update: Claire Savard, Nov. 2024

TrackletProcessorDisplaced::TrackletProcessorDisplaced(string name, Settings const& settings, Globals* globals)
    : TrackletCalculatorDisplaced(name, settings, globals),
      trpbuffer_(CircularBuffer<TrpEData>(3), 0, 0, 0, 0),
      innerTable_(settings),
      innerThirdTable_(settings) {
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
  TCIndex_ = (iSeed_ << settings.nbitsseed()) + iTC_;

  maxStep_ = settings_.maxStep("TPD");
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
  phimin_ = phimin;
  phimax_ = phimax;
  iSector_ = iSector;

  unsigned int countall = 0;
  unsigned int countsel = 0;
  int donecount = 0;

  // set the triplet engine units and buffer
  TripletEngineUnit trpunit(&settings_, layerdisk1_, layerdisk2_, layerdisk3_, iSeed_, innervmstubs_, outervmstubs_);
  trpunits_.resize(settings_.trpunits(iSeed_), trpunit);
  trpbuffer_ = tuple<CircularBuffer<TrpEData>, unsigned int, unsigned int, unsigned int, unsigned int>(
      CircularBuffer<TrpEData>(3), 0, 0, 0, middleallstubs_.size());

  // reset the trpunits
  for (auto& trpunit : trpunits_) {
    trpunit.reset();
  }

  // reset the tebuffer
  std::get<0>(trpbuffer_).reset();
  std::get<1>(trpbuffer_) = 0;
  std::get<2>(trpbuffer_) = std::get<3>(trpbuffer_);

  TrpEData trpdata;
  TrpEData trpdata__;
  TrpEData trpdata___;
  bool goodtrpdata = false;
  bool goodtrpdata__ = false;
  bool goodtrpdata___ = false;

  bool trpbuffernearfull;
  for (unsigned int istep = 0; istep < maxStep_; istep++) {
    CircularBuffer<TrpEData>& trpdatabuffer = std::get<0>(trpbuffer_);
    trpbuffernearfull = trpdatabuffer.nearfull();

    //
    // First block here checks if there is a trpunit with data that should be used
    // to calculate the tracklet parameters
    //

    // set pointer to the last filled trpunit
    TripletEngineUnit* trpunitptr = nullptr;
    for (auto& trpunit : trpunits_) {
      trpunit.setNearFull();
      if (!trpunit.empty()) {
        trpunitptr = &trpunit;
      }
    }

    if (trpunitptr != nullptr) {
      auto stubtriplet = trpunitptr->read();

      countall++;

      const Stub* innerFPGAStub = std::get<0>(stubtriplet);
      const Stub* middleFPGAStub = std::get<1>(stubtriplet);
      const Stub* outerFPGAStub = std::get<2>(stubtriplet);

      const L1TStub* innerStub = innerFPGAStub->l1tstub();
      const L1TStub* middleStub = middleFPGAStub->l1tstub();
      const L1TStub* outerStub = outerFPGAStub->l1tstub();

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "TrackletProcessorDisplaced execute " << getName() << "[" << iSector_ << "]";
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
    // The second block fills the trpunit if data in buffer and process TripletEngineUnit step
    //
    //

    bool notemptytrpbuffer = !trpdatabuffer.empty();
    for (auto& trpunit : trpunits_) {
      if (trpunit.idle() && notemptytrpbuffer) {  // only fill one idle unit every step
        trpunit.init(std::get<0>(trpbuffer_).read());
        notemptytrpbuffer = false;  //prevent initializing another triplet engine unit
      }
      trpunit.step();
    }

    //
    // The third block here checks if we have input stubs to process
    //
    //

    if (goodtrpdata___)
      trpdatabuffer.store(trpdata___);
    goodtrpdata = false;

    unsigned int& istub = std::get<1>(trpbuffer_);
    unsigned int& midmem = std::get<2>(trpbuffer_);
    unsigned int midmemend = std::get<4>(trpbuffer_);

    if ((!trpbuffernearfull) && midmem < midmemend && istub < middleallstubs_[midmem]->nStubs()) {
      const Stub* stub = middleallstubs_[midmem]->getStub(istub);

      if (settings_.debugTracklet()) {
        edm::LogVerbatim("Tracklet") << "In " << getName() << " have middle stub";
      }

      // get r/z index of the middle stub
      int indexz = (((1 << (stub->z().nbits() - 1)) + stub->z().value()) >> (stub->z().nbits() - nbitszfinebintable_));
      int indexr = -1;
      bool negdisk = (stub->disk().value() < 0);  // check if disk in negative z region
      if (layerdisk1_ >= LayerDisk::D1) {         // if a disk
        if (negdisk)
          indexz = (1 << nbitszfinebintable_) - indexz;
        indexr = stub->r().value();
        if (stub->isPSmodule()) {
          indexr = stub->r().value() >> (stub->r().nbits() - nbitsrfinebintable_);
        }
      } else {  // else a layer
        indexr = (((1 << (stub->r().nbits() - 1)) + stub->r().value()) >> (stub->r().nbits() - nbitsrfinebintable_));
      }

      // create lookupbits that define projections from middle stub
      int lutval = -1;
      const auto& lutshift = innerTable_.nbits();
      lutval = innerTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      int lutval2 = innerThirdTable_.lookup((indexz << nbitsrfinebintable_) + indexr);
      if (lutval != -1 && lutval2 != -1)
        lutval += (lutval2 << lutshift);

      if (lutval != -1) {
        unsigned int lutwidth = settings_.lutwidthtabextended(0, iSeed_);
        FPGAWord lookupbits(lutval, lutwidth, true, __LINE__, __FILE__);

        // get r/z bins for projection into outer layer/disk
        int nbitsrzbin_out = N_RZBITS;
        if (iSeed_ == Seed::D1D2L2)
          nbitsrzbin_out--;
        int rzbinfirst_out = lookupbits.bits(0, NFINERZBITS);
        int rzdiffmax_out = lookupbits.bits(NFINERZBITS + 1 + nbitsrzbin_out, NFINERZBITS);
        int start_out = lookupbits.bits(NFINERZBITS + 1, nbitsrzbin_out);  // first rz bin projection
        int next_out = lookupbits.bits(NFINERZBITS, 1);
        if (iSeed_ == Seed::D1D2L2 && negdisk)  // if projecting into disk
          start_out += (1 << nbitsrzbin_out);
        int last_out = start_out + next_out;  // last rz bin projection

        // get r/z bins for projection into third (inner) layer/disk
        int nbitsrzbin_in = N_RZBITS;
        int start_in = lookupbits.bits(lutshift + NFINERZBITS + 1, nbitsrzbin_in);  // first rz bin projection
        int next_in = lookupbits.bits(lutshift + NFINERZBITS, 1);
        if (iSeed_ == Seed::D1D2L2 && negdisk)  // if projecting from disk into layer
          start_in = settings_.NLONGVMBINS() - 1 - start_in - next_in;
        int last_in = start_in + next_in;  // last rz bin projection

        // fill trpdata with projection info of middle stub
        trpdata.stub_ = stub;
        trpdata.rzbinfirst_out_ = rzbinfirst_out;
        trpdata.rzdiffmax_out_ = rzdiffmax_out;
        trpdata.start_out_ = start_out;
        trpdata.start_in_ = start_in;

        // fill projection bins info for single engine unit
        trpdata.projbin_out_.clear();
        trpdata.projbin_in_.clear();
        for (int ibin_out = start_out; ibin_out <= last_out; ibin_out++) {
          for (unsigned int outmem = 0; outmem < outervmstubs_.size(); outmem++) {
            int nstubs_out = outervmstubs_[outmem]->nVMStubsBinned(ibin_out);
            if (nstubs_out > 0)
              trpdata.projbin_out_.emplace_back(tuple<int, int, int>(ibin_out - start_out, outmem, nstubs_out));
          }
        }
        for (int ibin_in = start_in; ibin_in <= last_in; ibin_in++) {
          for (unsigned int inmem = 0; inmem < innervmstubs_.size(); inmem++) {
            int nstubs_in = innervmstubs_[inmem]->nVMStubsBinned(ibin_in);
            if (nstubs_in > 0)
              trpdata.projbin_in_.emplace_back(tuple<int, int, int>(ibin_in - start_in, inmem, nstubs_in));
          }
        }

        if (!trpdata.projbin_in_.empty() && !trpdata.projbin_out_.empty()) {
          goodtrpdata = true;
        }
      }

      istub++;
      if (istub >= middleallstubs_[midmem]->nStubs()) {
        istub = 0;
        midmem++;
      }

    } else if ((!trpbuffernearfull) && midmem < midmemend && istub == 0)
      midmem++;

    goodtrpdata___ = goodtrpdata__;
    goodtrpdata__ = goodtrpdata;

    trpdata___ = trpdata__;
    trpdata__ = trpdata;

    //
    // stop looping over istep if done
    //

    bool done = true;

    if (midmem < midmemend || (!trpdatabuffer.empty())) {
      done = false;
    }

    for (auto& trpunit : trpunits_) {
      if (!(trpunit.idle() && trpunit.empty()))
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

  if (settings_.writeMonitorData("TPD")) {
    globals_->ofstream("trackletprocessordisplaced.txt")
        << getName() << " " << countall << " " << countsel << std::endl;
  }
}
