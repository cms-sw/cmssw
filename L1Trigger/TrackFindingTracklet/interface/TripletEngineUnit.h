#ifndef L1Trigger_TrackFindingTracklet_interface_TripletEngineUnit_h
#define L1Trigger_TrackFindingTracklet_interface_TripletEngineUnit_h

#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

#include <cassert>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;

  struct TrpEData {
    const Stub* stub_;
    int start_out_;
    int start_in_;
    int rzbinfirst_out_;
    int rzdiffmax_out_;
    std::vector<std::tuple<int, int, int> > projbin_out_;  // next z/r bin; outer stub mem; nstub
    std::vector<std::tuple<int, int, int> > projbin_in_;   // next z/r bin; inner stub mem; nstub
  };

  class TripletEngineUnit {
  public:
    TripletEngineUnit(const Settings* const settings,
                      unsigned int layerdisk1,
                      unsigned int layerdisk2,
                      unsigned int layerdisk3,
                      unsigned int iSeed,
                      std::vector<VMStubsTEMemory*> innervmstubs,
                      std::vector<VMStubsTEMemory*> outervmstubs);

    ~TripletEngineUnit() = default;

    void init(const TrpEData& trpdata);

    bool getGoodTriplet() { return goodtriplet__; }

    bool empty() const { return candtriplets_.empty(); }

    const std::tuple<const Stub*, const Stub*, const Stub*>& read() { return candtriplets_.read(); }

    const std::tuple<const Stub*, const Stub*, const Stub*>& peek() const { return candtriplets_.peek(); }

    bool idle() const { return idle_; }

    void setNearFull() { nearfull_ = candtriplets_.nearfull(); }

    void reset();

    void step();

    const Stub* innerStub() const { return trpdata_.stub_; }

  private:
    std::vector<VMStubsTEMemory*> innervmstubs_;
    std::vector<VMStubsTEMemory*> outervmstubs_;
    TrpEData trpdata_;
    const Settings* settings_;
    unsigned int layerdisk1_;
    unsigned int layerdisk2_;
    unsigned int layerdisk3_;
    unsigned int iSeed_;
    bool nearfull_;  //initialized at start of each processing step

    //unsigned int memory slot
    unsigned int nmem_out_;
    unsigned int nmem_in_;
    unsigned int istub_out_;
    unsigned int istub_in_;
    unsigned int next_out_;
    unsigned int next_in_;
    unsigned int nstub_out_;
    unsigned int nstub_in_;
    unsigned int outmem_;
    unsigned int inmem_;
    unsigned int nproj_out_;
    unsigned int nproj_in_;

    bool idle_;

    std::tuple<const Stub*, const Stub*, const Stub*> candtriplet_, candtriplet__;
    bool goodtriplet_, goodtriplet__;

    //save the candidate matches
    CircularBuffer<std::tuple<const Stub*, const Stub*, const Stub*> > candtriplets_;
  };  // TripletEngineUnit

};  // namespace trklet
#endif
