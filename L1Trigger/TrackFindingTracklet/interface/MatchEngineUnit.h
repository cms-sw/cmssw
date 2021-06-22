#ifndef L1Trigger_TrackFindingTracklet_interface_MatchEngineUnit_h
#define L1Trigger_TrackFindingTracklet_interface_MatchEngineUnit_h

#include "L1Trigger/TrackFindingTracklet/interface/VMStubsMEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/Tracklet.h"
#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <cassert>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;
  class TrackletLUT;

  class MatchEngineUnit {
  public:
    MatchEngineUnit(bool barrel, unsigned int layerdisk, const TrackletLUT& luttable);

    ~MatchEngineUnit() = default;

    void init(VMStubsMEMemory* vmstubsmemory,
              unsigned int nrzbin,
              unsigned int rzbin,
              unsigned int iphi,
              int shift,
              int projrinv,
              int projfinerz,
              int projfinephi,
              bool usefirstMinus,
              bool usefirstPlus,
              bool usesecondMinus,
              bool usesecondPlus,
              bool isPSseed,
              Tracklet* proj,
              bool print);

    bool empty() const { return candmatches_.empty(); }

    int TCID() const;

    std::pair<Tracklet*, const Stub*> read() { return candmatches_.read(); }

    std::pair<Tracklet*, const Stub*> peek() const { return candmatches_.peek(); }

    Tracklet* currentProj() const { return proj_; }

    bool idle() const { return idle_; }

    bool active() const { return !idle_ || goodpair_ || goodpair__ || !empty(); }

    //needed for consistency with HLS FW version ("_" vs "__" indicating different pipelining stages)
    bool have_() const { return havepair_; }
    bool have__() const { return havepair__; }

    void reset();

    unsigned int rptr() const { return candmatches_.rptr(); }
    unsigned int wptr() const { return candmatches_.wptr(); }

    void step(bool print);

  private:
    VMStubsMEMemory* vmstubsmemory_;

    unsigned int nrzbins_;
    unsigned int rzbin_;
    unsigned int phibin_;
    int shift_;

    unsigned int istub_;
    unsigned int iuse_;

    bool barrel_;
    int projrinv_;
    int projfinerz_;
    int projfinephi_;
    std::vector<std::pair<unsigned int, unsigned int>> use_;
    bool isPSseed_;
    Tracklet* proj_;

    bool idle_;

    unsigned int layerdisk_;

    //LUT for bend consistency with rinv
    const TrackletLUT& luttable_;

    //Pipeline variables
    std::pair<Tracklet*, const Stub*> tmppair_, tmppair__;
    bool goodpair_, goodpair__;
    bool havepair_, havepair__;

    //save the candidate matches
    CircularBuffer<std::pair<Tracklet*, const Stub*>> candmatches_;
  };

};  // namespace trklet
#endif
