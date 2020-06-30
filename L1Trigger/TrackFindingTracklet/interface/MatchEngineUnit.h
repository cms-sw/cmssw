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

  class MatchEngineUnit {
  public:
    MatchEngineUnit(bool barrel, std::vector<bool> table, std::vector<bool> tablePS, std::vector<bool> table2S);

    ~MatchEngineUnit() = default;

    void init(VMStubsMEMemory* vmstubsmemory,
              unsigned int slot,
              int projrinv,
              int projfinerz,
              int projfinephi,
              bool isPSseed,
              Tracklet* proj);

    bool empty() const { return candmatches_.empty(); }

    std::pair<Tracklet*, const Stub*> read() { return candmatches_.read(); }

    std::pair<Tracklet*, const Stub*> peek() const { return candmatches_.peek(); }

    Tracklet* currentProj() const { return proj_; }

    bool idle() const { return idle_; }

    void reset();

    void step();

  private:
    VMStubsMEMemory* vmstubsmemory_;

    //unsigned int memory slot
    unsigned int slot_;
    unsigned int istub_;

    bool barrel_;
    int projrinv_;
    int projfinerz_;
    int projfinephi_;
    bool isPSseed_;
    Tracklet* proj_;

    bool idle_;

    //used in the layers
    std::vector<bool> table_;

    //used in the disks
    std::vector<bool> tablePS_;
    std::vector<bool> table2S_;

    //save the candidate matches
    CircularBuffer<std::pair<Tracklet*, const Stub*> > candmatches_;
  };

};  // namespace trklet
#endif
