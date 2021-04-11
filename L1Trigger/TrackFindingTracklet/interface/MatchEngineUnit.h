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
    MatchEngineUnit(bool barrel, unsigned int layerdisk, std::vector<bool> table);

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
              Tracklet* proj);

    bool empty() const { return candmatches_.empty(); }

    std::pair<Tracklet*, const Stub*> read() { return candmatches_.read(); }

    std::pair<Tracklet*, const Stub*> peek() const { return candmatches_.peek(); }

    Tracklet* currentProj() const { return proj_; }

    bool idle() const { return idle_; }

    void reset();

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

    //used for bend consistency with rinv
    std::vector<bool> table_;

    //save the candidate matches
    CircularBuffer<std::pair<Tracklet*, const Stub*>> candmatches_;
  };

};  // namespace trklet
#endif
