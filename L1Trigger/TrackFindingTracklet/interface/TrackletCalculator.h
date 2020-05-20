#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletCalculator_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletCalculator_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class TrackletProjectionsMemory;
  class MemoryBase;
  class AllStubsMemory;
  class StubPairsMemory;

  class TrackletCalculator : public TrackletCalculatorBase {
  public:
    TrackletCalculator(std::string name, const Settings* const settings, Globals* globals, unsigned int iSector);

    ~TrackletCalculator() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);
    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

  private:
    int iTC_;

    std::vector<AllStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;
    std::vector<StubPairsMemory*> stubpairs_;
  };
};  // namespace trklet
#endif
