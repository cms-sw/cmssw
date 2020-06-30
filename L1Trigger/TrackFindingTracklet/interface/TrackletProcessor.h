// TrackletProcessor: this class is an evolved version, performing the tasks of the TrackletEngine+TrackletCalculator.
// It will combine TEs that feed into a TC to a single module.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubsTEMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/StubPairsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/AllStubsMemory.h"

#include <vector>
#include <map>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;

  class TrackletProcessor : public TrackletCalculatorBase {
  public:
    TrackletProcessor(std::string name, Settings const& settings, Globals* globals, unsigned int iSector);

    ~TrackletProcessor() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);

    void addOutput(MemoryBase* memory, std::string output) override;

    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    void setVMPhiBin();

    void writeTETable();

  private:
    int iTC_;

    std::vector<VMStubsTEMemory*> innervmstubs_;
    std::vector<VMStubsTEMemory*> outervmstubs_;

    std::vector<AllStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;

    bool extra_;

    std::map<unsigned int, std::vector<bool> > pttableinner_;
    std::map<unsigned int, std::vector<bool> > pttableouter_;

    int innerphibits_;
    int outerphibits_;
  };

};  // namespace trklet
#endif
