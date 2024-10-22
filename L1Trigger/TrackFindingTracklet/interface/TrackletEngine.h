#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletEngine_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletEngine_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

#include <vector>
#include <string>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class VMStubsTEMemory;
  class StubPairsMemory;

  class TrackletEngine : public ProcessBase {
  public:
    TrackletEngine(std::string name, Settings const& settings, Globals* global);

    ~TrackletEngine() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    void setVMPhiBin();

  private:
    //Which seed type and which layer/disk is used
    unsigned int iSeed_;
    unsigned int layerdisk1_;  //inner seeding layer
    unsigned int layerdisk2_;  //outer seeding layer

    //The input vmstubs memories
    VMStubsTEMemory* innervmstubs_;
    VMStubsTEMemory* outervmstubs_;

    //The output stub pair memory
    StubPairsMemory* stubpairs_;

    //The stub pt (bend) lookup table for the inner and outer stub
    TrackletLUT innerptlut_;
    TrackletLUT outerptlut_;

    //Number of phi bits used in the lookup table
    unsigned int innerphibits_;
    unsigned int outerphibits_;
  };

};  // namespace trklet
#endif
