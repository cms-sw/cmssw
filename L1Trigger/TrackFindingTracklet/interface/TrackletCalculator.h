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
  class VarInv;
  class VarBase;

  class TrackletCalculator : public TrackletCalculatorBase {
  public:
    TrackletCalculator(std::string name, Settings const& settings, Globals* globals);

    ~TrackletCalculator() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);
    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phimin, double phimax);

  private:
    int iTC_;

    std::vector<AllStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;
    std::vector<StubPairsMemory*> stubpairs_;

    void writeInvTable(void (*writeLUT)(const VarInv&, const std::string&));
    void writeFirmwareDesign(void (*writeDesign)(const std::vector<VarBase*>&, const std::string&));
  };
};  // namespace trklet
#endif
