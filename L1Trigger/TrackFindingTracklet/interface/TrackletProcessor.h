// TrackletProcessor: this class is an evolved version, performing the tasks of the TrackletEngine+TrackletCalculator.
// It will combine TEs that feed into a TC to a single module.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineUnit.h"

#include <vector>
#include <tuple>
#include <map>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class AllStubsMemory;
  class AllInnerStubsMemory;
  class VMStubsTEMemory;

  class TrackletProcessor : public TrackletCalculatorBase {
  public:
    TrackletProcessor(std::string name, Settings const& settings, Globals* globals);

    ~TrackletProcessor() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);

    void addOutput(MemoryBase* memory, std::string output) override;

    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phimin, double phimax);

  private:
    int iTC_;
    int iAllStub_;

    unsigned int maxStep_;

    VMStubsTEMemory* outervmstubs_;

    //                                 istub          imem          start imem    end imem
    std::tuple<CircularBuffer<TEData>, unsigned int, unsigned int, unsigned int, unsigned int> tebuffer_;

    std::vector<TrackletEngineUnit> teunits_;

    std::vector<AllInnerStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;

    TrackletLUT pttableinner_;
    TrackletLUT pttableouter_;
    TrackletLUT useregiontable_;

    int nbitsfinephi_;
    int nbitsfinephidiff_;

    int innerphibits_;
    int outerphibits_;

    unsigned int nbitszfinebintable_;
    unsigned int nbitsrfinebintable_;

    unsigned int nbitsrzbin_;

    TrackletLUT innerTable_;         //projection to next layer/disk
    TrackletLUT innerOverlapTable_;  //projection to disk from layer
  };

};  // namespace trklet
#endif
