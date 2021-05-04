// TrackletProcessor: this class is an evolved version, performing the tasks of the TrackletEngine+TrackletCalculator.
// It will combine TEs that feed into a TC to a single module.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletProcessor_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMRouterTable.h"
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
  class VMStubsTEMemory;

  class TrackletProcessor : public TrackletCalculatorBase {
  public:
    TrackletProcessor(std::string name, Settings const& settings, Globals* globals, unsigned int iSector);

    ~TrackletProcessor() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);

    void addOutput(MemoryBase* memory, std::string output) override;

    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    void writeTETable();

    void buildLUT();

  private:
    int iTC_;
    int iAllStub_;

    VMStubsTEMemory* outervmstubs_;

    // The use of a std::tuple here is awkward and should be fixed. This code is slotted for a significant
    // overhaul to allign with the HLS implementation. At that point the use fo the tuple should be
    // eliminated
    //                                               istub          imem        start imem    end imem
    std::vector<std::tuple<CircularBuffer<TEData>, unsigned int, unsigned int, unsigned int, unsigned int> >
        tedatabuffers_;

    std::vector<TrackletEngineUnit> teunits_;

    std::vector<AllStubsMemory*> innerallstubs_;
    std::vector<AllStubsMemory*> outerallstubs_;

    std::map<unsigned int, std::vector<bool> > pttableinner_;
    std::map<unsigned int, std::vector<bool> > pttableouter_;

    std::vector<bool> pttableinnernew_;
    std::vector<bool> pttableouternew_;

    std::vector<std::vector<bool> > useregion_;

    int nbitsfinephi_;
    int nbitsfinephidiff_;

    int innerphibits_;
    int outerphibits_;

    unsigned int nbitszfinebintable_;
    unsigned int nbitsrfinebintable_;

    unsigned int nbitsrzbin_;

    VMRouterTable vmrtable_;
  };

};  // namespace trklet
#endif
