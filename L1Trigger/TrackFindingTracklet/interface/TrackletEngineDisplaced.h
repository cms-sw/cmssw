// TrackletEngineDisplaced: this class forms tracklets (pairs of stubs) w/o beamspot constraint for the displaced (extended) tracking.
// Triplet seeds are formed in the TripletEngine from these (=TrackletEngineDisplaced) + a third stub.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletEngineDisplaced_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletEngineDisplaced_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"

#include <string>
#include <vector>
#include <set>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class VMStubsTEMemory;
  class StubPairsMemory;

  class TrackletEngineDisplaced : public ProcessBase {
  public:
    TrackletEngineDisplaced(std::string name, Settings const& settings, Globals* global);

    ~TrackletEngineDisplaced() override;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

    void readTables();

    short memNameToIndex(const std::string& name);

  private:
    int layer1_;
    int layer2_;
    int disk1_;
    int disk2_;

    std::vector<VMStubsTEMemory*> firstvmstubs_;
    VMStubsTEMemory* secondvmstubs_;

    std::vector<StubPairsMemory*> stubpairs_;

    std::vector<std::set<short> > table_;

    int firstphibits_;
    int secondphibits_;

    int iSeed_;
  };
};  // namespace trklet
#endif
