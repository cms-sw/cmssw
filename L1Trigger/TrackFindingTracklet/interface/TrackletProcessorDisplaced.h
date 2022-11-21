// TrackletProcessorDisplaced: This class performs the tasks of the TrackletEngineDisplaced+TripletEngine+TrackletCalculatorDisplaced.
#ifndef L1Trigger_TrackFindingTracklet_interface_TrackletProcessorDisplaced_h
#define L1Trigger_TrackFindingTracklet_interface_TrackletProcessorDisplaced_h

#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletCalculatorDisplaced.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"
#include "L1Trigger/TrackFindingTracklet/interface/CircularBuffer.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletEngineUnit.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletParametersMemory.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletProjectionsMemory.h"

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
  class StubPairsMemory;

  class TrackletProcessorDisplaced : public TrackletCalculatorDisplaced {
  public:
    TrackletProcessorDisplaced(std::string name, Settings const& settings, Globals* globals);

    ~TrackletProcessorDisplaced() override = default;

    void addOutputProjection(TrackletProjectionsMemory*& outputProj, MemoryBase* memory);

    void addOutput(MemoryBase* memory, std::string output) override;

    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector, double phimin, double phimax);

  private:
    int iTC_;
    int iAllStub_;
    unsigned int maxStep_;
    int count_;
    unsigned int layerdisk_;

    int layer1_;
    int layer2_;
    int layer3_;
    int disk1_;
    int disk2_;
    int disk3_;

    int firstphibits_;
    int secondphibits_;
    int thirdphibits_;

    int nbitszfinebintable_;
    int nbitsrfinebintable_;

    TrackletLUT innerTable_;         //projection to next layer/disk
    TrackletLUT innerOverlapTable_;  //projection to disk from layer
    TrackletLUT innerThirdTable_;    //projection to disk1 for extended - iseed=10

    std::vector<StubPairsMemory*> stubpairs_;
    /* std::vector<StubTripletsMemory*> stubtriplets_; */
    std::vector<VMStubsTEMemory*> innervmstubs_;
    std::vector<VMStubsTEMemory*> outervmstubs_;

    StubTripletsMemory* stubtriplets_;

    std::map<std::string, std::vector<std::vector<std::string> > > tmpSPTable_;
    std::map<std::string, std::vector<std::map<std::string, unsigned> > > spTable_;
    std::vector<bool> table_;
  };

};  // namespace trklet
#endif
