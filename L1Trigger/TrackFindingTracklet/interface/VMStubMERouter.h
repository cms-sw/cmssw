// VMStubMERouter: sorts input stubs into smaller units in phi (and possibly z), referred to as "Virtual Modules" (VMs) - implementation for combined modules
#ifndef L1Trigger_TrackFindingTracklet_interface_VMStubMERouter_h
#define L1Trigger_TrackFindingTracklet_interface_VMStubMERouter_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"
#include "L1Trigger/TrackFindingTracklet/interface/TrackletLUT.h"

#include <string>
#include <vector>
#include <utility>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class AllStubsMemory;
  class VMStubsMEMemory;

  class VMStubMERouter : public ProcessBase {
  public:
    VMStubMERouter(std::string name, Settings const& settings, Globals* global);

    ~VMStubMERouter() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute(unsigned int iSector);

  private:
    //0-5 are the layers and 6-10 are the disks
    unsigned int layerdisk_;

    int nbitszfinebintable_;
    int nbitsrfinebintable_;

    unsigned int nvmmebins_;  //number of long z/r bins in VM

    TrackletLUT meTable_;  //used for ME and outer TE barrel

    //The input stub memories
    std::vector<AllStubsMemory*> stubinputs_;

    //The all stub memories
    std::vector<AllStubsMemory*> allstubs_;

    //The VM stubs memories used by the MEs
    std::vector<VMStubsMEMemory*> vmstubsinput_;

    //The VM stubs memories used by the MEs
    std::vector<VMStubsMEMemory*> vmstubsMEPHI_;
  };
};  // namespace trklet

#endif

