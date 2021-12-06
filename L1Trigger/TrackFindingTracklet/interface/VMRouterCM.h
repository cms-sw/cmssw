// VMRouter: sorts input stubs into smaller units in phi (and possibly z), referred to as "Virtual Modules" (VMs) - implementation for combined modules
#ifndef L1Trigger_TrackFindingTracklet_interface_VMRouterCM_h
#define L1Trigger_TrackFindingTracklet_interface_VMRouterCM_h

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
  class InputLinkMemory;
  class AllStubsMemory;
  class AllInnerStubsMemory;
  class VMStubsMEMemory;
  class VMStubsTEMemory;

  struct VMStubsTEPHICM {
    VMStubsTEPHICM(unsigned int seednumber_, std::vector<VMStubsTEMemory*> vmstubmem_)
        : seednumber(seednumber_), vmstubmem(vmstubmem_){};

    unsigned int seednumber;                  //seed number [0,11]
    std::vector<VMStubsTEMemory*> vmstubmem;  // m_vmstubmem[n] is the VMStubsTEMemory for the nth copy
  };

  class VMRouterCM : public ProcessBase {
  public:
    VMRouterCM(std::string name, Settings const& settings, Globals* global);

    ~VMRouterCM() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

  private:
    //0-5 are the layers and 6-10 are the disks
    unsigned int layerdisk_;

    //overlapbits_ is the top bits of phicorr used to add or subtract one to see if stub should be added to
    //two VMs. nextrabits_ is the number of bits beyond the bits for the phivm that is used by overlapbits_
    unsigned int overlapbits_;
    unsigned int nextrabits_;

    int nbitszfinebintable_;
    int nbitsrfinebintable_;

    unsigned int nvmmebins_;  //number of long z/r bins in VM

    TrackletLUT meTable_;    //used for ME and outer TE barrel
    TrackletLUT diskTable_;  //outer disk used by D1, D2, and D4

    //The input stub memories
    std::vector<InputLinkMemory*> stubinputs_;

    //The all stub memories
    std::vector<AllStubsMemory*> allstubs_;
    std::vector<std::pair<char, AllInnerStubsMemory*> > allinnerstubs_;

    //The VM stubs memories used by the MEs
    std::vector<VMStubsMEMemory*> vmstubsMEPHI_;

    //The VM stubs memories used by the TEs (using structure defined above)
    std::vector<VMStubsTEPHICM> vmstubsTEPHI_;
  };
};  // namespace trklet
#endif
