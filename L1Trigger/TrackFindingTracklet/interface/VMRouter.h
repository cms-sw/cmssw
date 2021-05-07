// VMRouter: sorts input stubs into smaller units in phi (and possibly z), referred to as "Virtual Modules" (VMs)
#ifndef L1Trigger_TrackFindingTracklet_interface_VMRouter_h
#define L1Trigger_TrackFindingTracklet_interface_VMRouter_h

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
  class VMStubsMEMemory;
  class VMStubsTEMemory;

  struct VMStubsTEPHI {
    VMStubsTEPHI(unsigned int seednumber_,
                 unsigned int stubposition_,
                 std::vector<std::vector<VMStubsTEMemory*> > vmstubmem_)
        : seednumber(seednumber_), stubposition(stubposition_), vmstubmem(vmstubmem_){};

    unsigned int seednumber;    //seed number [0,11]
    unsigned int stubposition;  //stub position in the seed
    std::vector<std::vector<VMStubsTEMemory*> >
        vmstubmem;  // m_vmstubmem[iVM][n] is the VMStubsTEMemory for iVM and the nth copy
  };

  class VMRouter : public ProcessBase {
  public:
    VMRouter(std::string name, Settings const& settings, Globals* global);

    ~VMRouter() override = default;

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

    TrackletLUT meTable_;            //used for ME and outer TE barrel
    TrackletLUT diskTable_;          //outer disk used by D1, D2, and D4
    TrackletLUT innerTable_;         //projection to next layer/disk
    TrackletLUT innerOverlapTable_;  //projection to disk from layer
    TrackletLUT innerThirdTable_;    //projection to disk1 for extended - iseed=10

    //The input stub memories the two tmp inputs are used to build the order needed in HLS
    std::vector<InputLinkMemory*> stubinputs_, stubinputtmp_, stubinputdisk2stmp_;

    //The all stub memories
    std::vector<AllStubsMemory*> allstubs_;

    //The VM stubs memories used by the MEs
    std::vector<VMStubsMEMemory*> vmstubsMEPHI_;

    //The VM stubs memories used by the TEs (using structure defined above)
    std::vector<VMStubsTEPHI> vmstubsTEPHI_;
  };
};  // namespace trklet
#endif
