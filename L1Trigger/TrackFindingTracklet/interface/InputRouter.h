// InputRouter: sorts input stubs into layer/disk and phi region
#ifndef L1Trigger_TrackFindingTracklet_interface_InputRouter_h
#define L1Trigger_TrackFindingTracklet_interface_InputRouter_h

#include "L1Trigger/TrackFindingTracklet/interface/ProcessBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include <string>
#include <vector>
#include <utility>

namespace trklet {

  class Settings;
  class Globals;
  class MemoryBase;
  class DTCLinkMemory;
  class InputLinkMemory;

  class InputRouter : public ProcessBase {
  public:
    InputRouter(std::string name, Settings const& settings, Globals* global);

    ~InputRouter() override = default;

    void addOutput(MemoryBase* memory, std::string output) override;
    void addInput(MemoryBase* memory, std::string input) override;

    void execute();

  private:
    //The input stub memories
    DTCLinkMemory* dtcstubs_;

    //The all stub memories - the ints are layerdisk and phiregion
    std::vector<std::pair<std::pair<unsigned int, unsigned int>, InputLinkMemory*> > irstubs_;
  };
};  // namespace trklet
#endif
