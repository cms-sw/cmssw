// This class holds a list of stubs for an input link.
// This modules 'owns' the pointers to the stubs. All subsequent modules that handles stubs uses a pointer to the original stored here.
#ifndef L1Trigger_TrackFindingTracklet_interface_InputLinkMemory_h
#define L1Trigger_TrackFindingTracklet_interface_InputLinkMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class Stub;
  class L1TStub;
  class VMRouterPhiCorrTable;

  class InputLinkMemory : public MemoryBase {
  public:
    InputLinkMemory(std::string name, Settings const& settings, unsigned int iSector, double, double);

    ~InputLinkMemory() override = default;

    bool addStub(Settings const& settings, Globals* globals, L1TStub& al1stub, Stub& stub, std::string dtc);

    unsigned int nStubs() const { return stubs_.size(); }

    Stub* getStub(unsigned int i) { return stubs_[i]; }

    void writeStubs(bool first);

    void clean() override;

  private:
    std::vector<Stub*> stubs_;
    int phiregion_;
    unsigned int layerdisk_;
  };

};  // namespace trklet
#endif
