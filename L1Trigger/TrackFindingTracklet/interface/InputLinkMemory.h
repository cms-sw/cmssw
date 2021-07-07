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

  class InputLinkMemory : public MemoryBase {
  public:
    InputLinkMemory(std::string name, Settings const& settings, double, double);

    ~InputLinkMemory() override = default;

    void addStub(Stub* stub);

    unsigned int nStubs() const { return stubs_.size(); }

    Stub* getStub(unsigned int i) { return stubs_[i]; }

    void writeStubs(bool first, unsigned int iSector);

    void clean() override;

  private:
    std::vector<Stub*> stubs_;
  };

};  // namespace trklet
#endif
