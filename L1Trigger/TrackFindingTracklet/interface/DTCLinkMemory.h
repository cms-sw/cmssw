// This class holds a list of stubs for an DTC link.
// This modules 'owns' the pointers to the stubs. All subsequent modules that handles stubs uses a pointer to the original stored here.
#ifndef L1Trigger_TrackFindingTracklet_interface_DTCLinkMemory_h
#define L1Trigger_TrackFindingTracklet_interface_DTCLinkMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <vector>

namespace trklet {

  class Settings;
  class Globals;
  class Stub;
  class L1TStub;

  class DTCLinkMemory : public MemoryBase {
  public:
    DTCLinkMemory(std::string name, Settings const& settings, double, double);

    ~DTCLinkMemory() override = default;

    void addStub(const L1TStub& al1stub, const Stub& stub);

    unsigned int nStubs() const { return stubs_.size(); }

    Stub* getStub(unsigned int i) { return stubs_[i]; }

    void writeStubs(bool first, unsigned int iSector);

    void clean() override;

  private:
    std::vector<Stub*> stubs_;
  };

};  // namespace trklet
#endif
