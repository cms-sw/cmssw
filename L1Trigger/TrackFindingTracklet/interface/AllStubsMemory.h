#ifndef L1Trigger_TrackFindingTracklet_interface_AllStubsMemory_h
#define L1Trigger_TrackFindingTracklet_interface_AllStubsMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"

#include <utility>
#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;

  class AllStubsMemory : public MemoryBase {
  public:
    AllStubsMemory(std::string name, Settings const& settings);

    ~AllStubsMemory() override = default;

    void addStub(const Stub* stub) { stubs_.push_back(stub); }

    unsigned int nStubs() const { return stubs_.size(); }

    const Stub* getStub(unsigned int i) const { return stubs_[i]; }

    void clean() override { stubs_.clear(); }

    void writeStubs(bool first, unsigned int iSector);

  private:
    std::vector<const Stub*> stubs_;
  };

};  // namespace trklet
#endif
