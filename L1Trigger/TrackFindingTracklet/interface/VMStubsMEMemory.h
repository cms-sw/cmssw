#ifndef L1Trigger_TrackFindingTracklet_interface_VMStubsMEMemory_h
#define L1Trigger_TrackFindingTracklet_interface_VMStubsMEMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubME.h"

#include <string>
#include <vector>

namespace trklet {

  class Settings;
  class Stub;
  class L1TStub;

  class VMStubsMEMemory : public MemoryBase {
  public:
    VMStubsMEMemory(std::string name, Settings const& settings);

    ~VMStubsMEMemory() override = default;

    void addStub(VMStubME stub, unsigned int bin) {
      if (binnedstubs_[bin].size() < settings_.maxStubsPerBin()) {
        binnedstubs_[bin].push_back(stub);
      }
    }

    void resize(int nbins) { binnedstubs_.resize(nbins); }

    unsigned int nStubsBin(unsigned int bin) const {
      assert(bin < binnedstubs_.size());
      return binnedstubs_[bin].size();
    }

    const VMStubME& getVMStubMEBin(unsigned int bin, unsigned int i) const {
      assert(bin < binnedstubs_.size());
      assert(i < binnedstubs_[bin].size());
      return binnedstubs_[bin][i];
    }

    const Stub* getStubBin(unsigned int bin, unsigned int i) const {
      assert(bin < binnedstubs_.size());
      assert(i < binnedstubs_[bin].size());
      return binnedstubs_[bin][i].stub();
    }

    void clean() override {
      for (auto& binnedstub : binnedstubs_) {
        binnedstub.clear();
      }
    }

    void writeStubs(bool first, unsigned int iSector);

  private:
    std::vector<std::vector<VMStubME> > binnedstubs_;
  };

};  // namespace trklet
#endif
