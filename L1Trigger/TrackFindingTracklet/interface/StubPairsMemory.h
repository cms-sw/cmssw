#ifndef L1Trigger_TrackFindingTracklet_interface_StubPairsMemory_h
#define L1Trigger_TrackFindingTracklet_interface_StubPairsMemory_h

#include "L1Trigger/TrackFindingTracklet/interface/MemoryBase.h"
#include "L1Trigger/TrackFindingTracklet/interface/VMStubTE.h"

#include <vector>

namespace trklet {

  class Settings;

  class StubPairsMemory : public MemoryBase {
  public:
    StubPairsMemory(std::string name, Settings const& settings);

    ~StubPairsMemory() override = default;

    void addStubPair(const VMStubTE& stub1,
                     const VMStubTE& stub2,
                     const unsigned index = 0,
                     const std::string& tedName = "") {
      stubs_.emplace_back(stub1, stub2);
      indices_.push_back(index);
      tedNames_.push_back(tedName);
    }

    unsigned int nStubPairs() const { return stubs_.size(); }

    const VMStubTE& getVMStub1(unsigned int i) const { return stubs_[i].first; }
    const VMStubTE& getVMStub2(unsigned int i) const { return stubs_[i].second; }

    unsigned getIndex(const unsigned i) const { return indices_.at(i); }
    const std::string& getTEDName(const unsigned i) const { return tedNames_.at(i); }

    void clean() override {
      stubs_.clear();
      indices_.clear();
      tedNames_.clear();
    }

    void writeSP(bool first, unsigned int iSector);

  private:
    std::vector<std::pair<const VMStubTE, const VMStubTE> > stubs_;

    std::vector<unsigned> indices_;
    std::vector<std::string> tedNames_;
  };

};  // namespace trklet
#endif
