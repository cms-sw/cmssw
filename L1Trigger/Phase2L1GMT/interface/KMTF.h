#ifndef L1Trigger_Phase2L1GMT_KMTF_h
#define L1Trigger_Phase2L1GMT_KMTF_h
#include "DataFormats/L1TMuonPhase2/interface/MuonStub.h"
#include "L1Trigger/Phase2L1GMT/interface/KMTFCore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <cstdlib>

namespace Phase2L1GMT {

  class KMTF {
  public:
    KMTF(int verbose, const edm::ParameterSet& iConfig);
    ~KMTF();
    std::pair<std::vector<l1t::KMTFTrack>, std::vector<l1t::KMTFTrack> > process(const l1t::MuonStubRefVector& stubsAll,
                                                                                 int bx,
                                                                                 unsigned int MAXN);

  private:
    int verbose_;
    KMTFCore* trackMaker_;
    void overlapCleanTrack(l1t::KMTFTrack& source, const l1t::KMTFTrack& other, bool eq, bool vertex);
    std::vector<l1t::KMTFTrack> cleanRegion(const std::vector<l1t::KMTFTrack>& tracks2,
                                            const std::vector<l1t::KMTFTrack>& tracks3,
                                            const std::vector<l1t::KMTFTrack>& tracks4,
                                            bool vertex);
    void sort(std::vector<l1t::KMTFTrack>& in, bool vertex);
    void swap(std::vector<l1t::KMTFTrack>& list, int i, int j, bool vertex);

    class SeedSorter {
    public:
      SeedSorter() {}
      bool operator()(const l1t::MuonStubRef& a, const l1t::MuonStubRef& b) { return (a->id() < b->id()); }
    };
  };
}  // namespace Phase2L1GMT
#endif
