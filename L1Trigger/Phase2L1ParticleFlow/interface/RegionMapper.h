#ifndef L1Trigger_Phase2L1ParticleFlow_RegionMapper_h
#define L1Trigger_Phase2L1ParticleFlow_RegionMapper_h

#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TCorrelator/interface/TkMuon.h"
#include "DataFormats/L1TCorrelator/interface/TkMuonFwd.h"

#include <unordered_map>

namespace edm {
  class Event;
}

namespace l1tpf_impl {
  class RegionMapper {
    // This does the input and filling of regions.
  public:
    RegionMapper(const edm::ParameterSet &);

    // add object, without tracking references
    void addTrack(const l1t::PFTrack &t);
    void addMuon(const l1t::Muon &t);
    void addMuon(const l1t::TkMuon &t);
    void addCalo(const l1t::PFCluster &t);
    void addEmCalo(const l1t::PFCluster &t);

    // add object, tracking references
    void addTrack(const l1t::PFTrack &t, l1t::PFTrackRef ref);
    void addCalo(const l1t::PFCluster &t, l1t::PFClusterRef ref);
    void addEmCalo(const l1t::PFCluster &t, l1t::PFClusterRef ref);

    void clear();
    std::vector<Region> &regions() { return regions_; }

    std::unique_ptr<l1t::PFCandidateCollection> fetch(bool puppi = true, float ptMin = 0.01) const;
    std::unique_ptr<l1t::PFCandidateCollection> fetchCalo(float ptMin = 0.01, bool emcalo = false) const;
    std::unique_ptr<l1t::PFCandidateCollection> fetchTracks(float ptMin = 0.01, bool fromPV = false) const;

    void putEgObjects(edm::Event &iEvent,
                      const bool writeEgSta,
                      const std::string &egLablel,
                      const std::string &tkEmLabel,
                      const std::string &tkEleLabel,
                      const float ptMin = 0.01) const;

    std::pair<unsigned, unsigned> totAndMaxInput(/*Region::InputType*/ int type) const;
    std::pair<unsigned, unsigned> totAndMaxOutput(/*Region::OutputType*/ int type, bool puppi) const;
    std::unique_ptr<std::vector<unsigned>> vecInput(int type) const;
    std::unique_ptr<std::vector<unsigned>> vecOutput(int type, bool puppi) const;

  protected:
    std::vector<Region> regions_;
    bool useRelativeRegionalCoordinates_;  // whether the eta,phi in each region are global or relative to the region center
    enum class TrackAssoMode { atVertex, atCalo, any = 999 } trackRegionMode_;

    // these are used to link items back
    std::unordered_map<const l1t::PFCluster *, l1t::PFClusterRef> clusterRefMap_;
    std::unordered_map<const l1t::PFTrack *, l1t::PFTrackRef> trackRefMap_;
  };

}  // namespace l1tpf_impl
#endif
