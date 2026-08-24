// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

#ifndef PhysicsTools_TruthInfo_interface_RecoHitAdapters_h
#define PhysicsTools_TruthInfo_interface_RecoHitAdapters_h

// Adapters that expose a reco object's hits as a range of truth::RecoHit so the
// generic BranchHitAssociator / BranchRecoValidator can match any reco object to
// the truth Branch graph (the customization point envisaged by the HasTruthHits
// concept in BranchHitAssociator.h). These live here, not as member methods on the
// reco data formats, for two reasons: (a) only reco::Track owns its hits - a
// Trackster/TICLCandidate/PFCandidate references layer clusters / blocks that live
// in separate event collections, which a data-format method cannot reach; and (b)
// returning a PhysicsTools type from a DataFormats class would invert the package
// dependency. Each adapter therefore takes the object plus whatever external
// collection it needs.
//
// Tracker hits carry no per-cell energy to share, so they are exposed with unit
// energy and fraction (matching is by shared-hit multiplicity). Calorimeter hits
// are exposed with unit energy and the cell fraction, matching the convention the
// calo association producer / validator already use for CaloParticle/SimCluster, so
// the shared-energy metric compares cell fractions.

#include <algorithm>
#include <vector>

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "PhysicsTools/TruthInfo/interface/BranchHitAssociator.h"

namespace truth {

  // reco::Track -> its valid rechit DetIds (unit weight; tracker shared-hit metric).
  inline std::vector<RecoHit> recoHits(reco::Track const& track) {
    std::vector<RecoHit> hits;
    hits.reserve(track.recHitsSize());
    for (auto it = track.recHitsBegin(); it != track.recHitsEnd(); ++it) {
      TrackingRecHit const* hit = &(**it);
      if (hit->isValid())
        hits.push_back(RecoHit{hit->geographicalId().rawId(), 1.f, 1.f});
    }
    // One entry per DetId, ascending, which is what the merge-join in
    // BranchHitAssociator requires. Two valid rechits can carry one geographicalId, and
    // a repeated cell would be counted twice: that drives the shared-cell count above
    // the branch's own cell count, which makes the reverse score negative and the shared
    // fraction larger than one. The count is what this metric measures, so a duplicate
    // is dropped rather than summed.
    std::sort(hits.begin(), hits.end(), [](RecoHit const& a, RecoHit const& b) { return a.detId < b.detId; });
    hits.erase(
        std::unique(hits.begin(), hits.end(), [](RecoHit const& a, RecoHit const& b) { return a.detId == b.detId; }),
        hits.end());
    return hits;
  }

  // Sort by detId and coalesce duplicates in place (fractions summed), so the
  // merge-join in BranchHitAssociator sees each cell once, without a second vector.
  inline void sortAndCoalesce(std::vector<RecoHit>& hits) {
    std::sort(hits.begin(), hits.end(), [](RecoHit const& a, RecoHit const& b) { return a.detId < b.detId; });
    std::size_t w = 0;
    for (std::size_t r = 0; r < hits.size(); ++r) {
      if (w > 0 && hits[w - 1].detId == hits[r].detId)
        hits[w - 1].fraction += hits[r].fraction;
      else
        hits[w++] = hits[r];
    }
    hits.resize(w);
  }

  // reco::CaloCluster (a single layer cluster) -> its (DetId, fraction) hits (unit
  // energy; the calo shared-energy metric compares cell fractions). Sorted by detId,
  // and coalesced (fractions summed) so a repeated cell is seen once by the merge-join
  // in BranchHitAssociator. HGCAL layer clusters list each cell once, but the overload
  // is generic over reco::CaloCluster, so the dedup keeps it correct for any input.
  inline std::vector<RecoHit> recoHits(reco::CaloCluster const& layerCluster) {
    std::vector<RecoHit> hits;
    hits.reserve(layerCluster.hitsAndFractions().size());
    for (auto const& [detId, fraction] : layerCluster.hitsAndFractions())
      hits.push_back(RecoHit{detId.rawId(), 1.f, fraction});
    sortAndCoalesce(hits);
    return hits;
  }

  // ticl::Trackster -> the (DetId, fraction) of its layer clusters (unit energy; the
  // calo shared-energy metric then compares cell fractions). A layer cluster shared
  // by several tracksters contributes 1/multiplicity of its fraction, as in the TICL
  // trackster associations. Duplicate cells across the trackster's layer clusters
  // are coalesced (fractions summed) so the merge-join in BranchHitAssociator sees
  // each cell once.
  inline std::vector<RecoHit> recoHits(ticl::Trackster const& trackster,
                                       std::vector<reco::CaloCluster> const& layerClusters) {
    std::vector<RecoHit> hits;
    auto const& vertices = trackster.vertices();
    auto const& multiplicities = trackster.vertex_multiplicity();
    std::size_t nHits = 0;
    for (const unsigned int lc : vertices) {
      if (lc < layerClusters.size()) {
        nHits += layerClusters[lc].hitsAndFractions().size();
      }
    }
    hits.reserve(nHits);
    for (std::size_t v = 0; v < vertices.size(); ++v) {
      const unsigned int lc = vertices[v];
      if (lc >= layerClusters.size())
        continue;
      const float multiplicity = v < multiplicities.size() && multiplicities[v] > 0.f ? multiplicities[v] : 1.f;
      for (auto const& [detId, fraction] : layerClusters[lc].hitsAndFractions())
        hits.push_back(RecoHit{detId.rawId(), 1.f, fraction / multiplicity});
    }
    sortAndCoalesce(hits);
    return hits;
  }

}  // namespace truth

#endif
