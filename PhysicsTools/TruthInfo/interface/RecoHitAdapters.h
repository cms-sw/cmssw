// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
// Part of the MC-truth-graph prototype - under heavy development, not yet open
// to external contributions (see PhysicsTools/TruthInfo/README.md).

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
    return hits;
  }

  // ticl::Trackster -> the (DetId, fraction) of its layer clusters (unit energy; the
  // calo shared-energy metric then compares cell fractions). Duplicate cells across
  // the trackster's layer clusters are coalesced (fractions summed) so the
  // merge-join in BranchHitAssociator sees each cell once.
  inline std::vector<RecoHit> recoHits(ticl::Trackster const& trackster,
                                       std::vector<reco::CaloCluster> const& layerClusters) {
    std::vector<RecoHit> hits;
    for (unsigned int lc : trackster.vertices()) {
      if (lc >= layerClusters.size())
        continue;
      for (auto const& [detId, fraction] : layerClusters[lc].hitsAndFractions())
        hits.push_back(RecoHit{detId.rawId(), 1.f, fraction});
    }
    std::sort(hits.begin(), hits.end(), [](RecoHit const& a, RecoHit const& b) { return a.detId < b.detId; });
    std::vector<RecoHit> coalesced;
    coalesced.reserve(hits.size());
    for (auto const& h : hits) {
      if (!coalesced.empty() && coalesced.back().detId == h.detId)
        coalesced.back().fraction += h.fraction;
      else
        coalesced.push_back(h);
    }
    return coalesced;
  }

}  // namespace truth

#endif
