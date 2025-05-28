#ifndef RecoTracker_MkFit_plugins_convertHits_h
#define RecoTracker_MkFit_plugins_convertHits_h

#include "DataFormats/Provenance/interface/ProductID.h"

#include "FWCore/Utilities/interface/Likely.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"

namespace mkfit {
  template <typename Traits>
  struct MkFitHitFiller {
    MkFitHitFiller(const Traits& traits,
                   mkfit::HitVec& mkFitHits,
                   std::vector<TrackingRecHit const*>& clusterIndexToHit,
                   std::vector<int>& layerIndexToHit,
                   std::vector<unsigned int>& layerSize,
                   std::vector<float>& clusterChargeVec,
                   const TrackerTopology& ttopo,
                   const TransientTrackingRecHitBuilder& ttrhBuilder,
                   const MkFitGeometry& mkFitGeom)
        : traits{traits},
          mkFitHits{mkFitHits},
          clusterIndexToHit{clusterIndexToHit},
          layerIndexToHit{layerIndexToHit},
          layerSize{layerSize},
          clusterChargeVec{clusterChargeVec},
          ttopo{ttopo},
          ttrhBuilder{ttrhBuilder},
          mkFitGeom{mkFitGeom} {}

    template <typename Clusters, typename Hit>
    void fill(Clusters const& clusters, unsigned int clusterIndex, Hit const& hit, DetId detid, GeomDetUnit const& du) {
      const auto ilay = mkFitGeom.mkFitLayerNumber(detid);
      const auto uniqueIdInLayer = mkFitGeom.uniqueIdInLayer(ilay, detid.rawId());
      const auto chargeScale = traits.chargeScale(detid);
      const auto& surf = du.surface();

      const auto& cluster = traits.cluster(clusters, clusterIndex);

      const auto charge = traits.clusterCharge(cluster, chargeScale);

      if (!traits.passCCC(charge))
        return;

      const auto& gpos = surf.toGlobal(hit.localPosition());
      SVector3 pos(gpos.x(), gpos.y(), gpos.z());
      const auto& gerr = ErrorFrameTransformer::transform(hit.localPositionError(), surf);
      SMatrixSym33 err{{float(gerr.cxx()),
                        float(gerr.cyx()),
                        float(gerr.cyy()),
                        float(gerr.czx()),
                        float(gerr.czy()),
                        float(gerr.czz())}};

      LogTrace("MkFitHitConverter") << "Adding hit detid " << detid.rawId() << " subdet " << detid.subdetId()
                                    << " layer " << ttopo.layer(detid) << " isStereo " << ttopo.isStereo(detid)
                                    << " zplus "
                                    << " index " << clusterIndex << " ilay " << ilay;

      mkFitHits[clusterIndex] = mkfit::Hit(pos, err);
      clusterIndexToHit[clusterIndex] = &hit;
      layerIndexToHit[clusterIndex] = ilay;
      if constexpr (Traits::applyCCC()) {
        clusterChargeVec[clusterIndex] = charge;
      }
      layerSize[ilay]++;

      traits.setDetails(mkFitHits[clusterIndex], cluster, uniqueIdInLayer, charge);
    }

    const Traits& traits;
    mkfit::HitVec& mkFitHits;
    std::vector<TrackingRecHit const*>& clusterIndexToHit;
    std::vector<int>& layerIndexToHit;
    std::vector<unsigned int>& layerSize;
    std::vector<float>& clusterChargeVec;
    const TrackerTopology& ttopo;
    const TransientTrackingRecHitBuilder& ttrhBuilder;
    const MkFitGeometry& mkFitGeom;
  };

  template <typename Traits, typename HitCollection, typename ClusterCollection>
  edm::ProductID convertHits(const Traits& traits,
                             const HitCollection& hits,
                             const ClusterCollection& clusters,
                             mkfit::HitVec& mkFitHits,
                             std::vector<TrackingRecHit const*>& clusterIndexToHit,
                             std::vector<int>& layerIndexToHit,
                             std::vector<unsigned int>& layerSize,
                             std::vector<float>& clusterChargeVec,
                             const TrackerTopology& ttopo,
                             const TransientTrackingRecHitBuilder& ttrhBuilder,
                             const MkFitGeometry& mkFitGeom,
                             std::size_t maxSizeGuess = 0) {
    if (hits.empty())
      return edm::ProductID{};

    const auto& lastClusterRef = hits.data().back().firstClusterRef();
    edm::ProductID clusterID = lastClusterRef.id();
    auto const size = std::max(static_cast<std::size_t>(lastClusterRef.index() + 1), maxSizeGuess);
    if (mkFitHits.size() < size) {
      mkFitHits.resize(size);
      clusterIndexToHit.resize(size, nullptr);
      layerIndexToHit.resize(size, -1);
      if constexpr (Traits::applyCCC()) {
        clusterChargeVec.resize(size, -1.f);
      }
    }

    for (const auto& detset : hits) {
      if (detset.empty())
        continue;
      const DetId detid = detset.detId();
      const auto ilay = mkFitGeom.mkFitLayerNumber(detid);
      const auto uniqueIdInLayer = mkFitGeom.uniqueIdInLayer(ilay, detid.rawId());
      const auto chargeScale = traits.chargeScale(detid);
      const auto& surf = detset.begin()->det()->surface();

      for (const auto& hit : detset) {
        auto clusterRef = hit.firstClusterRef();
        if UNLIKELY (clusterRef.id() != clusterID) {
          throw cms::Exception("LogicError")
              << "Input hit collection has Refs to many cluster collections. Last hit had Ref to product " << clusterID
              << ", but encountered Ref to product " << clusterRef.id() << " on detid " << detid.rawId();
        }
        const auto clusterIndex = clusterRef.index();

        const auto& clu = traits.cluster(clusters, clusterIndex);
        const auto charge = traits.clusterCharge(clu, chargeScale);
        if (!traits.passCCC(charge))
          continue;

        const auto& gpos = surf.toGlobal(hit.localPosition());
        SVector3 pos(gpos.x(), gpos.y(), gpos.z());
        const auto& gerr = ErrorFrameTransformer::transform(hit.localPositionError(), surf);
        SMatrixSym33 err{{float(gerr.cxx()),
                          float(gerr.cyx()),
                          float(gerr.cyy()),
                          float(gerr.czx()),
                          float(gerr.czy()),
                          float(gerr.czz())}};

        LogTrace("MkFitHitConverter") << "Adding hit detid " << detid.rawId() << " subdet " << detid.subdetId()
                                      << " layer " << ttopo.layer(detid) << " isStereo " << ttopo.isStereo(detid)
                                      << " zplus "
                                      << " index " << clusterIndex << " ilay " << ilay;

        if UNLIKELY (clusterIndex >= mkFitHits.size()) {
          mkFitHits.resize(clusterIndex + 1);
          clusterIndexToHit.resize(clusterIndex + 1, nullptr);
          layerIndexToHit.resize(clusterIndex + 1, -1);
          if constexpr (Traits::applyCCC()) {
            clusterChargeVec.resize(clusterIndex + 1, -1.f);
          }
        }
        mkFitHits[clusterIndex] = mkfit::Hit(pos, err);
        clusterIndexToHit[clusterIndex] = &hit;
        layerIndexToHit[clusterIndex] = ilay;
        if constexpr (Traits::applyCCC()) {
          clusterChargeVec[clusterIndex] = charge;
        }
        layerSize[ilay]++;

        traits.setDetails(mkFitHits[clusterIndex], clu, uniqueIdInLayer, charge);
      }
    }
    return clusterID;
  }
}  // namespace mkfit

#endif
