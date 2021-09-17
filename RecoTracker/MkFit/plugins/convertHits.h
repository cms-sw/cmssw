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
#include "Hit.h"
#include "mkFit/HitStructures.h"

namespace mkfit {
  template <typename Traits, typename HitCollection>
  edm::ProductID convertHits(const Traits& traits,
                             const HitCollection& hits,
                             mkfit::HitVec& mkFitHits,
                             std::vector<TrackingRecHit const*>& clusterIndexToHit,
                             std::vector<float>& clusterChargeVec,
                             const TrackerTopology& ttopo,
                             const TransientTrackingRecHitBuilder& ttrhBuilder,
                             const MkFitGeometry& mkFitGeom) {
    if (hits.empty())
      return edm::ProductID{};

    edm::ProductID clusterID;
    {
      const auto& lastClusterRef = hits.data().back().firstClusterRef();
      clusterID = lastClusterRef.id();
      if (lastClusterRef.index() >= mkFitHits.size()) {
        auto const size = lastClusterRef.index();
        mkFitHits.resize(size);
        clusterIndexToHit.resize(size, nullptr);
        if constexpr (Traits::applyCCC()) {
          clusterChargeVec.resize(size, -1.f);
        }
      }
    }

    for (const auto& detset : hits) {
      const DetId detid = detset.detId();
      const auto ilay = mkFitGeom.mkFitLayerNumber(detid);

      for (const auto& hit : detset) {
        const auto charge = traits.clusterCharge(hit, detid);
        if (!traits.passCCC(charge))
          continue;

        const auto& gpos = hit.globalPosition();
        SVector3 pos(gpos.x(), gpos.y(), gpos.z());
        const auto& gerr = hit.globalPositionError();
        SMatrixSym33 err;
        err.At(0, 0) = gerr.cxx();
        err.At(1, 1) = gerr.cyy();
        err.At(2, 2) = gerr.czz();
        err.At(0, 1) = gerr.cyx();
        err.At(0, 2) = gerr.czx();
        err.At(1, 2) = gerr.czy();

        auto clusterRef = hit.firstClusterRef();
        if UNLIKELY (clusterRef.id() != clusterID) {
          throw cms::Exception("LogicError")
              << "Input hit collection has Refs to many cluster collections. Last hit had Ref to product " << clusterID
              << ", but encountered Ref to product " << clusterRef.id() << " on detid " << detid.rawId();
        }
        const auto clusterIndex = clusterRef.index();
        LogTrace("MkFitHitConverter") << "Adding hit detid " << detid.rawId() << " subdet " << detid.subdetId()
                                      << " layer " << ttopo.layer(detid) << " isStereo " << ttopo.isStereo(detid)
                                      << " zplus "
                                      << " index " << clusterIndex << " ilay " << ilay;

        if UNLIKELY (clusterIndex >= mkFitHits.size()) {
          mkFitHits.resize(clusterIndex + 1);
          clusterIndexToHit.resize(clusterIndex + 1, nullptr);
          if constexpr (Traits::applyCCC()) {
            clusterChargeVec.resize(clusterIndex + 1, -1.f);
          }
        }
        mkFitHits[clusterIndex] = mkfit::Hit(pos, err);
        clusterIndexToHit[clusterIndex] = &hit;
        if constexpr (Traits::applyCCC()) {
          clusterChargeVec[clusterIndex] = charge;
        }

        const auto uniqueIdInLayer = mkFitGeom.uniqueIdInLayer(ilay, detid.rawId());
        traits.setDetails(mkFitHits[clusterIndex], *(hit.cluster()), uniqueIdInLayer, charge);
      }
    }
    return clusterID;
  }
}  // namespace mkfit

#endif
