#include "RecoMTD/TrackExtender/interface/MTDHitMatcher.h"

#include <iomanip>
#include <set>

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMTD/TimingTools/interface/TrackTofPidInfo.h"

#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

using mtd::computeTrackTofPidInfo;
using mtd::MTDHitMatchingInfo;
using mtd::SigmaTofCalc;
using mtd::TofCalc;
using mtd::TrackSegments;

namespace {
  bool cmp_for_detset(const unsigned one, const unsigned two) { return one < two; }

  void find_hits_in_dets(const MTDTrackingDetSetVector& hits,
                         const Trajectory& traj,
                         const DetLayer* layer,
                         const TrajectoryStateOnSurface& tsos,
                         const float pmag2,
                         const float pathlength0,
                         const TrackSegments& trs0,
                         const float vtxTime,
                         const float vtxTimeError,
                         bool useVtxConstraint,
                         const reco::BeamSpot& bs,
                         const float bsTimeSpread,
                         const Propagator* prop,
                         const MeasurementEstimator* estimator,
                         std::set<MTDHitMatchingInfo>& out) {
    std::pair<bool, TrajectoryStateOnSurface> comp = layer->compatible(tsos, *prop, *estimator);
    if (comp.first) {
      const std::vector<DetLayer::DetWithState> compDets = layer->compatibleDets(tsos, *prop, *estimator);
      LogTrace("TrackExtenderWithMTD") << "Hit search: Compatible dets " << compDets.size();
      if (!compDets.empty()) {
        for (const auto& detWithState : compDets) {
          auto range = hits.equal_range(detWithState.first->geographicalId(), cmp_for_detset);
          if (range.first == range.second) {
            LogTrace("TrackExtenderWithMTD")
                << "Hit search: no hit in DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          auto pl = prop->propagateWithPath(tsos, detWithState.second.surface());
          if (pl.second == 0.) {
            LogTrace("TrackExtenderWithMTD")
                << "Hit search: no propagation to DetId " << detWithState.first->geographicalId().rawId();
            continue;
          }

          const float t_vtx = useVtxConstraint ? vtxTime : 0.f;
          const float t_vtx_err = useVtxConstraint ? vtxTimeError : bsTimeSpread;

          float lastpmag2 = trs0.segmentPathAndMom2(0).second;

          for (auto detitr = range.first; detitr != range.second; ++detitr) {
            for (const auto& hit : *detitr) {
              auto est = estimator->estimate(detWithState.second, hit);
              if (!est.first) {
                LogTrace("TrackExtenderWithMTD")
                    << "Hit search: no compatible estimate in DetId " << detWithState.first->geographicalId().rawId()
                    << " for hit at pos (" << std::fixed << std::setw(14) << hit.globalPosition().x() << ","
                    << std::fixed << std::setw(14) << hit.globalPosition().y() << "," << std::fixed << std::setw(14)
                    << hit.globalPosition().z() << ")";
                continue;
              }

              LogTrace("TrackExtenderWithMTD")
                  << "Hit search: spatial compatibility DetId " << detWithState.first->geographicalId().rawId()
                  << " TSOS dx/dy " << std::fixed << std::setw(14)
                  << std::sqrt(detWithState.second.localError().positionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(detWithState.second.localError().positionError().yy()) << " hit dx/dy "
                  << std::fixed << std::setw(14) << std::sqrt(hit.localPositionError().xx()) << " " << std::fixed
                  << std::setw(14) << std::sqrt(hit.localPositionError().yy()) << " chi2 " << std::fixed
                  << std::setw(14) << est.second;

              mtd::TrackTofPidInfo tof = computeTrackTofPidInfo(lastpmag2,
                                                                 std::abs(pl.second),
                                                                 trs0,
                                                                 hit.time(),
                                                                 hit.timeError(),
                                                                 t_vtx,
                                                                 t_vtx_err,
                                                                 false,
                                                                 TofCalc::kMixd,
                                                                 SigmaTofCalc::kMixd);
              MTDHitMatchingInfo mi;
              mi.hit = &hit;
              mi.estChi2 = est.second;
              mi.timeChi2 = tof.dtchi2_best;

              out.insert(mi);
            }
          }
        }
      }
    }
  }
}  // namespace

namespace mtd {

  MTDHitMatcher::MTDHitMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector iC)
      : btlChi2Cut_(pset.getParameter<double>("btlChi2Cut")),
        btlTimeChi2Cut_(pset.getParameter<double>("btlTimeChi2Cut")),
        etlChi2Cut_(pset.getParameter<double>("etlChi2Cut")),
        etlTimeChi2Cut_(pset.getParameter<double>("etlTimeChi2Cut")),
        bsTimeSpread_(pset.getParameter<double>("bsTimeSpread")),
        mtdRecHitBuilder_(pset.getParameter<std::string>("MTDRecHitBuilder")),
        theEstimator_(std::make_unique<Chi2MeasurementEstimator>(pset.getParameter<double>("estimatorMaxChi2"),
                                                                 pset.getParameter<double>("estimatorMaxNSigma"))) {
    hitBuilderToken_ = iC.esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
        edm::ESInputTag("", mtdRecHitBuilder_));
  }

  void MTDHitMatcher::setServices(const edm::EventSetup& es) { hitbuilder_ = es.getHandle(hitBuilderToken_); }

  void MTDHitMatcher::fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<double>("estimatorMaxChi2", 500.);
    desc.add<double>("estimatorMaxNSigma", 10.);
    desc.add<double>("btlChi2Cut", 50.);
    desc.add<double>("btlTimeChi2Cut", 10.);
    desc.add<double>("etlChi2Cut", 50.);
    desc.add<double>("etlTimeChi2Cut", 10.);
    desc.add<double>("bsTimeSpread", 0.2);
    desc.add<std::string>("MTDRecHitBuilder", "MTDRecHitBuilder");
  }

  MTDHitMatchResult MTDHitMatcher::matchBTL(const TrajectoryStateOnSurface& tsos,
                                             const Trajectory& traj,
                                             float pmag2,
                                             float pathlength0,
                                             const TrackSegments& trs0,
                                             const MTDTrackingDetSetVector& hits,
                                             const MTDDetLayerGeometry* geo,
                                             const Propagator* prop,
                                             const reco::BeamSpot& bs,
                                             float vtxTime,
                                             float vtxTimeError) const {
    return matchLayers(
        geo->allBTLLayers(), tsos, traj, pmag2, pathlength0, trs0, hits, prop, bs, vtxTime, vtxTimeError);
  }

  MTDHitMatchResult MTDHitMatcher::matchETL(const TrajectoryStateOnSurface& tsos,
                                             const Trajectory& traj,
                                             float pmag2,
                                             float pathlength0,
                                             const TrackSegments& trs0,
                                             const MTDTrackingDetSetVector& hits,
                                             const MTDDetLayerGeometry* geo,
                                             const Propagator* prop,
                                             const reco::BeamSpot& bs,
                                             float vtxTime,
                                             float vtxTimeError) const {
    // only propagate to the disk that's on the same side as the track
    std::vector<const DetLayer*> layers;
    for (const DetLayer* lay : geo->allETLLayers()) {
      const float diskZ = static_cast<const ForwardDetLayer*>(lay)->specificSurface().position().z();
      if (tsos.globalPosition().z() * diskZ > 0)
        layers.push_back(lay);
    }

    auto result = matchLayers(layers, tsos, traj, pmag2, pathlength0, trs0, hits, prop, bs, vtxTime, vtxTimeError);

    // the ETL hits order must be from the innermost to the outermost
    if (result.hits.size() == 2) {
      if (std::abs(result.hits[0]->globalPosition().z()) > std::abs(result.hits[1]->globalPosition().z())) {
        std::reverse(result.hits.begin(), result.hits.end());
      }
    }
    return result;
  }

  MTDHitMatchResult MTDHitMatcher::matchLayers(const std::vector<const DetLayer*>& layers,
                                                const TrajectoryStateOnSurface& tsos,
                                                const Trajectory& traj,
                                                float pmag2,
                                                float pathlength0,
                                                const TrackSegments& trs0,
                                                const MTDTrackingDetSetVector& hits,
                                                const Propagator* prop,
                                                const reco::BeamSpot& bs,
                                                float vtxTime,
                                                float vtxTimeError) const {
    MTDHitMatchResult result;
    for (const DetLayer* ilay : layers) {
      if (ilay->isBarrel()) {
        LogTrace("TrackExtenderWithMTD")
            << "Hit search: BTL layer at R= "
            << static_cast<const BarrelDetLayer*>(ilay)->specificSurface().radius();
      } else {
        LogTrace("TrackExtenderWithMTD")
            << "Hit search: ETL disk at Z = "
            << static_cast<const ForwardDetLayer*>(ilay)->specificSurface().position().z();
      }
      fillMatchingHits(
          ilay, tsos, traj, pmag2, pathlength0, trs0, hits, prop, bs, vtxTime, vtxTimeError, result.hits, result.bestHit);
    }
    return result;
  }

  void MTDHitMatcher::fillMatchingHits(const DetLayer* ilay,
                                        const TrajectoryStateOnSurface& tsos,
                                        const Trajectory& traj,
                                        float pmag2,
                                        float pathlength0,
                                        const TrackSegments& trs0,
                                        const MTDTrackingDetSetVector& hits,
                                        const Propagator* prop,
                                        const reco::BeamSpot& bs,
                                        float vtxTime,
                                        float vtxTimeError,
                                        TransientTrackingRecHit::ConstRecHitContainer& output,
                                        MTDHitMatchingInfo& bestHit) const {
    std::set<MTDHitMatchingInfo> hitsInLayer;
    bool hitMatched = false;

    using namespace std::placeholders;
    auto find_hits = std::bind(find_hits_in_dets,
                               std::cref(hits),
                               std::cref(traj),
                               ilay,
                               std::cref(tsos),
                               pmag2,
                               pathlength0,
                               trs0,
                               _1,
                               _2,
                               _3,
                               std::cref(bs),
                               bsTimeSpread_,
                               prop,
                               theEstimator_.get(),
                               std::ref(hitsInLayer));

    const bool matchVertex = vtxTimeError > 0.f;
    if (matchVertex) {
      find_hits(vtxTime, vtxTimeError, true);
    } else {
      find_hits(0, 0, false);
    }

    const float spaceChi2Cut = ilay->isBarrel() ? btlChi2Cut_ : etlChi2Cut_;
    const float timeChi2Cut = ilay->isBarrel() ? btlTimeChi2Cut_ : etlTimeChi2Cut_;

    if (!hitsInLayer.empty()) {
      auto const& firstHit = *hitsInLayer.begin();
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: matching trial 1: estChi2= " << firstHit.estChi2
                                       << " timeChi2= " << firstHit.timeChi2;
      if (firstHit.estChi2 < spaceChi2Cut && firstHit.timeChi2 < timeChi2Cut) {
        hitMatched = true;
        output.push_back(hitbuilder_->build(firstHit.hit));
        if (firstHit < bestHit)
          bestHit = firstHit;
      }
    }

    if (matchVertex && !hitMatched) {
      //try a second search with beamspot hypothesis
      hitsInLayer.clear();
      find_hits(0, 0, false);
      if (!hitsInLayer.empty()) {
        auto const& firstHit = *hitsInLayer.begin();
        LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: matching trial 2: estChi2= " << firstHit.estChi2
                                         << " timeChi2= " << firstHit.timeChi2;
        if (firstHit.timeChi2 < timeChi2Cut && firstHit.estChi2 < spaceChi2Cut) {
          hitMatched = true;
          output.push_back(hitbuilder_->build(firstHit.hit));
          if (firstHit < bestHit)
            bestHit = firstHit;
        }
      }
    }

#ifdef EDM_ML_DEBUG
    if (hitMatched) {
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: matched hit with time: " << bestHit.hit->time()
                                       << " +/- " << bestHit.hit->timeError();
    } else {
      LogTrace("TrackExtenderWithMTD") << "TrackExtenderWithMTD: no matched hit";
    }
#endif
  }

}  // namespace mtd
