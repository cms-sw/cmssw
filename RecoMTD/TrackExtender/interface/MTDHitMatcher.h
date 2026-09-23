#ifndef RecoMTD_TrackExtender_MTDHitMatcher_h
#define RecoMTD_TrackExtender_MTDHitMatcher_h

#include <memory>
#include <set>
#include <vector>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/TimingTools/interface/MTDHitMatchingInfo.h"
#include "RecoMTD/TimingTools/interface/TrackSegments.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"

namespace mtd {

  struct MTDHitMatchResult {
    TransientTrackingRecHit::ConstRecHitContainer hits;
    MTDHitMatchingInfo bestHit;
  };

  class MTDHitMatcher {
  public:
    MTDHitMatcher(const edm::ParameterSet& pset, edm::ConsumesCollector iC);

    void setServices(const edm::EventSetup& es);

    MTDHitMatchResult matchBTL(const TrajectoryStateOnSurface& tsos,
                               const Trajectory& traj,
                               float pmag2,
                               float pathlength0,
                               const TrackSegments& trs0,
                               const MTDTrackingDetSetVector& hits,
                               const MTDDetLayerGeometry* geo,
                               const Propagator* prop,
                               const reco::BeamSpot& bs,
                               float vtxTime,
                               float vtxTimeError) const;

    MTDHitMatchResult matchETL(const TrajectoryStateOnSurface& tsos,
                               const Trajectory& traj,
                               float pmag2,
                               float pathlength0,
                               const TrackSegments& trs0,
                               const MTDTrackingDetSetVector& hits,
                               const MTDDetLayerGeometry* geo,
                               const Propagator* prop,
                               const reco::BeamSpot& bs,
                               float vtxTime,
                               float vtxTimeError) const;

    float etlTimeChi2Cut() const { return etlTimeChi2Cut_; }

    static void fillPSetDescription(edm::ParameterSetDescription& desc);

  private:
    MTDHitMatchResult matchLayers(const std::vector<const DetLayer*>& layers,
                                  const TrajectoryStateOnSurface& tsos,
                                  const Trajectory& traj,
                                  float pmag2,
                                  float pathlength0,
                                  const TrackSegments& trs0,
                                  const MTDTrackingDetSetVector& hits,
                                  const Propagator* prop,
                                  const reco::BeamSpot& bs,
                                  float vtxTime,
                                  float vtxTimeError) const;

    void fillMatchingHits(const DetLayer* ilay,
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
                          MTDHitMatchingInfo& bestHit) const;

    const float btlChi2Cut_;
    const float btlTimeChi2Cut_;
    const float etlChi2Cut_;
    const float etlTimeChi2Cut_;
    const float bsTimeSpread_;
    const std::string mtdRecHitBuilder_;

    std::unique_ptr<MeasurementEstimator> theEstimator_;

    edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> hitBuilderToken_;
    edm::ESHandle<TransientTrackingRecHitBuilder> hitbuilder_;
  };

}  // namespace mtd

#endif
