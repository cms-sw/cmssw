#ifndef RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
#define RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h

#include <cstdint>

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2OTCAModuleParamsHost.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits {

  // the Phase2TrackerCluster1D quantities that the CPE reads, and the CA module of the cluster
  struct Cluster {
    uint16_t firstStrip;
    uint16_t column;
    uint16_t size;
    uint16_t module;
  };

  struct BeamSpotPosition {
    double x;
    double y;
    double z;
  };

  // one CA hit per cluster, in the same order, relative to the beam spot; the hits of CA module i get detectorIndex
  // firstDetectorIndex + i
  void makeCAHits(Queue& queue,
                  Phase2OTCAModuleParamsSoA::ConstView modules,
                  Cluster const* clusters,
                  BeamSpotPosition beamSpot,
                  uint16_t firstDetectorIndex,
                  ::reco::TrackingRecHitView hits);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::phase2OTCAHits

#endif  // RecoLocalTracker_Phase2TrackerRecHits_plugins_alpaka_Phase2OTCAHitsKernels_h
