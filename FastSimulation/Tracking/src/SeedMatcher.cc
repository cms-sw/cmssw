#include "FastSimulation/Tracking/interface/SeedMatcher.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "DataFormats/TrackerRecHit2D/interface/FastTrackerRecHit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "iostream"

std::vector<int32_t> SeedMatcher::matchRecHitCombinations(
    const TrajectorySeed& seed,
    const FastTrackerRecHitCombinationCollection& recHitCombinationCollection,
    const edm::SimTrackContainer& simTrackCollection,
    double maxMatchEstimator,
    const Propagator& propagator,
    const MagneticField& magneticField,
    const TrackerGeometry& trackerGeometry) {
  // container for result
  std::vector<int32_t> result;

  // seed state
  PTrajectoryStateOnDet ptod = seed.startingState();
  DetId seedState_detId(ptod.detId());
  const GeomDet* seedState_det = trackerGeometry.idToDet(seedState_detId);
  const Surface* seedState_surface = &seedState_det->surface();
  TrajectoryStateOnSurface seedState(trajectoryStateTransform::transientState(ptod, seedState_surface, &magneticField));

  // find matches
  int nSimTracks = simTrackCollection.size();
  for (unsigned recHitCombinationIndex = 0; recHitCombinationIndex < recHitCombinationCollection.size();
       recHitCombinationIndex++) {
    const auto& recHitCombination = recHitCombinationCollection[recHitCombinationIndex];
    int simTrackIndex = recHitCombination.back()->simTrackId(0);
    if (simTrackIndex < 0 || simTrackIndex >= nSimTracks) {
      throw cms::Exception("SeedMatcher") << "SimTrack index out of range: " << simTrackIndex << std::endl;
    }
    const auto& simTrack = simTrackCollection[recHitCombination.back()->simTrackId(0)];
    double matchEstimator = matchSimTrack(seedState, simTrack, propagator, magneticField);
    if (matchEstimator < maxMatchEstimator) {
      result.push_back(recHitCombinationIndex);
    }
  }
  return result;
}

double SeedMatcher::matchSimTrack(const TrajectoryStateOnSurface& seedState,
                                  const SimTrack& simTrack,
                                  const Propagator& propagator,
                                  const MagneticField& magneticField) {
  // simtrack and simvertex at tracker surface
  GlobalPoint simTrack_atTrackerSurface_position(simTrack.trackerSurfacePosition().x(),
                                                 simTrack.trackerSurfacePosition().y(),
                                                 simTrack.trackerSurfacePosition().z());
  GlobalVector simTrack_atTrackerSurface_momentum(simTrack.trackerSurfaceMomentum().x(),
                                                  simTrack.trackerSurfaceMomentum().y(),
                                                  simTrack.trackerSurfaceMomentum().z());

  // no match if seedstate and simtrack in oposite direction
  if (simTrack_atTrackerSurface_position.basicVector().dot(simTrack_atTrackerSurface_momentum.basicVector()) *
          seedState.globalPosition().basicVector().dot(seedState.globalMomentum().basicVector()) <
      0.) {
    return 9999.;
  }

  // find simtrack state on surface of seed state
  GlobalTrajectoryParameters simTrack_atTrackerSurface_parameters(
      simTrack_atTrackerSurface_position, simTrack_atTrackerSurface_momentum, simTrack.charge(), &magneticField);
  FreeTrajectoryState simtrack_atTrackerSurface_state(simTrack_atTrackerSurface_parameters);
  TrajectoryStateOnSurface simtrack_atSeedStateSurface_state =
      propagator.propagate(simtrack_atTrackerSurface_state, seedState.surface());

  // simtrack does not propagate to surface of seed state
  if (!simtrack_atSeedStateSurface_state.isValid()) {
    return 9999.;
  }

  // simtrack and seed state have opposite direction
  if (simtrack_atSeedStateSurface_state.globalPosition().basicVector().dot(
          simtrack_atSeedStateSurface_state.globalMomentum().basicVector()) *
          seedState.globalPosition().basicVector().dot(seedState.globalMomentum().basicVector()) <
      0.) {
    return 9999.;
  }

  AlgebraicVector5 difference(seedState.localParameters().vector() -
                              simtrack_atSeedStateSurface_state.localParameters().vector());
  AlgebraicSymMatrix55 error(seedState.localError().matrix());
  if (!error.Invert()) {
    edm::LogWarning("FastSim SeedToSimTrackMatcher") << "Cannot invert seed state error matrix => no match...";
    return 9999.;
  }

  return ROOT::Math::Similarity(difference, error);
}
