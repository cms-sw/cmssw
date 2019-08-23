#ifndef SeedFromNuclearInteraction_H
#define SeedFromNuclearInteraction_H

#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoTracker/NuclearSeedGenerator/interface/TangentHelix.h"

class FreeTrajectoryState;

class SeedFromNuclearInteraction {
private:
  typedef TrajectoryMeasurement TM;
  typedef TrajectoryStateOnSurface TSOS;
  typedef edm::OwnVector<TrackingRecHit> recHitContainer;
  typedef TransientTrackingRecHit::ConstRecHitPointer ConstRecHitPointer;
  typedef std::vector<ConstRecHitPointer> ConstRecHitContainer;

public:
  SeedFromNuclearInteraction(const Propagator* prop, const TrackerGeometry* geom, const edm::ParameterSet& iConfig);

  virtual ~SeedFromNuclearInteraction() {}

  /// Fill all data members from 2 TM's where the first one is supposed to be at the interaction point
  void setMeasurements(const TSOS& tsosAtInteractionPoint, ConstRecHitPointer ihit, ConstRecHitPointer ohit);

  /// Fill all data members from 1 TSOS and 2 rec Hits and using the circle associated to the primary track as constraint
  void setMeasurements(TangentHelix& primHelix,
                       const TSOS& inner_TSOS,
                       ConstRecHitPointer ihit,
                       ConstRecHitPointer ohit);

  PTrajectoryStateOnDet const& trajectoryState() const { return pTraj; }

  FreeTrajectoryState* stateWithError() const;

  FreeTrajectoryState* stateWithError(TangentHelix& helix) const;

  PropagationDirection direction() const { return alongMomentum; }

  recHitContainer hits() const;

  TrajectorySeed TrajSeed() const { return TrajectorySeed(trajectoryState(), hits(), direction()); }

  bool isValid() const { return isValid_; }

  const TSOS& updatedTSOS() const { return *updatedTSOS_; }

  const TSOS& initialTSOS() const { return *initialTSOS_; }

  GlobalPoint outerHitPosition() const {
    return theTrackerGeom->idToDet(outerHitDetId())->surface().toGlobal(outerHit_->localPosition());
  }

  DetId outerHitDetId() const { return outerHit_->geographicalId(); }

  ConstRecHitPointer outerHit() const { return outerHit_; }

  /// Return the rotation matrix to be applied to get parameters in
  /// a framework where the z direction is along perp
  AlgebraicMatrix33 rotationMatrix(const GlobalVector& perp) const;

private:
  bool isValid_; /**< check if the seed is valid */

  ConstRecHitContainer theHits; /**< all the hits to be used to update the */
                                /*   initial freeTS and to be fitted       */

  ConstRecHitPointer innerHit_; /**< Pointer to the hit of the inner TM */
  ConstRecHitPointer outerHit_; /**< Pointer to the outer hit */

  std::shared_ptr<TSOS> updatedTSOS_; /**< Final TSOS */

  std::shared_ptr<TSOS> initialTSOS_; /**< Initial TSOS used as input */

  std::shared_ptr<FreeTrajectoryState> freeTS_; /**< Initial FreeTrajectoryState */

  PTrajectoryStateOnDet pTraj; /**< the final persistent TSOS */

  // input parameters

  double ptMin; /**< Minimum transverse momentum of the seed */

  const Propagator* thePropagator;
  const TrackerGeometry* theTrackerGeom;

  bool construct();
};
#endif
