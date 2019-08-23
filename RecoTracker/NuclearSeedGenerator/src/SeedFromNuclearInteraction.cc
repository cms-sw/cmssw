#include <memory>

#include "RecoTracker/NuclearSeedGenerator/interface/SeedFromNuclearInteraction.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

SeedFromNuclearInteraction::SeedFromNuclearInteraction(const Propagator* prop,
                                                       const TrackerGeometry* geom,
                                                       const edm::ParameterSet& iConfig)
    : ptMin(iConfig.getParameter<double>("ptMin")), thePropagator(prop), theTrackerGeom(geom) {
  isValid_ = true;
  initialTSOS_ = std::make_shared<TrajectoryStateOnSurface>();
  updatedTSOS_ = std::make_shared<TrajectoryStateOnSurface>();
  freeTS_ = std::make_shared<FreeTrajectoryState>();
}

//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(const TSOS& inner_TSOS,
                                                 ConstRecHitPointer ihit,
                                                 ConstRecHitPointer ohit) {
  // delete pointer to TrackingRecHits
  theHits.clear();

  // get the inner and outer transient TrackingRecHits
  innerHit_ = ihit;
  outerHit_ = ohit;

  //theHits.push_back(  inner_TM.recHit() ); // put temporarily - TODO: remove this line
  theHits.push_back(outerHit_);

  initialTSOS_.reset(new TrajectoryStateOnSurface(inner_TSOS));

  // calculate the initial FreeTrajectoryState.
  freeTS_.reset(stateWithError());

  // check transverse momentum
  if (freeTS_->momentum().perp() < ptMin) {
    isValid_ = false;
  } else {
    // convert freeTS_ into a persistent TSOS on the outer surface
    isValid_ = construct();
  }
}
//----------------------------------------------------------------------
void SeedFromNuclearInteraction::setMeasurements(TangentHelix& thePrimaryHelix,
                                                 const TSOS& inner_TSOS,
                                                 ConstRecHitPointer ihit,
                                                 ConstRecHitPointer ohit) {
  // delete pointer to TrackingRecHits
  theHits.clear();

  // get the inner and outer transient TrackingRecHits
  innerHit_ = ihit;
  outerHit_ = ohit;

  GlobalPoint innerPos =
      theTrackerGeom->idToDet(innerHit_->geographicalId())->surface().toGlobal(innerHit_->localPosition());
  GlobalPoint outerPos =
      theTrackerGeom->idToDet(outerHit_->geographicalId())->surface().toGlobal(outerHit_->localPosition());

  TangentHelix helix(thePrimaryHelix, outerPos, innerPos);

  theHits.push_back(innerHit_);
  theHits.push_back(outerHit_);

  initialTSOS_.reset(new TrajectoryStateOnSurface(inner_TSOS));

  // calculate the initial FreeTrajectoryState from the inner and outer TM assuming that the helix equation is already known.
  freeTS_.reset(stateWithError(helix));

  if (freeTS_->momentum().perp() < ptMin) {
    isValid_ = false;
  } else {
    // convert freeTS_ into a persistent TSOS on the outer surface
    isValid_ = construct();
  }
}
//----------------------------------------------------------------------
FreeTrajectoryState* SeedFromNuclearInteraction::stateWithError() const {
  // Calculation of the helix assuming that the secondary track has the same direction
  // than the primary track and pass through the inner and outer hits.
  GlobalVector direction = initialTSOS_->globalDirection();
  GlobalPoint inner = initialTSOS_->globalPosition();
  TangentHelix helix(direction, inner, outerHitPosition());

  return stateWithError(helix);
}
//----------------------------------------------------------------------
FreeTrajectoryState* SeedFromNuclearInteraction::stateWithError(TangentHelix& helix) const {
  //   typedef TkRotation<float> Rotation;

  GlobalVector dirAtVtx = helix.directionAtVertex();
  const MagneticField& mag = initialTSOS_->globalParameters().magneticField();

  // Get the global parameters of the trajectory
  // we assume that the magnetic field at the vertex is equal to the magnetic field at the inner TM.
  GlobalTrajectoryParameters gtp(
      helix.vertexPoint(), dirAtVtx, helix.charge(mag.inTesla(helix.vertexPoint()).z()) / helix.rho(), 0, &mag);

  // Error matrix in a frame where z is in the direction of the track at the vertex
  AlgebraicSymMatrix66 primaryError(initialTSOS_->cartesianError().matrix());
  double p_max = initialTSOS_->globalParameters().momentum().mag();
  AlgebraicMatrix33 rot = this->rotationMatrix(dirAtVtx);

  AlgebraicMatrix66 globalRotation;
  globalRotation.Place_at(rot, 0, 0);
  globalRotation.Place_at(rot, 3, 3);
  AlgebraicSymMatrix66 primaryErrorInNewFrame = ROOT::Math::Similarity(globalRotation, primaryError);

  AlgebraicSymMatrix66 secondaryErrorInNewFrame = AlgebraicMatrixID();
  double p_perp_max = 2;  // energy max of a secondary track emited perpendicularly to the
                          // primary track is +/- 2 GeV
  secondaryErrorInNewFrame(0, 0) = primaryErrorInNewFrame(0, 0) + helix.vertexError() * p_perp_max / p_max;
  secondaryErrorInNewFrame(1, 1) = primaryErrorInNewFrame(1, 1) + helix.vertexError() * p_perp_max / p_max;
  secondaryErrorInNewFrame(2, 2) = helix.vertexError() * helix.vertexError();
  secondaryErrorInNewFrame(3, 3) = p_perp_max * p_perp_max;
  secondaryErrorInNewFrame(4, 4) = p_perp_max * p_perp_max;
  secondaryErrorInNewFrame(5, 5) = p_max * p_max;

  AlgebraicSymMatrix66 secondaryError = ROOT::Math::SimilarityT(globalRotation, secondaryErrorInNewFrame);

  return new FreeTrajectoryState(gtp, CartesianTrajectoryError(secondaryError));
}

//----------------------------------------------------------------------
bool SeedFromNuclearInteraction::construct() {
  // loop on all hits in theHits
  KFUpdator theUpdator;

  const TrackingRecHit* hit = nullptr;

  LogDebug("NuclearSeedGenerator") << "Seed ** initial state " << freeTS_->cartesianError().matrix();

  for (unsigned int iHit = 0; iHit < theHits.size(); iHit++) {
    hit = theHits[iHit]->hit();
    TrajectoryStateOnSurface state =
        (iHit == 0)
            ? thePropagator->propagate(*freeTS_, theTrackerGeom->idToDet(hit->geographicalId())->surface())
            : thePropagator->propagate(*updatedTSOS_, theTrackerGeom->idToDet(hit->geographicalId())->surface());

    if (!state.isValid())
      return false;

    const TransientTrackingRecHit::ConstRecHitPointer& tth = theHits[iHit];
    updatedTSOS_.reset(new TrajectoryStateOnSurface(theUpdator.update(state, *tth)));
  }

  LogDebug("NuclearSeedGenerator") << "Seed ** updated state " << updatedTSOS_->cartesianError().matrix();

  pTraj = trajectoryStateTransform::persistentState(*updatedTSOS_, outerHitDetId().rawId());
  return true;
}

//----------------------------------------------------------------------
edm::OwnVector<TrackingRecHit> SeedFromNuclearInteraction::hits() const {
  recHitContainer _hits;
  for (ConstRecHitContainer::const_iterator it = theHits.begin(); it != theHits.end(); it++) {
    _hits.push_back(it->get()->hit()->clone());
  }
  return _hits;
}
//----------------------------------------------------------------------
AlgebraicMatrix33 SeedFromNuclearInteraction::rotationMatrix(const GlobalVector& perp) const {
  AlgebraicMatrix33 result;

  // z axis coincides with perp
  GlobalVector zAxis = perp.unit();

  // x axis has no global Z component
  GlobalVector xAxis;
  if (zAxis.x() != 0 || zAxis.y() != 0) {
    // precision is not an issue here, just protect against divizion by zero
    xAxis = GlobalVector(-zAxis.y(), zAxis.x(), 0).unit();
  } else {  // perp coincides with global Z
    xAxis = GlobalVector(1, 0, 0);
  }

  // y axis obtained by cross product
  GlobalVector yAxis(zAxis.cross(xAxis));

  result(0, 0) = xAxis.x();
  result(0, 1) = xAxis.y();
  result(0, 2) = xAxis.z();
  result(1, 0) = yAxis.x();
  result(1, 1) = yAxis.y();
  result(1, 2) = yAxis.z();
  result(2, 0) = zAxis.x();
  result(2, 1) = zAxis.y();
  result(2, 2) = zAxis.z();
  return result;
}
