//  Author     : Gero Flucke (based on code by Edmund Widl replacing ORCA's TkReferenceTrack)
//  date       : 2006/09/17
//  last update: $Date$
//  by         : $Author$

#include "Alignment/CommonAlignmentAlgorithm/interface/ReferenceTrajectory.h"

#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"
#include "Geometry/Surface/interface/LocalError.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"

#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "TrackingTools/MaterialEffects/interface/MultipleScatteringUpdator.h"
#include "TrackingTools/MaterialEffects/interface/EnergyLossUpdator.h"
#include "TrackingTools/MaterialEffects/interface/CombinedMaterialEffectsUpdator.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

//__________________________________________________________________________________

ReferenceTrajectory::ReferenceTrajectory(const TrajectoryStateOnSurface &refTsos,
					 const TransientTrackingRecHit::ConstRecHitContainer 
					 &recHits,
					 const MagneticField *magField, 
					 MaterialEffects materialEffects, double mass) 
  : ReferenceTrajectoryBase(refTsos.localParameters().mixedFormatVector().num_row(), recHits.size())
{
  //  theRecHits    = recHits; // FIXME: do not want to copy!
  theParameters = refTsos.localParameters().mixedFormatVector();

  theValidityFlag = this->construct(refTsos, recHits, mass, materialEffects, magField);
}

//__________________________________________________________________________________

bool ReferenceTrajectory::construct(const TrajectoryStateOnSurface &refTsos, 
				    const TransientTrackingRecHit::ConstRecHitContainer &recHits,
				    double mass, MaterialEffects materialEffects,
				    const MagneticField *magField)
{

  MaterialEffectsUpdator *aMaterialEffectsUpdator = this->createUpdator(materialEffects, mass);
  if (!aMaterialEffectsUpdator) return false;

  AlgebraicMatrix                 fullJacobian(parameters().num_row(), parameters().num_row());
  std::vector<AlgebraicMatrix>    allJacobians(recHits.size());

  TransientTrackingRecHit::ConstRecHitPointer  previousHitPtr; // was 'startRec'...
  TrajectoryStateOnSurface                     previousTsos;   // was 'startTsos'...
  AlgebraicSymMatrix              previousChangeInCurvature(parameters().num_row(), 1); // was 'changeInCurvature'
  std::vector<AlgebraicSymMatrix> allCurvatureChanges(recHits.size());

  const LocalTrajectoryError zeroErrors(0., 0., 0., 0., 0.);

  std::vector<AlgebraicMatrix> allProjections(recHits.size());
  std::vector<AlgebraicSymMatrix> allDeltaParameterCovs(recHits.size()); // was deltaParameterCov

  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itRecHit = recHits.begin();
  for (unsigned int iRow = 0; itRecHit != recHits.end(); ++itRecHit, iRow += nMeasPerHit) { 
    const TransientTrackingRecHit::ConstRecHitPointer &hitPtr = *itRecHit;
    // GF FIXME: we have to care about invalid hits since also tracks with holes might be useful!
    TrajectoryStateOnSurface tmpTsos;
    if (0 == iRow) { 
      // compute the derivatives of the reference-track's parameters w.r.t. the initial ones
      // derivative of the initial reference-track parameters w.r.t. themselves is of course the identity 
      fullJacobian = AlgebraicSymMatrix(parameters().num_row(), 1); // FIXME: why 'Sym'?
      allJacobians.push_back(fullJacobian);
      theTsosVec.push_back(TrajectoryStateOnSurface(refTsos.globalParameters(),
						    refTsos.surface(), beforeSurface));
      // GF: why not just push_back(refTsos)? due to 'beforeSurface'? what about errors? 
      // we might be able to remove tmpTsos and use theTsosVec.back() instead later on?

      // take material effects into account. since trajectory-state is constructed with errors equal zero,
      // the updated state contains only the uncertainties due to interactions in the current layer.
      tmpTsos = TrajectoryStateOnSurface(refTsos.localParameters(), zeroErrors,
					 refTsos.surface(), magField, beforeSurface);
    } else {
      AlgebraicMatrix nextJacobian;
      TrajectoryStateOnSurface nextTsos;
      if (!this->propagate(previousTsos, hitPtr->det()->surface(), magField, nextJacobian, nextTsos)) {
	return false; // stop if problem...
      }

      allJacobians.push_back(nextJacobian);
      fullJacobian = nextJacobian * previousChangeInCurvature * fullJacobian;
      theTsosVec.push_back(nextTsos);

      // take material effects into account. since trajectory-state is constructed with errors equal zero,
      // the updated state contains only the uncertainties due to interactions in the current layer.
      tmpTsos = TrajectoryStateOnSurface(nextTsos.localParameters(), zeroErrors,
					 nextTsos.surface(), magField, beforeSurface);
    }

    // GF: all 'previous' settings originally duplicated in if () {} else {}
    const TrajectoryStateOnSurface updatedTsos = aMaterialEffectsUpdator->updateState(tmpTsos, alongMomentum);
    previousChangeInCurvature[0][0] = updatedTsos.localParameters().signedInverseMomentum() 
      / theTsosVec.back().localParameters().signedInverseMomentum();
    allCurvatureChanges.push_back(previousChangeInCurvature);
    // set start-parameters for next propagation. trajectory-state without error - no error propagation
    // needed here.
    previousHitPtr = hitPtr;
    previousTsos   = TrajectoryStateOnSurface(updatedTsos.globalParameters(), // GF why not copy? beforeSurface?
					      updatedTsos.surface(), beforeSurface);
    // GF: end of original duplication


    // projection-matrix tsos-parameters -> measurement-coordinates
    allProjections.push_back(hitPtr->projectionMatrix());
    // get multiple-scattering covariance-matrix GF: what is that? meaning 'material effects' instead of MS?
    allDeltaParameterCovs.push_back(updatedTsos.localError().matrix());

    this->fillDerivatives(allProjections.back(), fullJacobian, iRow);
    this->fillTrajectoryPositions(allProjections.back(),
				  theTsosVec.back().localParameters().mixedFormatVector(), iRow);
    this->fillMeasurementAndError(hitPtr, iRow);//, updatedTsos); FIXME: tsos needed?

  } // end of loop on hits

  if (materialEffects != none) {
    this->addMaterialEffectsCov(allJacobians, allProjections, allCurvatureChanges, allDeltaParameterCovs);
  }

  if (refTsos.hasError()) {
    theTrajectoryPositionCov = refTsos.localError().matrix().similarity(theDerivatives);
  } else {
    theTrajectoryPositionCov = AlgebraicSymMatrix(theDerivatives.num_row(), 1);
  }

  delete aMaterialEffectsUpdator;
  return false;
}

//__________________________________________________________________________________

MaterialEffectsUpdator*
ReferenceTrajectory::createUpdator(MaterialEffects materialEffects, double mass) const
{
  switch (materialEffects) {
    // MultipleScatteringUpdator doesn't change the trajectory-state
    // during update and can therefore be used if material effects should be ignored:
  case none:
  case multipleScattering: 
    return new MultipleScatteringUpdator(mass);
  case energyLoss:
    return new EnergyLossUpdator(mass);
  case combined:
    return new CombinedMaterialEffectsUpdator(mass);
  }

  return 0;
}

//__________________________________________________________________________________

bool ReferenceTrajectory::propagate(const TrajectoryStateOnSurface &previousTsos, const BoundPlane &surface,
				    const MagneticField *magField,
				    AlgebraicMatrix &newJacobian, TrajectoryStateOnSurface &newTsos) const
{
  // propagate to next layer
  AnalyticalPropagator aPropagator(magField);
  const std::pair<TrajectoryStateOnSurface, double> tsosWithPath = // was 'endTsosWithPath'
    aPropagator.propagateWithPath(previousTsos, surface);

  // stop if propagation wasn't successful
  if (!tsosWithPath.first.isValid()) return false; // GF FIXME: what does that mean?

  // calculate derivative of reference-track parameters on the actual layer w.r.t. the ones
  // on the previous layer (both in global coordinates)
  const AnalyticalCurvilinearJacobian aJacobian(previousTsos.globalParameters(), 
						tsosWithPath.first.globalPosition(),
						tsosWithPath.first.globalMomentum(),
						tsosWithPath.second);
  const AlgebraicMatrix &curvilinearJacobian = aJacobian.jacobian();


  // jacobian of the track parameters on the previous layer for local->global transformation
  const JacobianLocalToCurvilinear startTrafo(surface, previousTsos.localParameters(), *magField);
  const AlgebraicMatrix &localToCurvilinear = startTrafo.jacobian();

  // jacobian of the track parameters on the actual layer for global->local transformation
  const JacobianCurvilinearToLocal endTrafo(surface, tsosWithPath.first.localParameters(), *magField);
  const AlgebraicMatrix &curvilinearToLocal = endTrafo.jacobian();

  // compute derivative of reference-track parameters on the actual layer w.r.t. the ones on
  // the previous layer (both in their local representation)
  newJacobian = curvilinearToLocal * curvilinearJacobian * localToCurvilinear;
  newTsos     = tsosWithPath.first;

  return true;
}

//__________________________________________________________________________________

void ReferenceTrajectory::fillMeasurementAndError(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
						  unsigned int iRow) 
//const TrajectoryStateOnSurface &updatedTsos) FIXME: needed?
{
  // get the measurements and their errors FIXME: ORCA had these 'updatedTsos' as argument...???
  const LocalPoint localMeasurement    = hitPtr->localPosition();// updatedTsos); 
  const LocalError localMeasurementCov = hitPtr->localPositionError(); // updatedTsos);

  theMeasurements[iRow]   = localMeasurement.x();
  theMeasurements[iRow+1] = localMeasurement.y();
  theMeasurementsCov[iRow][iRow]     = localMeasurementCov.xx();
  theMeasurementsCov[iRow][iRow+1]   = localMeasurementCov.xy();
  theMeasurementsCov[iRow+1][iRow+1] = localMeasurementCov.yy();
}
  

//__________________________________________________________________________________

void ReferenceTrajectory::fillDerivatives(const AlgebraicMatrix &projection,
					  const AlgebraicMatrix &fullJacobian, unsigned int iRow)
{
  // derivatives of the local coordinates of the reference track w.r.t. to the inital track-parameters
  const AlgebraicMatrix projectedJacobian(projection * fullJacobian);
  for (int i = 0; i < parameters().num_row(); ++i) {
    theDerivatives[iRow  ][i] = projectedJacobian[0][i];
    theDerivatives[iRow+1][i] = projectedJacobian[1][i];
  }
}

//__________________________________________________________________________________

void ReferenceTrajectory::fillTrajectoryPositions(const AlgebraicMatrix &projection, 
						  const AlgebraicVector &mixedLocalParams, 
						  unsigned int iRow)
{
  // get the local coordinates of the reference trajectory
  const AlgebraicVector localPosition(projection * mixedLocalParams);
  theTrajectoryPositions[iRow] = localPosition[0];
  theTrajectoryPositions[iRow+1] = localPosition[1];
}

//__________________________________________________________________________________

void ReferenceTrajectory::addMaterialEffectsCov(const std::vector<AlgebraicMatrix> &allJacobians, 
						const std::vector<AlgebraicMatrix> &allProjections,
						const std::vector<AlgebraicSymMatrix> &allCurvatureChanges,
						const std::vector<AlgebraicSymMatrix> &allDeltaParameterCovs)
{
  // the uncertainty due to multiple scattering is 'transferred' to the error matrix of the hits

  AlgebraicSymMatrix materialEffectsCov(nMeasPerHit * allJacobians.size(), 0);

  // additional covariance-matrix of the measurements due to material-effects (single measurement)
  AlgebraicSymMatrix deltaMaterialEffectsCov;

  // additional covariance-matrix of the parameters due to material-effects
  AlgebraicSymMatrix paramMaterialEffectsCov(allDeltaParameterCovs[0]); //initialization

  AlgebraicMatrix tempParameterCov;
  AlgebraicMatrix tempMeasurementCov;

  // GF: all following '2' should probably become 'nMeasPerHit'...

  for (unsigned int k = 1; k < allJacobians.size(); ++k) {
    // error-propagation to next layer
    paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allJacobians[k]);

    // get dependences for the measurements
    deltaMaterialEffectsCov = paramMaterialEffectsCov.similarity(allProjections[k]);
    materialEffectsCov[2*k  ][2*k  ] = deltaMaterialEffectsCov[0][0];
    materialEffectsCov[2*k  ][2*k+1] = deltaMaterialEffectsCov[0][1];
    materialEffectsCov[2*k+1][2*k  ] = deltaMaterialEffectsCov[1][0];
    materialEffectsCov[2*k+1][2*k+1] = deltaMaterialEffectsCov[1][1];

    // add uncertainties for the following layers due to scattering at this layer
    paramMaterialEffectsCov += allDeltaParameterCovs[k];

    tempParameterCov = paramMaterialEffectsCov;

    // compute "inter-layer-dependencies"
    for (unsigned int l = k+1; l < allJacobians.size(); ++l) {
      tempParameterCov   = allJacobians[l]   * allCurvatureChanges[l] * tempParameterCov;
      tempMeasurementCov = allProjections[l] * tempParameterCov       * allProjections[k].T();

      materialEffectsCov[2*l][2*k] = tempMeasurementCov[0][0];
      materialEffectsCov[2*k][2*l] = tempMeasurementCov[0][0];

      materialEffectsCov[2*l][2*k+1] = tempMeasurementCov[0][1];
      materialEffectsCov[2*k+1][2*l] = tempMeasurementCov[0][1];

      materialEffectsCov[2*l+1][2*k] = tempMeasurementCov[1][0];
      materialEffectsCov[2*k][2*l+1] = tempMeasurementCov[1][0];

      materialEffectsCov[2*l+1][2*k+1] = tempMeasurementCov[1][1];
      materialEffectsCov[2*k+1][2*l+1] = tempMeasurementCov[1][1];
    }

    // error-propagation to state after energy loss
    paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allCurvatureChanges[k]);
  }

  theMeasurementsCov += materialEffectsCov;
}

