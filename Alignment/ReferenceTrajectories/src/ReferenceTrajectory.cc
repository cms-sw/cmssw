//  Author     : Gero Flucke (based on code by Edmund Widl replacing ORCA's TkReferenceTrack)
//  date       : 2006/09/17
//  last update: $Date: 2008/02/15 15:38:47 $
//  by         : $Author: ewidl $

#include "Alignment/ReferenceTrajectories/interface/ReferenceTrajectory.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/GeometrySurface/interface/LocalError.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
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

//__________________________________________________________________________________

ReferenceTrajectory::ReferenceTrajectory(const TrajectoryStateOnSurface &refTsos,
					 const TransientTrackingRecHit::ConstRecHitContainer
					 &recHits, bool hitsAreReverse,
					 const MagneticField *magField, 
					 MaterialEffects materialEffects,
					 PropagationDirection propDir,
					 double mass) 
  : ReferenceTrajectoryBase( refTsos.localParameters().mixedFormatVector().kSize,
			     numberOfUsedRecHits(recHits) )
{
  // no check against magField == 0

  theParameters = asHepVector<5>( refTsos.localParameters().mixedFormatVector() );

  if (hitsAreReverse) {
    TransientTrackingRecHit::ConstRecHitContainer fwdRecHits;
    fwdRecHits.reserve(recHits.size());
    for (TransientTrackingRecHit::ConstRecHitContainer::const_reverse_iterator it=recHits.rbegin();
	 it != recHits.rend(); ++it) {
      fwdRecHits.push_back(*it);
    }
    theValidityFlag = this->construct(refTsos, fwdRecHits, mass, materialEffects, propDir, magField);
  } else {
    theValidityFlag = this->construct(refTsos, recHits, mass, materialEffects, propDir, magField);
  }
}


//__________________________________________________________________________________

ReferenceTrajectory::ReferenceTrajectory( unsigned int nPar, unsigned int nHits )
  : ReferenceTrajectoryBase( nPar, nHits )
{}


//__________________________________________________________________________________

bool ReferenceTrajectory::construct(const TrajectoryStateOnSurface &refTsos, 
				    const TransientTrackingRecHit::ConstRecHitContainer &recHits,
				    double mass, MaterialEffects materialEffects,
				    const PropagationDirection propDir, const MagneticField *magField)
{
  MaterialEffectsUpdator *aMaterialEffectsUpdator = this->createUpdator(materialEffects, mass);
  if (!aMaterialEffectsUpdator) return false;

  AlgebraicMatrix                 fullJacobian(theParameters.num_row(), theParameters.num_row());
  std::vector<AlgebraicMatrix>    allJacobians; 
  allJacobians.reserve(theNumberOfHits);

  TransientTrackingRecHit::ConstRecHitPointer  previousHitPtr;
  TrajectoryStateOnSurface                     previousTsos;
  AlgebraicSymMatrix              previousChangeInCurvature(theParameters.num_row(), 1);
  std::vector<AlgebraicSymMatrix> allCurvatureChanges; 
  allCurvatureChanges.reserve(theNumberOfHits);

  const LocalTrajectoryError zeroErrors(0., 0., 0., 0., 0.);

  std::vector<AlgebraicMatrix> allProjections;
  allProjections.reserve(theNumberOfHits);
  std::vector<AlgebraicSymMatrix> allDeltaParameterCovs;
  allDeltaParameterCovs.reserve(theNumberOfHits);

  unsigned int iRow = 0;
  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itRecHit;
  for ( itRecHit = recHits.begin(); itRecHit != recHits.end(); ++itRecHit ) { 

    const TransientTrackingRecHit::ConstRecHitPointer &hitPtr = *itRecHit;

    if ( !useRecHit( hitPtr ) ) continue;

    theRecHits.push_back(hitPtr);

    if (0 == iRow) { 
      // compute the derivatives of the reference-track's parameters w.r.t. the initial ones
      // derivative of the initial reference-track parameters w.r.t. themselves is of course the identity 
      fullJacobian = AlgebraicMatrix(theParameters.num_row(), theParameters.num_row(), 1);
      allJacobians.push_back(fullJacobian);
      theTsosVec.push_back(refTsos);
    } else {
      AlgebraicMatrix nextJacobian;
      TrajectoryStateOnSurface nextTsos;
      if (!this->propagate(previousHitPtr->det()->surface(), previousTsos,
			   hitPtr->det()->surface(), nextTsos,
			   nextJacobian, propDir, magField)) {
	return false; // stop if problem...
      }

      allJacobians.push_back(nextJacobian);
      fullJacobian = nextJacobian * previousChangeInCurvature * fullJacobian;
      theTsosVec.push_back(nextTsos);
    }

    // take material effects into account. since trajectory-state is constructed with errors equal zero,
    // the updated state contains only the uncertainties due to interactions in the current layer.
    const TrajectoryStateOnSurface tmpTsos(theTsosVec.back().localParameters(), zeroErrors,
					   theTsosVec.back().surface(), magField, beforeSurface);
    const TrajectoryStateOnSurface updatedTsos = aMaterialEffectsUpdator->updateState(tmpTsos, propDir);
    if ( theTsosVec.back().localParameters().charge() )
    {
      previousChangeInCurvature[0][0] = updatedTsos.localParameters().signedInverseMomentum() 
	/ theTsosVec.back().localParameters().signedInverseMomentum();
    }
    
    // get multiple-scattering covariance-matrix
    allDeltaParameterCovs.push_back( asHepMatrix<5>(updatedTsos.localError().matrix()) );

    allCurvatureChanges.push_back(previousChangeInCurvature);

    // projection-matrix tsos-parameters -> measurement-coordinates
    allProjections.push_back(hitPtr->projectionMatrix());

    // set start-parameters for next propagation. trajectory-state without error
    //  - no error propagation needed here.
    previousHitPtr = hitPtr;
    previousTsos   = TrajectoryStateOnSurface(updatedTsos.globalParameters(),
					      updatedTsos.surface(), beforeSurface);

    this->fillDerivatives(allProjections.back(), fullJacobian, iRow);

    AlgebraicVector mixedLocalParams = asHepVector<5>(theTsosVec.back().localParameters().mixedFormatVector());
    this->fillTrajectoryPositions(allProjections.back(), mixedLocalParams, iRow);
    this->fillMeasurementAndError(hitPtr, iRow, updatedTsos);

    iRow += nMeasPerHit;
  } // end of loop on hits

  if (materialEffects != none) {
    this->addMaterialEffectsCov(allJacobians, allProjections, allCurvatureChanges, allDeltaParameterCovs);
  }

  if (refTsos.hasError()) {
    AlgebraicSymMatrix parameterCov = asHepMatrix<5>(refTsos.localError().matrix());
    theTrajectoryPositionCov = parameterCov.similarity(theDerivatives);
  } else {
    theTrajectoryPositionCov = AlgebraicSymMatrix(theDerivatives.num_row(), 1);
  }

  delete aMaterialEffectsUpdator;

  return true;
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

bool ReferenceTrajectory::propagate(const BoundPlane &previousSurface, const TrajectoryStateOnSurface &previousTsos,
				    const BoundPlane &newSurface, TrajectoryStateOnSurface &newTsos, AlgebraicMatrix &newJacobian,
				    const PropagationDirection propDir, const MagneticField *magField) const
{
  // propagate to next layer
  AnalyticalPropagator aPropagator(magField, propDir);
  const std::pair<TrajectoryStateOnSurface, double> tsosWithPath =
    aPropagator.propagateWithPath(previousTsos, newSurface);

  // stop if propagation wasn't successful
  if (!tsosWithPath.first.isValid())  return false;

  // calculate derivative of reference-track parameters on the actual layer w.r.t. the ones
  // on the previous layer (both in global coordinates)
  const AnalyticalCurvilinearJacobian aJacobian(previousTsos.globalParameters(), 
						tsosWithPath.first.globalPosition(),
						tsosWithPath.first.globalMomentum(),
						tsosWithPath.second);
  const AlgebraicMatrix curvilinearJacobian = asHepMatrix<5,5>(aJacobian.jacobian());


  // jacobian of the track parameters on the previous layer for local->global transformation
  const JacobianLocalToCurvilinear startTrafo(previousSurface, previousTsos.localParameters(), *magField);
  const AlgebraicMatrix localToCurvilinear = asHepMatrix<5>(startTrafo.jacobian());

  // jacobian of the track parameters on the actual layer for global->local transformation
  const JacobianCurvilinearToLocal endTrafo(newSurface, tsosWithPath.first.localParameters(), *magField);
  const AlgebraicMatrix curvilinearToLocal = asHepMatrix<5>(endTrafo.jacobian());

  // compute derivative of reference-track parameters on the actual layer w.r.t. the ones on
  // the previous layer (both in their local representation)
  newJacobian = curvilinearToLocal * curvilinearJacobian * localToCurvilinear;
  newTsos     = tsosWithPath.first;

  return true;
}

//__________________________________________________________________________________

void ReferenceTrajectory::fillMeasurementAndError(const TransientTrackingRecHit::ConstRecHitPointer &hitPtr,
						  unsigned int iRow,
						  const TrajectoryStateOnSurface &updatedTsos)
{
  // get the measurements and their errors, use information updated with tsos if improving
  // (GF: Also for measurements or only for errors or do the former not change?)
  TransientTrackingRecHit::ConstRecHitPointer newHitPtr(hitPtr->canImproveWithTrack() ?
							hitPtr->clone(updatedTsos) : hitPtr);

  const LocalPoint localMeasurement    = newHitPtr->localPosition();
  const LocalError localMeasurementCov = newHitPtr->localPositionError();

  theMeasurements[iRow]   = localMeasurement.x();
  theMeasurements[iRow+1] = localMeasurement.y();
  theMeasurementsCov[iRow][iRow]     = localMeasurementCov.xx();
  theMeasurementsCov[iRow][iRow+1]   = localMeasurementCov.xy();
  theMeasurementsCov[iRow+1][iRow+1] = localMeasurementCov.yy();
}
  

//__________________________________________________________________________________

void ReferenceTrajectory::fillDerivatives(const AlgebraicMatrix &projection,
					  const AlgebraicMatrix &fullJacobian,
					  unsigned int iRow)
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

  for (unsigned int k = 1; k < allJacobians.size(); ++k) {
    // error-propagation to next layer
    paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allJacobians[k]);

    // get dependences for the measurements
    deltaMaterialEffectsCov = paramMaterialEffectsCov.similarity(allProjections[k]);
    materialEffectsCov[nMeasPerHit*k  ][nMeasPerHit*k  ] = deltaMaterialEffectsCov[0][0];
    materialEffectsCov[nMeasPerHit*k  ][nMeasPerHit*k+1] = deltaMaterialEffectsCov[0][1];
    materialEffectsCov[nMeasPerHit*k+1][nMeasPerHit*k  ] = deltaMaterialEffectsCov[1][0];
    materialEffectsCov[nMeasPerHit*k+1][nMeasPerHit*k+1] = deltaMaterialEffectsCov[1][1];

    // add uncertainties for the following layers due to scattering at this layer
    paramMaterialEffectsCov += allDeltaParameterCovs[k];

    tempParameterCov = paramMaterialEffectsCov;

    // compute "inter-layer-dependencies"
    for (unsigned int l = k+1; l < allJacobians.size(); ++l) {
      tempParameterCov   = allJacobians[l]   * allCurvatureChanges[l] * tempParameterCov;
      tempMeasurementCov = allProjections[l] * tempParameterCov       * allProjections[k].T();

      materialEffectsCov[nMeasPerHit*l][nMeasPerHit*k] = tempMeasurementCov[0][0];
      materialEffectsCov[nMeasPerHit*k][nMeasPerHit*l] = tempMeasurementCov[0][0];

      materialEffectsCov[nMeasPerHit*l][nMeasPerHit*k+1] = tempMeasurementCov[0][1];
      materialEffectsCov[nMeasPerHit*k+1][nMeasPerHit*l] = tempMeasurementCov[0][1];

      materialEffectsCov[nMeasPerHit*l+1][nMeasPerHit*k] = tempMeasurementCov[1][0];
      materialEffectsCov[nMeasPerHit*k][nMeasPerHit*l+1] = tempMeasurementCov[1][0];

      materialEffectsCov[nMeasPerHit*l+1][nMeasPerHit*k+1] = tempMeasurementCov[1][1];
      materialEffectsCov[nMeasPerHit*k+1][nMeasPerHit*l+1] = tempMeasurementCov[1][1];
    }

    // error-propagation to state after energy loss
    paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allCurvatureChanges[k]);
  }

  theMeasurementsCov += materialEffectsCov;
}


unsigned int ReferenceTrajectory::numberOfUsedRecHits( const TransientTrackingRecHit::ConstRecHitContainer &recHits ) const
{
  unsigned int nUsedHits = 0;
  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itHit;
  for ( itHit = recHits.begin(); itHit != recHits.end(); ++itHit ) if ( useRecHit( *itHit ) ) ++nUsedHits;
  return nUsedHits;
}


bool ReferenceTrajectory::useRecHit( const TransientTrackingRecHit::ConstRecHitPointer& hitPtr ) const
{
  return hitPtr->isValid();
}
