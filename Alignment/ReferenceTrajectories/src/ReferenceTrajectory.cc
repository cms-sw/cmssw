//  Author     : Gero Flucke (based on code by Edmund Widl replacing ORCA's TkReferenceTrack)
//  date       : 2006/09/17
//  last update: $Date: 2009/09/15 16:21:55 $
//  by         : $Author: ckleinw $

#include <memory>

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
 : ReferenceTrajectoryBase( 
   (materialEffects >= brokenLinesCoarse) ? 1 : refTsos.localParameters().mixedFormatVector().kSize, 
   recHits.size(), 
   (materialEffects >= brokenLinesCoarse) ? 2*recHits.size()   : ( (materialEffects == breakPoints) ? 2*recHits.size()-2 : 0) , 
   (materialEffects >= brokenLinesCoarse) ? 2*recHits.size()-4 : ( (materialEffects == breakPoints) ? 2*recHits.size()-2 : 0) )	 
{

  // no check against magField == 0  
  theParameters = asHepVector<5>( refTsos.localParameters().mixedFormatVector() );
  // CHK for debug and coarse broken lines
  theGlobalPars = refTsos.globalParameters().vector();
    
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

ReferenceTrajectory::ReferenceTrajectory( unsigned int nPar, unsigned int nHits, MaterialEffects materialEffects)
  : ReferenceTrajectoryBase( 
   (materialEffects >= brokenLinesCoarse) ? 1 : nPar, 
   nHits, 
   (materialEffects >= brokenLinesCoarse) ? 2*nHits   : ( (materialEffects == breakPoints) ? 2*nHits-2 : 0) , 
   (materialEffects >= brokenLinesCoarse) ? 2*nHits-4 : ( (materialEffects == breakPoints) ? 2*nHits-2 : 0) )  
{}


//__________________________________________________________________________________

bool ReferenceTrajectory::construct(const TrajectoryStateOnSurface &refTsos, 
				    const TransientTrackingRecHit::ConstRecHitContainer &recHits,
				    double mass, MaterialEffects materialEffects,
				    const PropagationDirection propDir, const MagneticField *magField)
{   
  const SurfaceSide surfaceSide = this->surfaceSide(propDir);
  // auto_ptr to avoid memory leaks in case of not reaching delete at end of method:
  std::auto_ptr<MaterialEffectsUpdator> aMaterialEffectsUpdator
    (this->createUpdator(materialEffects, mass));
  if (!aMaterialEffectsUpdator.get()) return false; // empty auto_ptr

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

  // CHK
  std::vector<AlgebraicMatrix> allLocalToCurv;
  allLocalToCurv.reserve(theNumberOfHits); 
  std::vector<double> allSteps;
  allSteps.reserve(theNumberOfHits); 
  std::vector<AlgebraicMatrix>    allCurvlinJacobians; 
  allCurvlinJacobians.reserve(theNumberOfHits);
   
  unsigned int iRow = 0;
  TransientTrackingRecHit::ConstRecHitContainer::const_iterator itRecHit;
  for ( itRecHit = recHits.begin(); itRecHit != recHits.end(); ++itRecHit ) { 

    const TransientTrackingRecHit::ConstRecHitPointer &hitPtr = *itRecHit;
    theRecHits.push_back(hitPtr);

    if (0 == iRow) { 
      // compute the derivatives of the reference-track's parameters w.r.t. the initial ones
      // derivative of the initial reference-track parameters w.r.t. themselves is of course the identity 
      fullJacobian = AlgebraicMatrix(theParameters.num_row(), theParameters.num_row(), 1);
      allJacobians.push_back(fullJacobian);
      theTsosVec.push_back(refTsos);
      const JacobianLocalToCurvilinear startTrafo(hitPtr->det()->surface(), refTsos.localParameters(), *magField);
      const AlgebraicMatrix localToCurvilinear =  asHepMatrix<5>(startTrafo.jacobian());
      allLocalToCurv.push_back(localToCurvilinear);
      allSteps.push_back(0.);
      allCurvlinJacobians.push_back(fullJacobian);
         
    } else {
      AlgebraicMatrix nextJacobian;
      AlgebraicMatrix nextCurvlinJacobian;
      double nextStep = 0.;
      TrajectoryStateOnSurface nextTsos;
      if (!this->propagate(previousHitPtr->det()->surface(), previousTsos,
			   hitPtr->det()->surface(), nextTsos,
			   nextJacobian, nextCurvlinJacobian, nextStep, propDir, magField)) {
	return false; // stop if problem...// no delete aMaterialEffectsUpdator needed
      }

      allJacobians.push_back(nextJacobian);
      fullJacobian = nextJacobian * previousChangeInCurvature * fullJacobian;
      theTsosVec.push_back(nextTsos);
      
      const JacobianLocalToCurvilinear startTrafo(hitPtr->det()->surface(), nextTsos.localParameters(), *magField);
      const AlgebraicMatrix localToCurvilinear =  asHepMatrix<5>(startTrafo.jacobian());
      allLocalToCurv.push_back(localToCurvilinear);
      if (nextStep == 0.) { 
	 // brokenLinesFine will not work, brokenLinesCoarse combines close by layers
	 if (materialEffects == brokenLinesFine) return false;
      }
      allSteps.push_back(nextStep);
      allCurvlinJacobians.push_back(nextCurvlinJacobian);
    }
    // take material effects into account. since trajectory-state is constructed with errors equal zero,
    // the updated state contains only the uncertainties due to interactions in the current layer.
    const TrajectoryStateOnSurface tmpTsos(theTsosVec.back().localParameters(), zeroErrors,
					   theTsosVec.back().surface(), magField, surfaceSide);
    const TrajectoryStateOnSurface updatedTsos = aMaterialEffectsUpdator->updateState(tmpTsos, propDir);

    if ( !updatedTsos.isValid() ) return false;// no delete aMaterialEffectsUpdator needed

    if ( theTsosVec.back().localParameters().charge() )
    {
      previousChangeInCurvature[0][0] = updatedTsos.localParameters().signedInverseMomentum() 
	/ theTsosVec.back().localParameters().signedInverseMomentum();
    }
    
    // get multiple-scattering covariance-matrix
    allDeltaParameterCovs.push_back( asHepMatrix<5>(updatedTsos.localError().matrix()) );
    allCurvatureChanges.push_back(previousChangeInCurvature);
    
    // projection-matrix tsos-parameters -> measurement-coordinates
    if ( useRecHit( hitPtr ) ) { allProjections.push_back(hitPtr->projectionMatrix()); }
    else                       { allProjections.push_back( AlgebraicMatrix( 2, 5, 0) ); } // get size from ???
    
    // set start-parameters for next propagation. trajectory-state without error
    //  - no error propagation needed here.
    previousHitPtr = hitPtr;
    previousTsos   = TrajectoryStateOnSurface(updatedTsos.globalParameters(),
					      updatedTsos.surface(), surfaceSide);
    if (materialEffects<brokenLinesCoarse)
    this->fillDerivatives(allProjections.back(), fullJacobian, iRow);

    AlgebraicVector mixedLocalParams = asHepVector<5>(theTsosVec.back().localParameters().mixedFormatVector());
    this->fillTrajectoryPositions(allProjections.back(), mixedLocalParams, iRow);
    if ( useRecHit( hitPtr ) ) this->fillMeasurementAndError(hitPtr, iRow, updatedTsos);

    iRow += nMeasPerHit;
  } // end of loop on hits

  switch (materialEffects) {
  case none: break;
  case multipleScattering:
  case energyLoss:
  case combined:
    this->addMaterialEffectsCov(allJacobians, allProjections, allCurvatureChanges, allDeltaParameterCovs); break;
  case breakPoints:
    this->addMaterialEffectsBp (allJacobians, allProjections, allCurvatureChanges, allDeltaParameterCovs, allLocalToCurv); break;
  case brokenLinesCoarse:
    this->addMaterialEffectsBrl(allProjections, allDeltaParameterCovs, allLocalToCurv, allSteps); break;
  case brokenLinesFine:
    this->addMaterialEffectsBrl(allCurvlinJacobians, allProjections, allCurvatureChanges, allDeltaParameterCovs, allLocalToCurv, allSteps);
  }
 
  if (refTsos.hasError()) {
    AlgebraicSymMatrix parameterCov = asHepMatrix<5>(refTsos.localError().matrix());
    AlgebraicMatrix  parDeriv;
    if (theNumberOfMsPars>0)
    { parDeriv = theDerivatives.sub( 1, nMeasPerHit*allJacobians.size(), 1, theParameters.num_row() ); 
    } else { parDeriv = theDerivatives; }
    theTrajectoryPositionCov = parameterCov.similarity(parDeriv);
  } else {
    theTrajectoryPositionCov = AlgebraicSymMatrix(theDerivatives.num_row(), 1);
  }

  // delete aMaterialEffectsUpdator; // not needed since auto_ptr

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
  case breakPoints:
    return new CombinedMaterialEffectsUpdator(mass);
  case brokenLinesCoarse:
  case brokenLinesFine:
    return new CombinedMaterialEffectsUpdator(mass);
}

  return 0;
}

//__________________________________________________________________________________

bool ReferenceTrajectory::propagate(const BoundPlane &previousSurface, const TrajectoryStateOnSurface &previousTsos,
				    const BoundPlane &newSurface, TrajectoryStateOnSurface &newTsos, AlgebraicMatrix &newJacobian, 
				    AlgebraicMatrix &newCurvlinJacobian, double &nextStep,
				    const PropagationDirection propDir, const MagneticField *magField) const
{
  // propagate to next layer
  AnalyticalPropagator aPropagator(magField, propDir);
  const std::pair<TrajectoryStateOnSurface, double> tsosWithPath =
    aPropagator.propagateWithPath(previousTsos, newSurface);

  // stop if propagation wasn't successful
  if (!tsosWithPath.first.isValid())  return false;

  nextStep = tsosWithPath.second;
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
  newCurvlinJacobian = curvilinearJacobian;
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
  // GF 10/2008: I doubt that it makes sense to update the hit with the tsos here:
  //             That is an analytical extrapolation and not the best guess of the real 
  //             track state on the module, but the latter should be better to get the best
  //             hit uncertainty estimate!
  TransientTrackingRecHit::ConstRecHitPointer newHitPtr(hitPtr->canImproveWithTrack() ?
							hitPtr->clone(updatedTsos) : hitPtr);

  const LocalPoint localMeasurement    = newHitPtr->localPosition();
  const LocalError localMeasurementCov = newHitPtr->localPositionError();

  theMeasurements[iRow]   = localMeasurement.x();
  theMeasurements[iRow+1] = localMeasurement.y();
  theMeasurementsCov[iRow][iRow]     = localMeasurementCov.xx();
  theMeasurementsCov[iRow][iRow+1]   = localMeasurementCov.xy();
  theMeasurementsCov[iRow+1][iRow+1] = localMeasurementCov.yy();
  // GF: Should be a loop once the hit dimension is not hardcoded as nMeasPerHit (to be checked):
  // for (int i = 0; i < hitPtr->dimension(); ++i) {
  //   theMeasurements[iRow+i]   = hitPtr->parameters()[i]; // fixme: parameters() is by value!
  //   for (int j = i; j < hitPtr->dimension(); ++j) {
  //     theMeasurementsCov[iRow+i][iRow+j] = hitPtr->parametersError()[i][j];
  //   }
  // }
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
    // GF: Should be a loop once the hit dimension is not hardcoded as nMeasPerHit (to be checked):
    // for (int j = 0; j < projection.num_col(); ++j) {
    //   theDerivatives[iRow+j][i] = projectedJacobian[j][i];
    // }
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
  // GF: Should be a loop once the hit dimension is not hardcoded as nMeasPerHit (to be checked):
  // for (int j = 0; j < projection.num_col(); ++j) {
  //   theTrajectoryPositions[iRow+j] = localPosition[j];
  // }
}

//__________________________________________________________________________________

void ReferenceTrajectory::addMaterialEffectsCov(const std::vector<AlgebraicMatrix> &allJacobians, 
						const std::vector<AlgebraicMatrix> &allProjections,
						const std::vector<AlgebraicSymMatrix> &allCurvatureChanges,
						const std::vector<AlgebraicSymMatrix> &allDeltaParameterCovs)
{
  // the uncertainty due to multiple scattering is 'transferred' to the error matrix of the hits

  // GF: Needs update once hit dimension is not hardcoded as nMeasPerHit!

  AlgebraicSymMatrix materialEffectsCov(nMeasPerHit * allJacobians.size(), 0);

  // additional covariance-matrix of the measurements due to material-effects (single measurement)
  AlgebraicSymMatrix deltaMaterialEffectsCov;

  // additional covariance-matrix of the parameters due to material-effects
  AlgebraicSymMatrix paramMaterialEffectsCov(allDeltaParameterCovs[0]); //initialization
  // error-propagation to state after energy loss
  //GFback  paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allCurvatureChanges[0]);

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

    // GFback add uncertainties for the following layers due to scattering at this layer
    paramMaterialEffectsCov += allDeltaParameterCovs[k];
    // end GFback
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
    // add uncertainties for the following layers due to scattering at this layer
    // GFback paramMaterialEffectsCov += allDeltaParameterCovs[k];    
    // error-propagation to state after energy loss
    paramMaterialEffectsCov = paramMaterialEffectsCov.similarity(allCurvatureChanges[k]);
     
  }
  theMeasurementsCov += materialEffectsCov;
}

//__________________________________________________________________________________

void ReferenceTrajectory::addMaterialEffectsBp(const std::vector<AlgebraicMatrix> &allJacobians, 
						const std::vector<AlgebraicMatrix> &allProjections,
						const std::vector<AlgebraicSymMatrix> &allCurvatureChanges,
						const std::vector<AlgebraicSymMatrix> &allDeltaParameterCovs,
						const std::vector<AlgebraicMatrix> &allLocalToCurv)
{
//CHK: add material effects using break points
  int offsetPar = theNumberOfPars; 
  int offsetMeas = nMeasPerHit * allJacobians.size();
  int ierr; 

  AlgebraicMatrix tempJacobian;
  AlgebraicMatrix MSprojection(2,5);
  MSprojection[0][1] = 1;
  MSprojection[1][2] = 1;
  AlgebraicSymMatrix tempMSCov; 
  AlgebraicSymMatrix tempMSCovProj;
  AlgebraicMatrix tempMSJacProj;   

  for (unsigned int k = 1; k < allJacobians.size(); ++k) {
// CHK 
    int kbp = k-1;
    tempJacobian = allJacobians[k] * allCurvatureChanges[k];
    tempMSCov = allDeltaParameterCovs[k-1].similarity(allLocalToCurv[k-1]);
    tempMSCovProj = tempMSCov.similarity(MSprojection);
    theMeasurementsCov[offsetMeas+nMeasPerHit*kbp  ][offsetMeas+nMeasPerHit*kbp  ] = tempMSCovProj[0][0];
    theMeasurementsCov[offsetMeas+nMeasPerHit*kbp+1][offsetMeas+nMeasPerHit*kbp+1]=  tempMSCovProj[1][1];
    theDerivatives[offsetMeas+nMeasPerHit*kbp  ][offsetPar+2*kbp  ] = 1.0;
    theDerivatives[offsetMeas+nMeasPerHit*kbp+1][offsetPar+2*kbp+1] = 1.0 ; 

    tempMSJacProj = (allProjections[k] * ( tempJacobian * allLocalToCurv[k-1].inverse(ierr) )) * MSprojection.T();
    if (ierr) {
      edm::LogError("Alignment") << "@SUB=ReferenceTrajectory::addMaterialEffectsBp"
	                         << "Inversion 1 for break points failed: " << ierr; }
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*kbp  ] =  tempMSJacProj[0][0];  
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*kbp+1] =  tempMSJacProj[0][1]; 
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*kbp  ] =  tempMSJacProj[1][0];  
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*kbp+1] =  tempMSJacProj[1][1];
              
    for (unsigned int l = k+1; l < allJacobians.size(); ++l) {
// CHK    
      int kbp = k-1;
      tempJacobian = allJacobians[l] * allCurvatureChanges[l] * tempJacobian; 
      tempMSJacProj = (allProjections[l] * ( tempJacobian * allLocalToCurv[k-1].inverse(ierr) )) * MSprojection.T();
      if (ierr) {
	edm::LogError("Alignment") << "@SUB=ReferenceTrajectory::addMaterialEffectsBp"
				   << "Inversion 2 for break points failed: " << ierr; }
      theDerivatives[nMeasPerHit*l  ][offsetPar+2*kbp  ] =  tempMSJacProj[0][0];  
      theDerivatives[nMeasPerHit*l  ][offsetPar+2*kbp+1] =  tempMSJacProj[0][1]; 
      theDerivatives[nMeasPerHit*l+1][offsetPar+2*kbp  ] =  tempMSJacProj[1][0];  
      theDerivatives[nMeasPerHit*l+1][offsetPar+2*kbp+1] =  tempMSJacProj[1][1]; 

    }

  }
  
}

//__________________________________________________________________________________

void ReferenceTrajectory::addMaterialEffectsBrl(const std::vector<AlgebraicMatrix> &allCurvlinJacobians, 
						const std::vector<AlgebraicMatrix> &allProjections,
						const std::vector<AlgebraicSymMatrix> &allCurvatureChanges,
						const std::vector<AlgebraicSymMatrix> &allDeltaParameterCovs,
						const std::vector<AlgebraicMatrix> &allLocalToCurv,
						const std::vector<double> &allSteps)
{
//CHK: add material effects using broken lines
//BrokenLinesFine: use exact Jacobians
  
  int offsetPar = theNumberOfPars; 
  int offsetMeas = nMeasPerHit*allCurvlinJacobians.size();
  int ierr; 
  double cosLambda = sqrt((theGlobalPars[3]*theGlobalPars[3]+theGlobalPars[4]*theGlobalPars[4])/
  (theGlobalPars[3]*theGlobalPars[3]+theGlobalPars[4]*theGlobalPars[4]+theGlobalPars[5]*theGlobalPars[5]));
    
  AlgebraicMatrix tempJacobian(allCurvatureChanges[0]);
  AlgebraicMatrix MSprojAngle(2,5);
  MSprojAngle[0][1] = 1;
  MSprojAngle[1][2] = cosLambda;
  AlgebraicMatrix MSprojOffset(2,5);
  MSprojOffset[0][3] = 1;
  MSprojOffset[1][4] = 1;

  AlgebraicSymMatrix tempMSCov; 
  AlgebraicSymMatrix tempMSCovProj;
  AlgebraicMatrix tempMSJacProj; 
  AlgebraicMatrix tempJacL; 
  AlgebraicMatrix tempJacOffsetL; 
  AlgebraicMatrix tempJacN; 
  AlgebraicMatrix tempJacOffsetN; 
  AlgebraicMatrix tempJacBending;
           
// measurements from hits  
  for (unsigned int k = 0; k < allCurvlinJacobians.size(); ++k) {
    tempMSJacProj = (allProjections[k] * allLocalToCurv[k].inverse(ierr) ) * MSprojOffset.T();  
    if (ierr) {
       edm::LogError("Alignment") << "@SUB=ReferenceTrajectory::addMaterialEffectsBrl"
				   << "Inversion 1 for broken lines failed: " << ierr; }
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*k  ] =  tempMSJacProj[0][0];  
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*k+1] =  tempMSJacProj[0][1]; 
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*k  ] =  tempMSJacProj[1][0];  
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*k+1] =  tempMSJacProj[1][1]; 
  }

// measurement of MS kink  
  for (unsigned int k = 1; k < allCurvlinJacobians.size()-1; ++k) {
// CHK 
    int iMsMeas = k-1; 
    int l    = k-1; // last hit
    int n    = k+1; // next hit

// amount of multiple scattering in layer k  (angular error perp to direction)  
    tempMSCov = allDeltaParameterCovs[k].similarity(allLocalToCurv[k]);
    tempMSCovProj = tempMSCov.similarity(MSprojAngle);
    theMeasurementsCov[offsetMeas+nMeasPerHit*iMsMeas  ][offsetMeas+nMeasPerHit*iMsMeas  ] = tempMSCovProj[1][1];
    theMeasurementsCov[offsetMeas+nMeasPerHit*iMsMeas+1][offsetMeas+nMeasPerHit*iMsMeas+1]=  tempMSCovProj[0][0];
// broken line measurements for layer k
    double deltaK (1.0/allSteps[k]);    
    double deltaN (1.0/allSteps[n]);    
    // transformation matices for offsets ( l -> k <- n )
    tempJacL = allCurvlinJacobians[k] * tempJacobian;
    tempJacOffsetL = tempJacL.sub(4,5,4,5);
    tempJacobian = tempJacobian * allCurvatureChanges[k];
    tempJacN = allCurvlinJacobians[n] * tempJacobian;
    tempJacOffsetN = tempJacN.sub(4,5,4,5).inverse(ierr); // .T() should do
    if (ierr) {
       edm::LogError("Alignment") << "@SUB=ReferenceTrajectory::addMaterialEffectsBrl"
				   << "Inversion 2 for broken lines failed: " << ierr; }
    tempJacBending = tempJacL.sub(2,3,1,1) + tempJacN.sub(2,3,1,1);
    // bending    
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][              0] = tempJacBending[1][0]/cosLambda;
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][              0] = tempJacBending[0][0];
    // last layer
    tempJacOffsetL *= deltaK;
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*l  ] = tempJacOffsetL[0][0];    
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*l+1] = tempJacOffsetL[0][1];    
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*l  ] = tempJacOffsetL[1][0];    
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*l+1] = tempJacOffsetL[1][1];        
    // current layer
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*k  ] = -(deltaK + deltaN);
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*k+1] = -(deltaK + deltaN);   
    // next layer
    tempJacOffsetN *= deltaN;
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*n  ] = tempJacOffsetN[0][0];
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*n+1] = tempJacOffsetN[0][1];
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*n  ] = tempJacOffsetN[1][0];
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*n+1] = tempJacOffsetN[1][1];   
        
  }
  
}

void ReferenceTrajectory::addMaterialEffectsBrl(const std::vector<AlgebraicMatrix> &allProjections,
						const std::vector<AlgebraicSymMatrix> &allDeltaParameterCovs,
						const std::vector<AlgebraicMatrix> &allLocalToCurv,
						const std::vector<double> &allSteps,
						const double minStep)
{
//CHK: add material effects using broken lines
//BrokenLinesCoarse: combine close by detectors, use aprroximate Jacobians from Steps, bending only in RPhi

  int offsetPar = theNumberOfPars; 
  int offsetMeas = nMeasPerHit*allSteps.size();
  int ierr; 
  double cosLambda = sqrt((theGlobalPars[3]*theGlobalPars[3]+theGlobalPars[4]*theGlobalPars[4])/
  (theGlobalPars[3]*theGlobalPars[3]+theGlobalPars[4]*theGlobalPars[4]+theGlobalPars[5]*theGlobalPars[5]));
      
  AlgebraicMatrix MSprojAngle(2,5);
  MSprojAngle[0][1] = 1;
  MSprojAngle[1][2] = cosLambda;
  AlgebraicMatrix MSprojOffset(2,5);
  MSprojOffset[0][3] = 1;
  MSprojOffset[1][4] = 1;

  AlgebraicSymMatrix tempMSCov; 
  AlgebraicSymMatrix tempMSCovProj;
  AlgebraicMatrix tempMSJacProj; 
  
  // combine closeby detectors into single plane
  std::vector<unsigned int> first(allSteps.size());         
  std::vector<unsigned int> last (allSteps.size());
  std::vector<unsigned int> plane(allSteps.size());
  std::vector<double> sPlane(allSteps.size());
  unsigned int nPlane = 0;
  double sTot = 0;
  
  for (unsigned int k = 0; k < allSteps.size(); ++k) {
    sTot += allSteps[k];
    if (fabs(allSteps[k])>minStep) { nPlane += 1; first[nPlane] = k; }
    last[nPlane] = k;
    plane[k] = nPlane;
    sPlane[nPlane] += sTot;
  }
  for (unsigned int k = 0; k <= nPlane; ++k) { sPlane[k] /= (double) (last[k]-first[k]+1); }

  // measurements from hits  
  for (unsigned int k = 0; k < allSteps.size(); ++k) {
    tempMSJacProj = (allProjections[k] * allLocalToCurv[k].inverse(ierr) ) * MSprojOffset.T(); 
    if (ierr) {
       edm::LogError("Alignment") << "@SUB=ReferenceTrajectory::addMaterialEffectsBrl"
				   << "Inversion 3 for broken lines failed: " << ierr; }   
    unsigned int iPlane = plane[k]; 
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*iPlane  ] =  tempMSJacProj[0][0];  
    theDerivatives[nMeasPerHit*k  ][offsetPar+2*iPlane+1] =  tempMSJacProj[0][1]; 
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*iPlane  ] =  tempMSJacProj[1][0];  
    theDerivatives[nMeasPerHit*k+1][offsetPar+2*iPlane+1] =  tempMSJacProj[1][1]; 
  }

// measurement of MS kink 
  for (unsigned int i = 1; i < nPlane; ++i) {
// CHK 
    int iMsMeas = i-1; 
    int l    = i-1; // last hit
    int n    = i+1; // next hit

// amount of multiple scattering in plane k 
    for (unsigned int k = first[i]; k <= last[i]; ++k) {   
      tempMSCov = allDeltaParameterCovs[k].similarity(allLocalToCurv[k]);
      tempMSCovProj = tempMSCov.similarity(MSprojAngle);
      theMeasurementsCov[offsetMeas+nMeasPerHit*iMsMeas  ][offsetMeas+nMeasPerHit*iMsMeas  ] += tempMSCovProj[0][0];
      theMeasurementsCov[offsetMeas+nMeasPerHit*iMsMeas+1][offsetMeas+nMeasPerHit*iMsMeas+1] += tempMSCovProj[1][1];
    }
// broken line measurements for layer k, correlations between both planes neglected
    double stepK = sPlane[i] - sPlane[l];
    double stepN = sPlane[n] - sPlane[i];
    double deltaK (1.0/stepK);    
    double deltaN (1.0/stepN);    
    // bending (only in RPhi)  
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][              0] = stepK+stepN;
    // last layer
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*l  ] = deltaK;    
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*l+1] = deltaK;        
    // current layer
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*i  ] = -(deltaK + deltaN);
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*i+1] = -(deltaK + deltaN);   
    // next layer
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas  ][offsetPar+2*n  ] = deltaN;
    theDerivatives[offsetMeas+nMeasPerHit*iMsMeas+1][offsetPar+2*n+1] = deltaN;   
        
  }
  
}
