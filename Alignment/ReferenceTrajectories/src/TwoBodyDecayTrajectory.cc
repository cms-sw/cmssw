#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h" 
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "Alignment/ReferenceTrajectories/interface/TwoBodyDecayTrajectory.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h" 
#include "DataFormats/Math/interface/Error.h" 
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayParameters.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

TwoBodyDecayTrajectory::TwoBodyDecayTrajectory(const TwoBodyDecayTrajectoryState& tsos,
                                               const ConstRecHitCollection& recHits,
                                               const MagneticField* magField,
                                               const reco::BeamSpot& beamSpot,
                                               const ReferenceTrajectoryBase::Config& config) :
  ReferenceTrajectoryBase(
     TwoBodyDecayParameters::dimension, recHits.first.size() + recHits.second.size(),
     (config.materialEffects >= breakPoints) ? 2*(recHits.first.size() + recHits.second.size())-4 : 0,
     (config.materialEffects >= breakPoints) ? 2*(recHits.first.size() + recHits.second.size())-3 : 1 ),
  materialEffects_(config.materialEffects),
  propDir_(config.propDir),
  useRefittedState_(config.useRefittedState),
  constructTsosWithErrors_(config.constructTsosWithErrors)

{
  if ( config.hitsAreReverse )
  {
    TransientTrackingRecHit::ConstRecHitContainer::const_reverse_iterator itRecHits;
    ConstRecHitCollection fwdRecHits;

    fwdRecHits.first.reserve( recHits.first.size() );
    for ( itRecHits = recHits.first.rbegin(); itRecHits != recHits.first.rend(); ++itRecHits )
    {
      fwdRecHits.first.push_back( *itRecHits );
    }

    fwdRecHits.second.reserve( recHits.second.size() );
    for ( itRecHits = recHits.second.rbegin(); itRecHits != recHits.second.rend(); ++itRecHits )
    {
      fwdRecHits.second.push_back( *itRecHits );
    }

    theValidityFlag = this->construct(tsos, fwdRecHits, magField, beamSpot);
  }
  else
  {
    theValidityFlag = this->construct(tsos, recHits, magField, beamSpot);
  }
}


TwoBodyDecayTrajectory::TwoBodyDecayTrajectory( void )
  : ReferenceTrajectoryBase( 0, 0, 0, 0),
  materialEffects_(none),
  propDir_(anyDirection),
  useRefittedState_(false),
  constructTsosWithErrors_(false)
{}


bool TwoBodyDecayTrajectory::construct(const TwoBodyDecayTrajectoryState& state,
                                       const ConstRecHitCollection& recHits,
                                       const MagneticField* field,
                                       const reco::BeamSpot& beamSpot)
{  
  const TwoBodyDecayTrajectoryState::TsosContainer& tsos = state.trajectoryStates(useRefittedState_);
  const TwoBodyDecayTrajectoryState::Derivatives& deriv = state.derivatives();
  double mass = state.particleMass();

  //
  // first track
  //

  // construct a trajectory (hits should be already in correct order)
  ReferenceTrajectoryBase::Config config(materialEffects_, propDir_, mass);
  config.useBeamSpot = false;
  config.hitsAreReverse = false;

  ReferenceTrajectory trajectory1(tsos.first, recHits.first, field, beamSpot, config);

  // check if construction of trajectory was successful
  if ( !trajectory1.isValid() ) return false;

  //
  // second track
  //

  ReferenceTrajectory trajectory2(tsos.second, recHits.second, field, beamSpot, config);

  if ( !trajectory2.isValid() ) return false;
  
  //
  // combine both tracks
  //
  unsigned int nLocal = deriv.first.num_row();
  unsigned int nTbd   = deriv.first.num_col();

  if (materialEffects_ >= localGBL) {
    // GBL trajectory inputs
    // convert to TMatrix
    TMatrixD tbdToLocal1(nLocal, nTbd);
    for (unsigned int row = 0; row < nLocal; ++row) {
      for (unsigned int col = 0; col < nTbd; ++col) {
        tbdToLocal1(row,col) = deriv.first[row][col];
      }
    }
    // add first body
    theGblInput.push_back(std::make_pair(trajectory1.gblInput().front().first, 
                                         trajectory1.gblInput().front().second*tbdToLocal1));
    // convert to TMatrix
    TMatrixD tbdToLocal2(nLocal, nTbd);
    for (unsigned int row = 0; row < nLocal; ++row) {
      for (unsigned int col = 0; col < nTbd; ++col) {
        tbdToLocal2(row,col) = deriv.second[row][col];
      }
    }
    // add second body
    theGblInput.push_back(std::make_pair(trajectory2.gblInput().front().first, 
                                         trajectory2.gblInput().front().second*tbdToLocal2));
    // add virtual mass measurement
    theGblExtDerivatives.ResizeTo(1,nTbd);
    theGblExtDerivatives(0,TwoBodyDecayParameters::mass) = 1.0;
    theGblExtMeasurements.ResizeTo(1);
    theGblExtMeasurements(0) = state.primaryMass() - state.decayParameters()[TwoBodyDecayParameters::mass];
    theGblExtPrecisions.ResizeTo(1);
    theGblExtPrecisions(0) = 1.0 / (state.primaryWidth() * state.primaryWidth());
    // nominal field
    theNomField = trajectory1.nominalField();
  } else {
    unsigned int nHitMeas1     = trajectory1.numberOfHitMeas();
    unsigned int nVirtualMeas1 = trajectory1.numberOfVirtualMeas(); 
    unsigned int nPar1         = trajectory1.numberOfPar();
    unsigned int nVirtualPar1  = trajectory1.numberOfVirtualPar(); 
     
    // derivatives of the trajectory w.r.t. to the decay parameters
    AlgebraicMatrix fullDeriv1 = trajectory1.derivatives().sub(1,nHitMeas1+nVirtualMeas1,1,nLocal) * trajectory1.localToTrajectory() * deriv.first;

    unsigned int nHitMeas2     = trajectory2.numberOfHitMeas();
    unsigned int nVirtualMeas2 = trajectory2.numberOfVirtualMeas();  
    unsigned int nPar2         = trajectory2.numberOfPar();
    unsigned int nVirtualPar2  = trajectory2.numberOfVirtualPar();

    AlgebraicMatrix fullDeriv2 = trajectory2.derivatives().sub(1,nHitMeas2+nVirtualMeas2,1,nLocal) * trajectory2.localToTrajectory() * deriv.second;

    theNumberOfRecHits.first = recHits.first.size();
    theNumberOfRecHits.second = recHits.second.size();

    theNumberOfHits = trajectory1.numberOfHits() + trajectory2.numberOfHits(); 
    theNumberOfPars = nPar1 + nPar2;
    theNumberOfVirtualPars = nVirtualPar1 + nVirtualPar2;
    theNumberOfVirtualMeas = nVirtualMeas1 + nVirtualMeas2 + 1; // add virtual mass measurement
  
    // hit measurements from trajectory 1
    int rowOffset = 1;
    int colOffset = 1; 
    theDerivatives.sub( rowOffset, colOffset,                fullDeriv1.sub(            1, nHitMeas1,                     1, nTbd ) );
    colOffset += nTbd;
    theDerivatives.sub( rowOffset, colOffset, trajectory1.derivatives().sub(            1, nHitMeas1,            nLocal + 1, nPar1 + nVirtualPar1 ) );
    // hit measurements from trajectory 2
    rowOffset += nHitMeas1;
    colOffset = 1; 
    theDerivatives.sub( rowOffset, colOffset,                fullDeriv2.sub(            1, nHitMeas2,                     1, nTbd ) );
    colOffset += (nPar1 + nVirtualPar1 + nTbd - nLocal);
    theDerivatives.sub( rowOffset, colOffset, trajectory2.derivatives().sub(            1, nHitMeas2,            nLocal + 1, nPar2 + nVirtualPar2 ) );  
    // MS measurements from trajectory 1
    rowOffset += nHitMeas2;  
    colOffset = 1; 
    theDerivatives.sub( rowOffset, colOffset,                fullDeriv1.sub(nHitMeas1 + 1, nHitMeas1 + nVirtualMeas1,          1, nTbd ) );  
    colOffset += nTbd;
    theDerivatives.sub( rowOffset, colOffset, trajectory1.derivatives().sub(nHitMeas1 + 1, nHitMeas1 + nVirtualMeas1, nLocal + 1, nPar1 + nVirtualPar1 ) ); 
    // MS measurements from trajectory 2
    rowOffset += nVirtualMeas1;  
    colOffset = 1; 
    theDerivatives.sub( rowOffset, colOffset,                fullDeriv2.sub(nHitMeas2 + 1, nHitMeas2 + nVirtualMeas2,          1, nTbd ) );
    colOffset += (nPar1 + nVirtualPar1 + nTbd - nLocal);  
    theDerivatives.sub( rowOffset, colOffset, trajectory2.derivatives().sub(nHitMeas2 + 1, nHitMeas2 + nVirtualMeas2, nLocal + 1, nPar2 + nVirtualPar2 ) ); 
        
    theMeasurements.sub(                                         1, trajectory1.measurements().sub(            1, nHitMeas1 ) );
    theMeasurements.sub( nHitMeas1                             + 1, trajectory2.measurements().sub(            1, nHitMeas2 ) );
    theMeasurements.sub( nHitMeas1 + nHitMeas2                 + 1, trajectory1.measurements().sub(nHitMeas1 + 1, nHitMeas1 + nVirtualMeas1 ) );
    theMeasurements.sub( nHitMeas1 + nHitMeas2 + nVirtualMeas1 + 1, trajectory2.measurements().sub(nHitMeas2 + 1, nHitMeas2 + nVirtualMeas2 ) );

    theMeasurementsCov.sub(                                         1, trajectory1.measurementErrors().sub(            1, nHitMeas1 ) );
    theMeasurementsCov.sub( nHitMeas1                             + 1, trajectory2.measurementErrors().sub(            1, nHitMeas2 ) );
    theMeasurementsCov.sub( nHitMeas1 + nHitMeas2                 + 1, trajectory1.measurementErrors().sub(nHitMeas1 + 1, nHitMeas1 + nVirtualMeas1 ) );
    theMeasurementsCov.sub( nHitMeas1 + nHitMeas2 + nVirtualMeas1 + 1, trajectory2.measurementErrors().sub(nHitMeas2 + 1, nHitMeas2 + nVirtualMeas2 ) );

    theTrajectoryPositions.sub(             1, trajectory1.trajectoryPositions() );
    theTrajectoryPositions.sub( nHitMeas1 + 1, trajectory2.trajectoryPositions() );

    theTrajectoryPositionCov = state.decayParameters().covariance().similarity( theDerivatives.sub(1, nHitMeas1 + nHitMeas2, 1, 9) );

    theParameters = state.decayParameters().parameters();

    // add virtual mass measurement
    rowOffset += nVirtualMeas2;
    int indMass = rowOffset-1;
    theMeasurements[indMass] = state.primaryMass() - state.decayParameters()[TwoBodyDecayParameters::mass];
    theMeasurementsCov[indMass][indMass] = state.primaryWidth() * state.primaryWidth();
    theDerivatives[indMass][TwoBodyDecayParameters::mass] = 1.0;
  }

  theRecHits.insert( theRecHits.end(), recHits.first.begin(), recHits.first.end() );
  theRecHits.insert( theRecHits.end(), recHits.second.begin(), recHits.second.end() );

  if (constructTsosWithErrors_)
  {
    constructTsosVecWithErrors( trajectory1, trajectory2, field );
  }
  else
  {
    theTsosVec.insert( theTsosVec.end(),
		       trajectory1.trajectoryStates().begin(),
		       trajectory1.trajectoryStates().end() );

    theTsosVec.insert( theTsosVec.end(),
		       trajectory2.trajectoryStates().begin(),
		       trajectory2.trajectoryStates().end() );
  }

  return true;
}


void TwoBodyDecayTrajectory::constructTsosVecWithErrors( const ReferenceTrajectory& traj1,
							 const ReferenceTrajectory& traj2,
							 const MagneticField* field )
{
  int iTsos = 0;

  std::vector< TrajectoryStateOnSurface >::const_iterator itTsos;

  for ( itTsos = traj1.trajectoryStates().begin(); itTsos != traj1.trajectoryStates().end(); itTsos++ )
  {
    constructSingleTsosWithErrors( *itTsos, iTsos, field );
    iTsos++;
  }

  for ( itTsos = traj2.trajectoryStates().begin(); itTsos != traj2.trajectoryStates().end(); itTsos++ )
  {
    constructSingleTsosWithErrors( *itTsos, iTsos, field );
    iTsos++;
  }
}


void TwoBodyDecayTrajectory::constructSingleTsosWithErrors( const TrajectoryStateOnSurface& tsos,
							    int iTsos,
							    const MagneticField* field )
{
  AlgebraicSymMatrix55 localErrors;
  const double coeff = 1e-2;

  double invP = tsos.localParameters().signedInverseMomentum();
  LocalVector p = tsos.localParameters().momentum();

  // rough estimate for the errors of q/p, dx/dz and dy/dz, assuming that
  // sigma(px) = sigma(py) = sigma(pz) = coeff*p.
  float dpinv = coeff*( fabs(p.x()) + fabs(p.y()) + fabs(p.z()) )*invP*invP;
  float dxdir = coeff*( fabs(p.x()) + fabs(p.z()) )/p.z()/p.z();
  float dydir = coeff*( fabs(p.y()) + fabs(p.z()) )/p.z()/p.z();
  localErrors[0][0] = dpinv*dpinv;
  localErrors[1][1] = dxdir*dxdir;
  localErrors[2][2] = dydir*dydir;

  // exact values for the errors on local x and y
  localErrors[3][3] = theTrajectoryPositionCov[nMeasPerHit*iTsos][nMeasPerHit*iTsos];
  localErrors[3][4] = theTrajectoryPositionCov[nMeasPerHit*iTsos][nMeasPerHit*iTsos+1];
  localErrors[4][4] = theTrajectoryPositionCov[nMeasPerHit*iTsos+1][nMeasPerHit*iTsos+1];

  // construct tsos with local errors
  theTsosVec[iTsos] =  TrajectoryStateOnSurface( tsos.localParameters(),
						 LocalTrajectoryError( localErrors ),
						 tsos.surface(),
						 field,
						 tsos.surfaceSide() );
}
