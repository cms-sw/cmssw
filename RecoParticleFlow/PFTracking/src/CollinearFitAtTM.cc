#include "RecoParticleFlow/PFTracking/interface/CollinearFitAtTM.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

CollinearFitAtTM::CollinearFitAtTM ()
{
  //
  // Jacobian
  //
  for ( int i=0; i<12; ++i ) {
    for ( int j=0; j<6; ++j )  jacobian_(i,j) = 0;
  }
  for ( int i=1; i<5; ++i ) {
    jacobian_(i,ParQpOut+i) = jacobian_(i+5,ParQpOut+i) = 1;
  }
  jacobian_(0,ParQpIn) = 1.;
  jacobian_(5,ParQpOut) = 1.;
}

bool
CollinearFitAtTM::fit (const TrajectoryMeasurement& tm,
		       ResultVector& parameters,
		       ResultMatrix& covariance,
		       double& chi2)
{
  //
  // check input
  //
  if ( !tm.forwardPredictedState().isValid() ||
       !tm.backwardPredictedState().isValid() ||
       !tm.updatedState().isValid() ) {
    edm::LogWarning("CollinearFitAtTM") << "Invalid state in TrajectoryMeasurement";
    return false;
  }
  //
  // prepare fit
  //
  AlgebraicVector5 fwdPar = tm.forwardPredictedState().localParameters().vector();
  AlgebraicSymMatrix55 fwdCov = tm.forwardPredictedState().localError().matrix();
  AlgebraicVector5 bwdPar = tm.backwardPredictedState().localParameters().vector();
  AlgebraicSymMatrix55 bwdCov = tm.backwardPredictedState().localError().matrix();

  LocalPoint hitPos(0.,0.,0.);
  LocalError hitErr(-1.,-1.,-1.);
  if ( tm.recHit()->isValid() ) {
    hitPos = tm.recHit()->localPosition();
    hitErr = tm.recHit()->localPositionError();
  }

  return fit(fwdPar,fwdCov,bwdPar,bwdCov,
	     hitPos,hitErr,
	     parameters,covariance,chi2);
}

bool
CollinearFitAtTM::fit (const AlgebraicVector5& fwdParameters, 
		       const AlgebraicSymMatrix55& fwdCovariance,
		       const AlgebraicVector5& bwdParameters, 
		       const AlgebraicSymMatrix55& bwdCovariance,
		       const LocalPoint& hitPos, const LocalError& hitErr,
		       ResultVector& parameters, ResultMatrix& covariance,
		       double& chi2)
{

  if ( hitErr.xx()>0 )
    jacobian_(10,ParX) = jacobian_(11,ParY) = 1;
  else
    jacobian_(10,ParX) = jacobian_(11,ParY) = 0;

  for ( int i=0; i<12; ++i ) {
    for ( int j=0; j<12; ++j )  weightMatrix_(i,j) = 0;
  }

  for ( int i=0; i<5; ++i )  measurements_(i) = fwdParameters(i);
  weightMatrix_.Place_at(fwdCovariance,0,0);
  for ( int i=0; i<5; ++i )  measurements_(i+5) = bwdParameters(i);
  weightMatrix_.Place_at(bwdCovariance,5,5);
  if ( hitErr.xx()>0 ) {
    measurements_(10) = hitPos.x();
    measurements_(11) = hitPos.y();
    weightMatrix_(10,10) = hitErr.xx();
    weightMatrix_(10,11) = weightMatrix_(11,10) = hitErr.xy();
    weightMatrix_(11,11) = hitErr.yy();
  }
  else {
    measurements_(10) = measurements_(11) = 0.;
    weightMatrix_(10,10) = weightMatrix_(11,11) = 1.;
    weightMatrix_(10,11) = weightMatrix_(11,10) = 0.;
  }
  //
  // invert covariance matrix
  //
  if ( !weightMatrix_.Invert() ) {
    edm::LogWarning("CollinearFitAtTM") << "Inversion of input covariance matrix failed";
    return false;
  }

  //
  projectedMeasurements_ = ROOT::Math::Transpose(jacobian_)*(weightMatrix_*measurements_);
  //
  // Fitted parameters and covariance matrix
  // 
  covariance = ROOT::Math::SimilarityT(jacobian_,weightMatrix_);
  if ( !covariance.Invert() ) {
    edm::LogWarning("CollinearFitAtTM") << "Inversion of resulting weight matrix failed";
    return false;
  }

  parameters = covariance*projectedMeasurements_;

  //
  // chi2
  //
  chi2 = ROOT::Math::Similarity(measurements_,weightMatrix_) -
    ROOT::Math::Similarity(projectedMeasurements_,covariance);

  return true;
}
