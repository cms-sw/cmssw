//
// -*- C++ -*-
// Package:    PFTracking
// Class:      PFGsfHelper
// 
// Original Author:  Daniele Benedetti 

#include "RecoParticleFlow/PFTracking/interface/PFGsfHelper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "FastSimulation/BaseParticlePropagator/interface/BaseParticlePropagator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
// Add by Daniele
#include "TrackingTools/GsfTools/interface/MultiGaussianStateTransform.h"
#include "TrackingTools/GsfTools/interface/MultiGaussianState1D.h"
#include "TrackingTools/GsfTools/interface/GaussianSumUtilities1D.h"
#include "RecoParticleFlow/PFTracking/interface/CollinearFitAtTM.h"
using namespace std;
using namespace reco;
using namespace edm;


PFGsfHelper::PFGsfHelper(const TrajectoryMeasurement& tm){
  
  /* LogInfo("PFGsfHelper")<<" PFGsfHelper  built"; */

  
  // TrajectoryStateOnSurface theUpdateState = tm.forwardPredictedState();
  theUpdateState = tm.updatedState();
  theForwardState = tm.forwardPredictedState();
  theBackwardState = tm.backwardPredictedState();
  

  Valid = true;
  if ( !theUpdateState.isValid() ||
       !theForwardState.isValid() ||
	!theBackwardState.isValid() )  Valid = false;

  if (Valid){
      
    mode_Px = 0.;
    mode_Py = 0.;
    mode_Pz = 0.;
    std::vector<TrajectoryStateOnSurface> components(theForwardState.components());
    unsigned int numb = components.size();

    std::vector<SingleGaussianState1D> pxStates; pxStates.reserve(numb);
    std::vector<SingleGaussianState1D> pyStates; pyStates.reserve(numb);
    std::vector<SingleGaussianState1D> pzStates; pzStates.reserve(numb);
    for ( std::vector<TrajectoryStateOnSurface>::const_iterator ic=components.begin();
	  ic!=components.end(); ++ic ) {
      GlobalVector momentum(ic->globalMomentum());
      AlgebraicSymMatrix66 cov(ic->cartesianError().matrix());
	pxStates.push_back(SingleGaussianState1D(momentum.x(),cov(3,3),ic->weight()));
	pyStates.push_back(SingleGaussianState1D(momentum.y(),cov(4,4),ic->weight()));
	pzStates.push_back(SingleGaussianState1D(momentum.z(),cov(5,5),ic->weight()));
	//	cout<<"COMP "<<momentum<<endl;
    }
    MultiGaussianState1D pxState(pxStates);
    MultiGaussianState1D pyState(pyStates);
    MultiGaussianState1D pzState(pzStates);
    GaussianSumUtilities1D pxUtils(pxState);
    GaussianSumUtilities1D pyUtils(pyState);
    GaussianSumUtilities1D pzUtils(pzState);
    mode_Px = pxUtils.mode().mean();
    mode_Py = pyUtils.mode().mean();
    mode_Pz = pzUtils.mode().mean();

  
    dp = 0.;
    sigmaDp = 0.;
    
    //
    // prepare input parameter vectors and covariance matrices
    //
 
    AlgebraicVector5 fwdPars = theForwardState.localParameters().vector();
    AlgebraicSymMatrix55 fwdCov = theForwardState.localError().matrix();
    computeQpMode(theForwardState,fwdPars,fwdCov);
    AlgebraicVector5 bwdPars = theBackwardState.localParameters().vector();
    AlgebraicSymMatrix55 bwdCov = theBackwardState.localError().matrix();
    computeQpMode(theBackwardState,bwdPars,bwdCov);
    LocalPoint hitPos(0.,0.,0.);
    LocalError hitErr(-1.,-1.,-1.);
    if ( tm.recHit()->isValid() ) {
      hitPos = tm.recHit()->localPosition();
      hitErr = tm.recHit()->localPositionError();
    }
    
    CollinearFitAtTM collinearFit;
    CollinearFitAtTM::ResultVector fitParameters;
    CollinearFitAtTM::ResultMatrix fitCovariance;
    double fitChi2;
    bool CollFit = true;
    if ( !collinearFit.fit(fwdPars,fwdCov,bwdPars,bwdCov,
			   hitPos,hitErr,
			   fitParameters,fitCovariance,fitChi2) )  CollFit = false;

    if (CollFit){
      double qpIn = fitParameters(0);
      double sig2In = fitCovariance(0,0);
      double qpOut = fitParameters(1);
      double sig2Out = fitCovariance(1,1);
      double corrInOut = fitCovariance(0,1);
      double pIn = 1./fabs(qpIn);
      double pOut = 1./fabs(qpOut);
      double sig2DeltaP = pIn/qpIn*pIn/qpIn*sig2In - 2*pIn/qpIn*pOut/qpOut*corrInOut + 
	pOut/qpOut*pOut/qpOut*sig2Out;
      //   std::cout << "fitted delta p = " << pOut-pIn << " sigma = " 
      // 	    << sqrt(sig2DeltaP) << std::endl;
      dp = pOut - pIn;      
      sigmaDp = sqrt(sig2DeltaP);
  
    }
    
    
  }
}

PFGsfHelper::~PFGsfHelper(){
}
GlobalVector PFGsfHelper::computeP(bool ComputeMode) const {
  GlobalVector gsfp;
  if (ComputeMode) gsfp = GlobalVector(mode_Px,mode_Py,mode_Pz);
  else gsfp = theUpdateState.globalMomentum();
  return gsfp;
}
double PFGsfHelper::fittedDP () const
{  
  return dp;
}
double PFGsfHelper::sigmafittedDP () const
{
  return sigmaDp;
}
bool PFGsfHelper::isValid () const
{
  return Valid;
} 

void PFGsfHelper::computeQpMode (const TrajectoryStateOnSurface tsos,
			 AlgebraicVector5& parameters, AlgebraicSymMatrix55& covariance) const
{
  //
  // parameters and errors from combined state
  //
  parameters = tsos.localParameters().vector();
  covariance = tsos.localError().matrix();
  //
  // mode for parameter 0 (q/p)
  //
  MultiGaussianState1D qpState(MultiGaussianStateTransform::multiState1D(tsos,0));
  GaussianSumUtilities1D qpGS(qpState);
  if ( !qpGS.modeIsValid() )  return;
  double qp = qpGS.mode().mean();
  double varQp = qpGS.mode().variance();
  //
  // replace q/p value and variance, rescale correlation terms
  //   (heuristic procedure - alternative would be mode in 5D ...)
  //
  double VarQpRatio = sqrt(varQp/covariance(0,0));
  parameters(0) = qp;
  covariance(0,0) = varQp;
  for ( int i=1; i<5; ++i )  covariance(i,0) *= VarQpRatio;
//   std::cout << "covariance " << VarQpRatio << " "
// 	    << covariance(1,0) << " " << covariance(0,1) << std::endl;
}


