
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/22 09:57:10 $
 *  $Revision: 1.1 $
 *  \author G. Mila - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonChi2MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double maxChi2, double nSigma = 3.)
  :Chi2MeasurementEstimatorBase(maxChi2,nSigma) , 
   theDtChi2Estimator(maxChi2, nSigma) , 
   theCscChi2Estimator(maxChi2, nSigma) ,
   theRpcChi2Estimator(maxChi2, nSigma) {}


MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double dtMaxChi2, double cscMaxChi2, double rpcMaxChi2, double nSigma = 3.)
  :Chi2MeasurementEstimatorBase(dtMaxChi2,nSigma) , // fake value for maxChi2
   theDtChi2Estimator(dtMaxChi2, nSigma) , 
   theCscChi2Estimator(cscMaxChi2, nSigma) ,
   theRpcChi2Estimator(rpcMaxChi2, nSigma) {}


std::pair<bool,double> 
MuonChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				       const TransientTrackingRecHit& recHit) const {
  
   DetId id = recHit.geographicalId();
   std::pair<bool,double> result;

   // chi2 choise based on recHit provenance
   if(id.det() == DetId::Muon &&
      (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC || id.subdetId() == MuonSubdetId::RPC)){
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT )
       result=theDtChi2Estimator.estimate(tsos,recHit);
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC )
       result=theCscChi2Estimator.estimate(tsos,recHit);
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC ) 
       result=theRpcChi2Estimator.estimate(tsos,recHit);
   }
   else
     edm::LogError("invalidRecHitId")
       <<"RecHit with MuonId but not SubDetId from dt or csc or rpc!";

   return result;

}
