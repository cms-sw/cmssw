
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/07/16 10:38:25 $
 *  $Revision: 1.4 $
 *  \author G. Mila - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonChi2MeasurementEstimator.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"


MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double maxChi2, double nSigma = 3.)
  :Chi2MeasurementEstimatorBase(maxChi2,nSigma) , 
   theDtChi2Estimator(maxChi2, nSigma) , 
   theCscChi2Estimator(maxChi2, nSigma) ,
   theRpcChi2Estimator(maxChi2, nSigma) {}


MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double dtMaxChi2, double cscMaxChi2, double rpcMaxChi2, double nSigma = 3.)
  :Chi2MeasurementEstimatorBase(dtMaxChi2,nSigma) , // to fix
   theDtChi2Estimator(dtMaxChi2, nSigma) , 
   theCscChi2Estimator(cscMaxChi2, nSigma) ,
   theRpcChi2Estimator(rpcMaxChi2, nSigma) {}


Chi2MeasurementEstimator
MuonChi2MeasurementEstimator::estimate(const TransientTrackingRecHit& recHit) const {
  
   DetId id = recHit.geographicalId();
   Chi2MeasurementEstimator* theChi2Estimator=0;

   // chi2 choise based on recHit provenance
   if(id.det() == DetId::Muon &&
      (id.subdetId() == MuonSubdetId::DT || id.subdetId() == MuonSubdetId::CSC || id.subdetId() == MuonSubdetId::RPC)){
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::DT )
       (*theChi2Estimator)=theDtChi2Estimator;
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::CSC )
       (*theChi2Estimator)=theCscChi2Estimator;
     if (id.det() == DetId::Muon && id.subdetId() == MuonSubdetId::RPC ) 
       (*theChi2Estimator)=theRpcChi2Estimator;
   }
   else
     cms::Exception("RecHit of invalid id (not dt,csc,rpc");

   return (*theChi2Estimator);

}
