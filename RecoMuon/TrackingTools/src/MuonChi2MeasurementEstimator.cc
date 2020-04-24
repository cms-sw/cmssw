/** \class MuonChi2MeasurementEstimator
 *  Class to handle different chi2 cut parameters for each muon sub-system.
 *  MuonChi2MeasurementEstimator inherits from the Chi2MeasurementEstimatorBase class and uses
 *  3 different estimators.
 *
 *  \author Giorgia Mila - INFN Torino
 */

#include "RecoMuon/TrackingTools/interface/MuonChi2MeasurementEstimator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/getRef.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
 

MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double maxChi2, double nSigma)
  :Chi2MeasurementEstimatorBase(maxChi2,nSigma), 
   theDTChi2Estimator(maxChi2, nSigma), 
   theCSCChi2Estimator(maxChi2, nSigma),
   theRPCChi2Estimator(maxChi2, nSigma){}


MuonChi2MeasurementEstimator::MuonChi2MeasurementEstimator(double dtMaxChi2, double cscMaxChi2, double rpcMaxChi2, double nSigma = 3.)
  :Chi2MeasurementEstimatorBase(dtMaxChi2,nSigma),
   theDTChi2Estimator(dtMaxChi2, nSigma), 
   theCSCChi2Estimator(cscMaxChi2, nSigma),
   theRPCChi2Estimator(rpcMaxChi2, nSigma){}


std::pair<bool,double> 
MuonChi2MeasurementEstimator::estimate(const TrajectoryStateOnSurface& tsos,
				       const TrackingRecHit& recHit) const {
  
  DetId id = recHit.geographicalId();
  
  // chi2 choise based on recHit provenance
  if(id.det() == DetId::Muon){
    if(id.subdetId() == MuonSubdetId::DT)
      return theDTChi2Estimator.estimate(tsos,recHit);
    else if(id.subdetId() == MuonSubdetId::CSC)
      return theCSCChi2Estimator.estimate(tsos,recHit);
    else if(id.subdetId() == MuonSubdetId::RPC) 
      return theRPCChi2Estimator.estimate(tsos,recHit);
    else{
      edm::LogWarning("Muon|RecoMuon|MuonChi2MeasurementEstimator")
	<<"RecHit with MuonId but not with a SubDetId neither from DT, CSC or rpc. [Use the parameters used for DTs]";
      return theDTChi2Estimator.estimate(tsos,recHit);
    }
  }
  else{
    edm::LogWarning("Muon|RecoMuon|MuonChi2MeasurementEstimator")
      <<"Rechit with a non-muon det id. [Use the parameters used for DTs]";
    return theDTChi2Estimator.estimate(tsos,recHit);
  }
}
