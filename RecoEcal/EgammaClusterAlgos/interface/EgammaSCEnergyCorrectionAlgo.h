#ifndef RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_
#define RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 

#include <map>
#include <string>

class EgammaSCEnergyCorrectionAlgo {
public:
  
  // public member functions
  EgammaSCEnergyCorrectionAlgo(float noise, 
			       reco::CaloCluster::AlgoId theAlgo,
			       const edm::ParameterSet& pset);
  ~EgammaSCEnergyCorrectionAlgo(){}
  
  // take a SuperCluster and return a corrected SuperCluster
  reco::SuperCluster applyCorrection(const reco::SuperCluster &cl, 
				     const EcalRecHitCollection &rhc, 
				     reco::CaloCluster::AlgoId theAlgo, 
				     const CaloSubdetectorGeometry* geometry,
				     EcalClusterFunctionBaseClass* energyCorrectionFunction,
				     std::string energyCorrectorName_,
				     int modeEB_,
				     int modeEE_);
  
  // take a SuperCluster and return a crack-corrected SuperCluster
  reco::SuperCluster applyCrackCorrection(const reco::SuperCluster &cl,
					  EcalClusterFunctionBaseClass* crackCorrectionFunction);
  
  // take a SuperCluster and return a local containment corrected SuperCluster

  reco::SuperCluster applyLocalContCorrection(const reco::SuperCluster &cl,
					  EcalClusterFunctionBaseClass* localContCorrectionFunction);

private:
  
  // correction factor as a function of number of crystals,
  // BasicCluster algo and location in the detector    
  float fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const;
  
  // Return the number of crystals in a BasicCluster above 
  // 2sigma noise level
  int nCrystalsGT2Sigma(reco::BasicCluster const & seed, EcalRecHitCollection const & rhc) const;
    
  float sigmaElectronicNoise_;
      
  reco::CaloCluster::AlgoId theAlgo_;
  
};

#endif /*RecoECAL_ECALClusters_EgammaSCEnergyCorrectionAlgo_h_*/
