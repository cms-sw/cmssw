#ifndef RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_
#define RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "FWCore/Framework/interface/ESHandle.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include <map>
#include <string>

class HiEgammaSCEnergyCorrectionAlgo {
public:
  // the Verbosity levels
  enum VerbosityLevel { pDEBUG = 0, pWARNING = 1, pINFO = 2, pERROR = 3 }; 
  
  // public member functions
  HiEgammaSCEnergyCorrectionAlgo(float noise, 
				 reco::CaloCluster::AlgoId theAlgo,
				 const edm::ParameterSet& pSet, 
				 VerbosityLevel verbosity = pERROR
                                 );
  ~HiEgammaSCEnergyCorrectionAlgo(){}
  
  // take a SuperCluster and return a corrected SuperCluster
  reco::SuperCluster applyCorrection(const reco::SuperCluster &cl, 
				     const EcalRecHitCollection &rhc, 
				     reco::CaloCluster::AlgoId theAlgo, 
				     const CaloSubdetectorGeometry* geometry,
				     const CaloTopology *topology,
				     EcalClusterFunctionBaseClass* EnergyCorrectionClass);
  
  // function to set the verbosity level
  void setVerbosity(VerbosityLevel verbosity)
  {
    verbosity_ = verbosity;
  }
 
private:
  
  // correction factor as a function of number of crystals,
  // BasicCluster algo and location in the detector    
  float fNCrystals(int nCry, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const ;
  float fBrem(float widthRatio, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const;
  float fEta(float eta, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const;
  float fEtEta(float et, float eta, reco::CaloCluster::AlgoId theAlgo, EcalSubdetector theBase) const;
  
  // Return the number of crystals in a BasicCluster above 
  // 2sigma noise level
  int nCrystalsGT2Sigma(reco::BasicCluster const & seed, EcalRecHitCollection const &rhc) const;
    
    float sigmaElectronicNoise_;
  
   //  the verbosity level
  VerbosityLevel verbosity_;
  
  reco::CaloCluster::AlgoId theAlgo_;
  
  // parameters
  std::vector<double> p_fEta_;
  std::vector<double> p_fBremTh_, p_fBrem_;
  std::vector<double> p_fEtEta_;
  
  double minR9Barrel_;
  double minR9Endcap_;
  double maxR9_;
};

#endif /*RecoECAL_ECALClusters_HiEgammaSCEnergyCorrectionAlgo_h_*/
