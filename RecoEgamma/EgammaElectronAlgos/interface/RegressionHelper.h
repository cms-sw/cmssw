
#ifndef __EgammaElectronAlgos_RegressionHelper_h__
#define __EgammaElectronAlgos_RegressionHelper_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"



class EcalClusterLocal;

class RegressionHelper {
  public:
  struct Configuration
  {
      // weight files for the regression
    std::vector<std::string> ecalRegressionWeightLabels;
    bool ecalWeightsFromDB ;
    std::vector<std::string> ecalRegressionWeightFiles;
    std::vector<std::string> combinationRegressionWeightLabels;
    bool combinationWeightsFromDB;
    std::vector<std::string> combinationRegressionWeightFiles;
  };
  
  RegressionHelper( const Configuration & ) ;
  void checkSetup( const edm::EventSetup & ) ;
  void readEvent( const edm::Event & ) ;
  void applyEcalRegression(reco::GsfElectron & electron,
			   const edm::Handle<reco::VertexCollection>& vertices,
			   const edm::Handle<EcalRecHitCollection>& rechitsEB,
			   const edm::Handle<EcalRecHitCollection>& rechitsEE) const;

  void applyCombinationRegression(reco::GsfElectron & ele) const;
  ~RegressionHelper() ;

 private:
  void getEcalRegression(const reco::SuperCluster & sc,
			 const edm::Handle<reco::VertexCollection>& vertices,
			 const edm::Handle<EcalRecHitCollection>& rechitsEB,
			 const edm::Handle<EcalRecHitCollection>& rechitsEE,
			 double & energyFactor,
			 double & errorFactor) const ;
  
 private:
  
  const Configuration cfg_ ;
  const CaloTopology * caloTopology_;
  const CaloGeometry * caloGeometry_;
  bool ecalRegressionInitialized_;
  bool combinationRegressionInitialized_;

  unsigned long long caloTopologyCacheId_;
  unsigned long long caloGeometryCacheId_;
  unsigned long long regressionCacheId_;

  //  edm::ESHandle<GBRForest> ecalRegBarrel_ ;
  const GBRForest * ecalRegBarrel_ ;
  const GBRForest * ecalRegEndcap_ ;
  const GBRForest * ecalRegErrorBarrel_ ;  
  const GBRForest * ecalRegErrorEndcap_ ;  
  const GBRForest * combinationReg_ ;
   
  EcalClusterLocal ecl_;
};

#endif



