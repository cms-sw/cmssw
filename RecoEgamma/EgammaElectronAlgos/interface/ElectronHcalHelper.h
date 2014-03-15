
#ifndef ElectronHcalHelper_h
#define ElectronHcalHelper_h

class EgammaHcalIsolation ;
class EgammaTowerIsolation ;

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class EgammaHadTower;

class ElectronHcalHelper {
  public:
  
  struct Configuration {
    // common parameters
    double hOverEConeSize;
    
    // strategy
    bool useTowers;
    
    // specific parameters if use towers
    edm::EDGetTokenT<CaloTowerCollection> hcalTowers; 
    double hOverEPtMin ;            // min tower Et for H/E evaluation
    
    // specific parameters if use rechits
    edm::EDGetTokenT<HBHERecHitCollection> hcalRecHits ;
    double hOverEHBMinE ;
    double hOverEHFMinE ;
  } ;
  
  ElectronHcalHelper( const Configuration & ) ;
  void checkSetup( const edm::EventSetup & ) ;
  void readEvent( const edm::Event & ) ;
  ~ElectronHcalHelper() ;
  
  double hcalESum( const reco::SuperCluster & , const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
  double hcalESumDepth1( const reco::SuperCluster &, const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
  double hcalESumDepth2( const reco::SuperCluster & ,const std::vector<CaloTowerDetId> * excludeTowers = 0 ) ;
  double hOverEConeSize() const { return cfg_.hOverEConeSize ; }
  
  // Behind clusters
  std::vector<CaloTowerDetId> hcalTowersBehindClusters( const reco::SuperCluster & sc ) ;
  double hcalESumDepth1BehindClusters( const std::vector<CaloTowerDetId> & towers ) ;
  double hcalESumDepth2BehindClusters( const std::vector<CaloTowerDetId> & towers ) ;
  
 private:
  
  const Configuration cfg_ ;
  
  // event setup data (rechits strategy)
  unsigned long long caloGeomCacheId_ ;
  edm::ESHandle<CaloGeometry> caloGeom_ ;
  
  // event data (rechits strategy)    
  EgammaHcalIsolation * hcalIso_ ;
  
  // event data (towers strategy)    
  EgammaTowerIsolation * towerIso1_ ;
  EgammaTowerIsolation * towerIso2_ ;
  EgammaHadTower * hadTower_;
};

#endif



