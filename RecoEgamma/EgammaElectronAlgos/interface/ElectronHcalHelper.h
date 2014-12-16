
#ifndef ElectronHcalHelper_h
#define ElectronHcalHelper_h

class EgammaHcalIsolation ;
class EgammaTowerIsolation ;

#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

class EgammaHadTower;

class ElectronHcalHelper
 {
  public:

    struct Configuration
     {
      // common parameters
      double hOverEConeSize ;

      // method (0 = cone; 1 = single tower ; 2 towers behind cluster)
      int hOverEMethod ; 

      // strategy
      bool useTowers ;

      // specific parameters if use towers
      edm::InputTag hcalTowers ;
      double hOverEPtMin ;            // min tower Et for H/E evaluation

      // specific parameters if use rechits
      edm::InputTag hcalRecHits ;
      double hOverEHBMinE ;
      double hOverEHFMinE ;

      // specific parameters if use hgcal HF Clusters
      edm::InputTag hcalClusters ;
     } ;

    ElectronHcalHelper( const Configuration & ) ;
    void checkSetup( const edm::EventSetup & ) ;
    void readEvent( const edm::Event & ) ;
    ~ElectronHcalHelper() ;

    double hcalESum( const reco::SuperCluster & , const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
    double hcalESumDepth1( const reco::SuperCluster &, const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
    double hcalESumDepth2( const reco::SuperCluster & ,const std::vector<CaloTowerDetId> * excludeTowers = 0 ) ;

    double hcalESumCone( const reco::SuperCluster & , const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
    double hcalESumDepth1Cone( const reco::SuperCluster &, const std::vector<CaloTowerDetId> * excludeTowers = 0) ;
    double hcalESumDepth2Cone( const reco::SuperCluster & ,const std::vector<CaloTowerDetId> * excludeTowers = 0 ) ;
    double hOverEConeSize() const { return cfg_.hOverEConeSize ; }

    // Behind clusters
    std::vector<CaloTowerDetId> hcalTowersBehindClusters( const reco::SuperCluster & sc ) ;
    double hcalESumDepth1BehindClusters( const std::vector<CaloTowerDetId> & towers ) ;
    double hcalESumDepth2BehindClusters( const std::vector<CaloTowerDetId> & towers ) ;
    
    // HGCal using HCAL clusters
    double HCALClustersBehindSC( const reco::SuperCluster & ) ;
    
    const Configuration& getConfig() const { return cfg_; }

  private:

    const Configuration cfg_ ;

    // event setup data (rechits strategy)
    unsigned long long caloGeomCacheId_ ;
    edm::ESHandle<CaloGeometry> caloGeom_ ;

    // event data (rechits strategy)
    edm::Handle<HBHERecHitCollection> * hbhe_ ;
    HBHERecHitMetaCollection * mhbhe_ ;
    EgammaHcalIsolation * hcalIso_ ;

    // event data (towers strategy)
    edm::Handle<CaloTowerCollection> * towersH_ ;
    EgammaTowerIsolation * towerIso1_ ;
    EgammaTowerIsolation * towerIso2_ ;
    EgammaHadTower * hadTower_;
    
    //hgcal HF Clusters
    edm::Handle<reco::PFClusterCollection> * hcalClusters_ ;
 
 } ;

#endif



