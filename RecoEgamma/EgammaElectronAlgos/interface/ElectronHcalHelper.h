
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

class ElectronHcalHelper
 {
  public:

    struct Configuration
     {
      // common parameters
      double hOverEConeSize ;

      // strategy
      bool useTowers ;

      // specific parameters if use towers
      edm::InputTag hcalTowers ;
      double hOverEPtMin ;            // min tower Et for H/E evaluation

      // specific parameters if use rechits
      edm::InputTag hcalRecHits ;
      double hOverEHBMinE ;
      double hOverEHFMinE ;
     } ;

    ElectronHcalHelper( const Configuration & ) ;
    void checkSetup( const edm::EventSetup & ) ;
    void readEvent( edm::Event & ) ;
    double hcalESum( const reco::SuperCluster & ) ;
    double hcalESumDepth1( const reco::SuperCluster & ) ;
    double hcalESumDepth2( const reco::SuperCluster & ) ;
    ~ElectronHcalHelper() ;

    double hOverEConeSize() const { return cfg_.hOverEConeSize ; }


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

 } ;

#endif



