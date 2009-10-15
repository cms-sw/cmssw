#ifndef ElectronSeedProducer_h
#define ElectronSeedProducer_h

//
// Package:         RecoEgamma/ElectronTrackSeedProducers
// Class:           ElectronSeedProducer
//
// Description:     Calls ElectronSeedGenerator
//                  to find TrackingSeeds.


class ElectronSeedGenerator ;
class SeedFilter ;
class EgammaHcalIsolation ;
class ElectronHcalHelper ;

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/MetaCollections/interface/CaloRecHitMetaCollections.h"
#include "RecoCaloTools/Selectors/interface/CaloDualConeSelector.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class ElectronSeedProducer : public edm::EDProducer
 {
  public:

    explicit ElectronSeedProducer(const edm::ParameterSet& conf);

    virtual ~ElectronSeedProducer();

    virtual void produce(edm::Event& e, const edm::EventSetup& c);

  private:

    void filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters,
        /*HBHERecHitMetaCollection*mhbhe,*/ reco::SuperClusterRefVector &sclRefs);
    void filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs);

    edm::InputTag superClusters_[2] ;
    edm::InputTag initialSeeds_ ;

    //const edm::ParameterSet conf_;
    ElectronSeedGenerator * matcher_ ;
    SeedFilter * seedFilter_;

    TrajectorySeedCollection * theInitialSeedColl ;

    // for the filter
  //  edm::ESHandle<CaloGeometry> caloGeom_ ;
  //  unsigned long long caloGeomCacheId_ ;
  //  EgammaHcalIsolation * hcalIso_ ;
  ////  CaloDualConeSelector * doubleConeSel_ ;
  //  HBHERecHitMetaCollection * mhbhe_ ;

    // H/E
  //  edm::InputTag hcalRecHits_;
    ElectronHcalHelper * hcalHelper_ ;
    double maxHOverE_ ;
  //  double hOverEConeSize_;
  //  double hOverEHBMinE_;
  //  double hOverEHFMinE_;

    // super cluster Et cut
    double SCEtCut_;


    bool fromTrackerSeeds_;
    bool prefilteredSeeds_;

 } ;

#endif



