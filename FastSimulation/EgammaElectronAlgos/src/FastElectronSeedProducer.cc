// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      FastElectronSeedProducer
//
/**

 Description: EDProducer of ElectronGSPixelSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Patrick Janot
//
//

// user include files
#include "FastElectronSeedProducer.h"
#include "FastElectronSeedGenerator.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaHcalIsolation.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"
#include "RecoCaloTools/Selectors/interface/CaloConeSelector.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <iostream>

FastElectronSeedProducer::FastElectronSeedProducer(const edm::ParameterSet& iConfig)
 : matcher_(0), caloGeomCacheId_(0), hcalIso_(0), /*doubleConeSel_(0),*/ mhbhe_(0)
 {
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverE_=pset.getParameter<double>("maxHOverE");
  hOverEConeSize_=pset.getParameter<double>("hOverEConeSize");
  hOverEHBMinE_=pset.getParameter<double>("hOverEHBMinE");
  hOverEHFMinE_=pset.getParameter<double>("hOverEHFMinE");
  fromTrackerSeeds_=pset.getParameter<bool>("fromTrackerSeeds");
  initialSeeds_=pset.getParameter<edm::InputTag>("initialSeeds");

  matcher_ = new FastElectronSeedGenerator(pset,
					      iConfig.getParameter<double>("pTMin"),
					      iConfig.getParameter<edm::InputTag>("beamSpot"));

 //  get labels from config'
  clusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  clusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");
  simTracks_=iConfig.getParameter<edm::InputTag>("simTracks");
  trackerHits_=iConfig.getParameter<edm::InputTag>("trackerHits");
  hcalRecHits_= pset.getParameter<edm::InputTag>("hcalRecHits");

  //register your products
  produces<reco::ElectronSeedCollection>();

}


FastElectronSeedProducer::~FastElectronSeedProducer()
 {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete matcher_ ;
  delete mhbhe_ ;
  //delete doubleConeSel_ ;
  delete hcalIso_ ;
 }

void
FastElectronSeedProducer::beginRun(edm::Run & run, const edm::EventSetup & es)
 {
  // get calo geometry
  if (caloGeomCacheId_!=es.get<CaloGeometryRecord>().cacheIdentifier())
   {
	es.get<CaloGeometryRecord>().get(caloGeom_);
	caloGeomCacheId_=es.get<CaloGeometryRecord>().cacheIdentifier();
   }

//  // The H/E calculator
//  calc_=HoECalculator(caloGeom_);

  matcher_->setupES(es) ;

 }

void
FastElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  LogDebug("FastElectronSeedProducer")<<"[FastElectronSeedProducer::produce] entering " ;

  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_) {

    edm::Handle<TrajectorySeedCollection> hSeeds;
    e.getByLabel(initialSeeds_, hSeeds);
    initialSeedColl_ = const_cast<TrajectorySeedCollection *> (hSeeds.product());

  } else {

    initialSeedColl_=0;// not needed in this case

  }

  reco::ElectronSeedCollection * seeds = new reco::ElectronSeedCollection ;

  // Get the Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSTC;
  e.getByLabel(simTracks_,theSTC);
  const edm::SimTrackContainer* theSimTracks = &(*theSTC);

  // Get the collection of Tracker RecHits
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theRHC;
  e.getByLabel(trackerHits_, theRHC);
  const SiTrackerGSMatchedRecHit2DCollection* theGSRecHits = &(*theRHC);

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHand;
  iSetup.get<IdealGeometryRecord>().get(tTopoHand);
  const TrackerTopology *tTopo=tTopoHand.product();


  // get Hcal Rechit collection
  edm::Handle<HBHERecHitCollection> hbhe ;
  delete mhbhe_ ;
  if (e.getByLabel(hcalRecHits_,hbhe))
   { mhbhe_=  new HBHERecHitMetaCollection(*hbhe) ; }
  else
   { mhbhe_ = 0 ; }

  // define cone for H/E
//  delete doubleConeSel_;
//  doubleConeSel_ = new CaloDualConeSelector(0.,hOverEConeSize_,caloGeom_.product(),DetId::Hcal) ;

  // HCAL iso deposits
  delete hcalIso_ ;
  hcalIso_ = new EgammaHcalIsolation(hOverEConeSize_,0.,hOverEHBMinE_,hOverEHFMinE_,0.,0.,caloGeom_,mhbhe_) ;

  // Get the two supercluster collections
  for (unsigned int i=0; i<2; i++) {

    // invoke algorithm
    edm::Handle<reco::SuperClusterCollection> clusters;
    e.getByLabel(clusters_[i],clusters);
    reco::SuperClusterRefVector clusterRefs;
    filterClusters(clusters,/*mhbhe_,*/clusterRefs) ;
    matcher_->run(e,clusterRefs,theGSRecHits,theSimTracks,initialSeedColl_,tTopo,*seeds);

  }

  // Save event content
  std::auto_ptr<reco::ElectronSeedCollection> pSeeds(seeds) ;
  e.put(pSeeds);

}


void
FastElectronSeedProducer::filterClusters
 ( const edm::Handle<reco::SuperClusterCollection> & superClusters,
   //HBHERecHitMetaCollection * mhbhe,
   reco::SuperClusterRefVector & sclRefs )
 {
  // filter the superclusters
  // - with EtCut
  // - with HoE using calo cone
  for (unsigned int i=0;i<superClusters->size();++i)
   {
    const reco::SuperCluster & scl=(*superClusters)[i] ;
    if (scl.energy()/cosh(scl.eta())>SCEtCut_)
     {
//      //double HoE=calc_(&scl,mhbhe);
//	  double HoE = 0. ;
//      double hcalE = 0. ;
//      if (mhbhe_)
//       {
//        math::XYZPoint caloPos = scl.position() ;
//        GlobalPoint pclu(caloPos.x(),caloPos.y(),caloPos.z()) ;
//        std::auto_ptr<CaloRecHitMetaCollectionV> chosen
//         = doubleConeSel_->select(pclu,*mhbhe_) ;
//        CaloRecHitMetaCollectionV::const_iterator i ;
//        for ( i = chosen->begin () ; i != chosen->end () ; ++i )
//         {
//          double hcalHit_E = i->energy() ;
//          if ( i->detid().subdetId()==HcalBarrel && hcalHit_E > hOverEHBMinE_)
//           { hcalE += hcalHit_E ; } //HB case
//          //if ( i->detid().subdetId()==HcalBarrel)
//          // { std::cout << "[ElectronSeedProducer] HcalBarrel: hcalHit_E, hOverEHBMinE_ " << hcalHit_E << " " << hOverEHBMinE_ << std::endl; }
//          if ( i->detid().subdetId()==HcalEndcap && hcalHit_E > hOverEHFMinE_)
//           { hcalE += hcalHit_E ; } //HF case
//          //if ( i->detid().subdetId()==HcalEndcap)
//          // { std::cout << "[ElectronSeedProducer] HcalEndcap: hcalHit_E, hOverEHFMinE_ " << hcalHit_E << " " << hOverEHFMinE_ << std::endl; }
//         }
//       }
//      HoE = hcalE/scl.energy() ;
      //double hcalE = hcalHelper_->hcalESum(scl), HoE = hcalE/scl.energy() ;
      double newHcalE = hcalIso_->getHcalESum(&scl), newHoE = newHcalE/scl.energy() ;
      //std::cout << "[ElectronSeedProducer] HoE, maxHOverE_ " << newHoE << " " << HoE << " " << maxHOverE_ << std::endl ;
      if (newHoE<=maxHOverE_)
       { sclRefs.push_back(edm::Ref<reco::SuperClusterCollection>(superClusters,i)) ; }
     }
   }

  LogDebug("ElectronSeedProducer")<<"Filtered out "
    <<sclRefs.size()<<" superclusters from "<<superClusters->size() ;
 }
