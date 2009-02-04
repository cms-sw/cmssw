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
#include "FastSimulation/EgammaElectronAlgos/plugins/FastElectronSeedProducer.h"
#include "FastSimulation/EgammaElectronAlgos/interface/FastElectronSeedGenerator.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoEgamma/EgammaTools/interface/HoECalculator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <iostream>

FastElectronSeedProducer::FastElectronSeedProducer(const edm::ParameterSet& iConfig)
{

  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverE_=pset.getParameter<double>("maxHOverE");
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
      delete matcher_;
}

void 
FastElectronSeedProducer::beginRun(edm::Run & run, const edm::EventSetup & es) {
  // Get the calo geometry
  edm::ESHandle<CaloGeometry> theCaloGeom;
  es.get<CaloGeometryRecord>().get(theCaloGeom);

  // The H/E calculator
  calc_=HoECalculator(theCaloGeom);

  matcher_->setupES(es);  

}

void FastElectronSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("entering");
  LogDebug("")  <<"[FastElectronSeedProducer::produce] entering " ;

  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_) {

    edm::Handle<TrajectorySeedCollection> hSeeds;
    e.getByLabel(initialSeeds_, hSeeds);
    theInitialSeedColl = const_cast<TrajectorySeedCollection *> (hSeeds.product());

  } else { 

    theInitialSeedColl=0;// not needed in this case

  }
  
  std::auto_ptr<reco::ElectronSeedCollection> pSeeds;
  reco::ElectronSeedCollection *seeds= new reco::ElectronSeedCollection;

  // Get the Monte Carlo truth (SimTracks)
  edm::Handle<edm::SimTrackContainer> theSTC;
  e.getByLabel(simTracks_,theSTC);
  const edm::SimTrackContainer* theSimTracks = &(*theSTC);
  
  // Get the collection of Tracker RecHits
  edm::Handle<SiTrackerGSMatchedRecHit2DCollection> theRHC;
  e.getByLabel(trackerHits_, theRHC);
  const SiTrackerGSMatchedRecHit2DCollection* theGSRecHits = &(*theRHC);

  // get Hcal Rechit collection
  edm::Handle<HBHERecHitCollection> hbhe;
  HBHERecHitMetaCollection *mhbhe=0;
  bool got = e.getByLabel(hcalRecHits_,hbhe);  
  if (got) mhbhe = new HBHERecHitMetaCollection(*hbhe);

  // Get the two supercluster collections
  for (unsigned int i=0; i<2; i++) {  

    // invoke algorithm
    edm::Handle<reco::SuperClusterCollection> clusters;
    e.getByLabel(clusters_[i],clusters);
    reco::SuperClusterRefVector clusterRefs;
    filterClusters(clusters,mhbhe,clusterRefs);
    matcher_->run(e,clusterRefs,theGSRecHits,theSimTracks,theInitialSeedColl,*seeds);

  }

  // Save event content
  pSeeds=  std::auto_ptr<reco::ElectronSeedCollection>(seeds);
  e.put(pSeeds);

  // Clean memory
  delete mhbhe;
}


void 
FastElectronSeedProducer::filterClusters(const edm::Handle<reco::SuperClusterCollection>& superClusters,
					    HBHERecHitMetaCollection*mhbhe, 
					    reco::SuperClusterRefVector &sclRefs) 
{

  // filter the superclusters
  // - with EtCut
  // with HoE
  for (unsigned int i=0;i<superClusters->size();++i) {

    const reco::SuperCluster &scl=(*superClusters)[i];
    
    if (scl.energy()/cosh(scl.eta())>SCEtCut_) {
      
      double HoE=calc_(&scl,mhbhe);
      if (HoE <= maxHOverE_) {
	sclRefs.push_back(edm::Ref<reco::SuperClusterCollection> (superClusters,i));
      }
      
    }
    
  }

  LogDebug("ElectronSeedProducer")  <<"Filtered out "<<sclRefs.size() <<" superclusters from "<<superClusters->size() ;

}
