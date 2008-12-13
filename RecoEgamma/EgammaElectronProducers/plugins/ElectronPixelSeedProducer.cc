// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronPixelSeedProducer
// 
/**\class ElectronPixelSeedProducer RecoEgamma/ElectronProducers/src/ElectronPixelSeedProducer.cc

 Description: EDProducer of ElectronPixelSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedProducer.cc,v 1.25 2008/08/27 14:36:43 charlot Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/SeedFilter.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"

#include "ElectronPixelSeedProducer.h"

#include <string>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) :conf_(iConfig),seedFilter_(0),cacheID_(0)
{
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  initialSeeds_=pset.getParameter<edm::InputTag>("initialSeeds");
  SCEtCut_=pset.getParameter<double>("SCEtCut");
  maxHOverEDepth1_=pset.getParameter<double>("maxHOverEDepth1");
  maxHOverEDepth2_=pset.getParameter<double>("maxHOverEDepth2");
  hOverEConeSize_=pset.getParameter<double>("hOverEConeSize");
  hOverEPtMin_=pset.getParameter<double>("hOverEPtMin");
  fromTrackerSeeds_=pset.getParameter<bool>("fromTrackerSeeds");
  prefilteredSeeds_=pset.getParameter<bool>("preFilteredSeeds");

  matcher_ = new ElectronPixelSeedGenerator(pset);
 
  if (prefilteredSeeds_) seedFilter_ = new SeedFilter(pset);

  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");
  hcalTowers_ = pset.getParameter<edm::InputTag>("hcalTowers");

  //register your products
  produces<ElectronPixelSeedCollection>();
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
      delete seedFilter_;
}

void ElectronPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
}

void ElectronPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("ElectronPixelSeedProducer");
  LogDebug("ElectronPixelSeedProducer")  <<"[ElectronPixelSeedProducer::produce] entering " ;

  matcher_->setupES(iSetup);  

  // get Hcal towers collection
  edm::Handle<CaloTowerCollection> towersHandle;
  e.getByLabel(hcalTowers_, towersHandle);
  const CaloTowerCollection* towers = towersHandle.product();
  
  // get initial TrajectorySeeds if necessary
  if (fromTrackerSeeds_) {
    if (!prefilteredSeeds_) {
      edm::Handle<TrajectorySeedCollection> hSeeds;
      e.getByLabel(initialSeeds_, hSeeds);
      theInitialSeedColl = const_cast<TrajectorySeedCollection *> (hSeeds.product());
    }
    else theInitialSeedColl =new TrajectorySeedCollection;
  }else
    theInitialSeedColl=0;// not needed in this case
 
  ElectronPixelSeedCollection *seeds= new ElectronPixelSeedCollection;
  std::auto_ptr<ElectronPixelSeedCollection> pSeeds;

  // HCAL iso deposits
  towerIso1_  = new EgammaTowerIsolation(hOverEConeSize_,0.,hOverEPtMin_,1,towers) ;  
  towerIso2_  = new EgammaTowerIsolation(hOverEConeSize_,0.,hOverEPtMin_,2,towers) ;  

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  
   // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    if (e.getByLabel(superClusters_[i],clusters))   {
	SuperClusterRefVector clusterRefs;
	filterClusters(clusters,towers,clusterRefs);
	if ((fromTrackerSeeds_) && (prefilteredSeeds_)) filterSeeds(e,iSetup,clusterRefs);
        matcher_->run(e,iSetup,clusterRefs,theInitialSeedColl,*seeds);

    }
  }

  // store the accumulated result
  pSeeds=  std::auto_ptr<ElectronPixelSeedCollection>(seeds);
  for (ElectronPixelSeedCollection::iterator is=pSeeds->begin(); is!=pSeeds->end();is++) {
    LogDebug("ElectronPixelSeedProducer")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
  }
  e.put(pSeeds);
  delete towers;
  if (fromTrackerSeeds_ && prefilteredSeeds_) delete theInitialSeedColl;
 }

void ElectronPixelSeedProducer::filterClusters(const edm::Handle<reco::SuperClusterCollection> &superClusters, 
 const CaloTowerCollection *towers, SuperClusterRefVector &sclRefs) {

  // filter the superclusters with Et cut and HCAL towers content behind SC position
  for (unsigned int i=0;i<superClusters->size();++i) {
    const SuperCluster &scl=(*superClusters)[i];

    if (scl.energy()/cosh(scl.eta())>SCEtCut_) {

      double HoE1=towerIso1_->getTowerESum(&scl)/scl.energy();
      double HoE2=towerIso2_->getTowerESum(&scl)/scl.energy();
      if ( HoE1 <= maxHOverEDepth1_ && HoE2 <= maxHOverEDepth2_ ) {
	sclRefs.push_back(edm::Ref<reco::SuperClusterCollection> (superClusters,i));
      }
 
    }

  }
  LogDebug("ElectronPixelSeedProducer")  <<"Filtered out "<<sclRefs.size() <<" superclusters from "<<superClusters->size() ;
}

void ElectronPixelSeedProducer::filterSeeds(edm::Event& e, const edm::EventSetup& setup, reco::SuperClusterRefVector &sclRefs)
{

  for  (unsigned int i=0;i<sclRefs.size();++i) {
    seedFilter_->seeds(e, setup, sclRefs[i], theInitialSeedColl);

    LogDebug("ElectronPixelSeedProducer")<< "Number fo Seeds: " << theInitialSeedColl->size() ;
  }
 

}
