// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      ElectronGSPixelSeedProducer
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
#include "FastSimulation/EgammaElectronAlgos/plugins/ElectronGSPixelSeedProducer.h"
#include "FastSimulation/EgammaElectronAlgos/interface/ElectronGSPixelSeedGenerator.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include <iostream>

//using namespace reco;
 
ElectronGSPixelSeedProducer::ElectronGSPixelSeedProducer(const edm::ParameterSet& iConfig)
{

  matcher_ = new ElectronGSPixelSeedGenerator(iConfig.getParameter<double>("ePhiMin1"),
					      iConfig.getParameter<double>("ePhiMax1"),
					      iConfig.getParameter<double>("pPhiMin1"),
					      iConfig.getParameter<double>("pPhiMax1"),
					      iConfig.getParameter<double>("pPhiMin2"),
					      iConfig.getParameter<double>("pPhiMax2"),
					      iConfig.getParameter<double>("ZMin1"),
					      iConfig.getParameter<double>("ZMax1"),
					      iConfig.getParameter<double>("ZMin2"),
					      iConfig.getParameter<double>("ZMax2"),
					      iConfig.getParameter<bool>("dynamicPhiRoad") );
					      
  matcher_->setup(true); //always set to offline in our case!
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");

  //register your products
  produces<TrajectorySeedCollection>(label_[0]);
  produces<TrajectorySeedCollection>(label_[1]);
  produces<reco::SeedSuperClusterAssociationCollection>(label_[0]);
  produces<reco::SeedSuperClusterAssociationCollection>(label_[1]);
}


ElectronGSPixelSeedProducer::~ElectronGSPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
}

void ElectronGSPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
     matcher_->setupES(iSetup);  
}

void ElectronGSPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("entering");
  LogDebug("")  <<"[ElectronGSPixelSeedProducer::produce] entering " ;

  std::auto_ptr<reco::ElectronPixelSeedCollection> pSeeds[2];
  std::auto_ptr<reco::SeedSuperClusterAssociationCollection> pOutAssos[2];
  std::auto_ptr<TrajectorySeedCollection> pOutSeeds[2];
  pSeeds[0]=  std::auto_ptr<reco::ElectronPixelSeedCollection>(new reco::ElectronPixelSeedCollection);
  pSeeds[1]=  std::auto_ptr<reco::ElectronPixelSeedCollection>(new reco::ElectronPixelSeedCollection);
  pOutAssos[0] =std::auto_ptr<reco::SeedSuperClusterAssociationCollection>(new reco::SeedSuperClusterAssociationCollection);
  pOutAssos[1] =std::auto_ptr<reco::SeedSuperClusterAssociationCollection>(new reco::SeedSuperClusterAssociationCollection);
  pOutSeeds[0] =std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
  pOutSeeds[1] =std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
 
  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  

    // invoke algorithm
    edm::Handle<reco::SuperClusterCollection> clusters;
    e.getByLabel(label_[i],instanceName_[i],clusters);
    matcher_->run(e,clusters,*pSeeds[i]);
  
    // convert ElectronPixelSeeds into trajectorySeeds 
    // we have first to create AND put the TrajectorySeedCollection
    // in order to get a working Handle
    // if not there is a problem when inserting into the map

    pOutSeeds[i]->reserve(pSeeds[i]->size());  
    for (reco::ElectronPixelSeedCollection::iterator is=pSeeds[i]->begin(); is!=pSeeds[i]->end();is++) {
      //
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
      TrajectorySeed *seed = &(*is);
      pOutSeeds[i]->push_back(*seed);
    }
    const edm::OrphanHandle<TrajectorySeedCollection> refprod =  e.put(pOutSeeds[i],label_[i]);

    // now we can put the Ref-s into the associationmap
    unsigned int id=0;
    for (reco::ElectronPixelSeedCollection::iterator is=pSeeds[i]->begin(); is!=pSeeds[i]->end();is++) {
      //
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
      reco::SuperClusterRef refsc = is->superCluster();
      edm::Ref<TrajectorySeedCollection> refseed(refprod,id++);
      LogDebug("")  <<" Adding scl ref with PID "<<refsc.id();
      pOutAssos[i]->insert(refseed,refsc);
    }
    e.put(pOutAssos[i],label_[i]);
  }

}


