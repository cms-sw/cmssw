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
// $Id: ElectronPixelSeedProducer.cc,v 1.4 2006/10/09 16:35:16 uberthon Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaElectronProducers/interface/ElectronPixelSeedProducer.h"
#include "RecoEgamma/EgammaElectronAlgos/interface/ElectronPixelSeedGenerator.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SeedSuperClusterAssociation.h"

#include <iostream>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{

  matcher_ = new ElectronPixelSeedGenerator(iConfig.getParameter<double>("ePhiMin1"),
					    iConfig.getParameter<double>("ePhiMax1"),
					    iConfig.getParameter<double>("pPhiMin1"),
					    iConfig.getParameter<double>("pPhiMax1"),
					    iConfig.getParameter<double>("pPhiMin2"),
					    iConfig.getParameter<double>("pPhiMax2"),
					    iConfig.getParameter<double>("ZMin1"),
					    iConfig.getParameter<double>("ZMax1"),
					    iConfig.getParameter<double>("ZMin2"),
					    iConfig.getParameter<double>("ZMax2") );
					      
  matcher_->setup(true); //always set to offline in our case!
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");

  //register your products
  produces<TrajectorySeedCollection>(label_[0]);
  produces<TrajectorySeedCollection>(label_[1]);
  produces<SeedSuperClusterAssociationCollection>(label_[0]);
  produces<SeedSuperClusterAssociationCollection>(label_[1]);
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
}

void ElectronPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
     matcher_->setupES(iSetup,conf_);  
}

void ElectronPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("entering");
  LogDebug("")  <<"[ElectronPixelSeedProducer::produce] entering " ;

  std::auto_ptr<ElectronPixelSeedCollection> pSeeds[2];
  std::auto_ptr<SeedSuperClusterAssociationCollection> pOutAssos[2];
  std::auto_ptr<TrajectorySeedCollection> pOutSeeds[2];
  pSeeds[0]=  std::auto_ptr<ElectronPixelSeedCollection>(new ElectronPixelSeedCollection);
  pSeeds[1]=  std::auto_ptr<ElectronPixelSeedCollection>(new ElectronPixelSeedCollection);
  pOutAssos[0] =std::auto_ptr<SeedSuperClusterAssociationCollection>(new SeedSuperClusterAssociationCollection);
  pOutAssos[1] =std::auto_ptr<SeedSuperClusterAssociationCollection>(new SeedSuperClusterAssociationCollection);
  pOutSeeds[0] =std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
  pOutSeeds[1] =std::auto_ptr<TrajectorySeedCollection>(new TrajectorySeedCollection);
 
  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  
    // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    e.getByLabel(label_[i],instanceName_[i],clusters);
    matcher_->run(e,clusters,*pSeeds[i]);
  
    /*
    // test of building a ref on ElectronPixelSeedCollection from getRefBeforePut
    // for this test to work on ElectronPixelSeedCollection, ElectronPixelSeed should be registered as 
    // product using produces<ElectronPixelSeedCollection>();
    LogDebug("")  <<"[ElectronPixelSeedProducer::produce] found " << pSeeds->size() << " seeds" ;
    edm::RefProd<ElectronPixelSeedCollection> refp = e.getRefBeforePut<ElectronPixelSeedCollection>();  
    LogDebug("")  <<" RefProd<ElectronPixelSeedCollection> id = " << refp.id() ;
    if (refp.isNull()) LogDebug("")  << "refp isNull !";
    else LogDebug("")  << "refp is NOT Null !" << std::endl;
    edm::Ref<ElectronPixelSeedCollection>::key_type idx = 0;
    edm::Ref<ElectronPixelSeedCollection> ref(refp,idx);
    if (ref.isNull()) LogDebug("")  << "ref isNull !" ;
    else LogDebug("")  << "ref is NOT Null !";
    // up to there, no problem
    // adding the following line leads to a segv with message "InvalidID get by product ID: no product with given id: 63"
    //LogDebug("")  <<"ref has " << ref->nHits() << " hits ";
    */
  
    // convert ElectronPixelSeeds into trajectorySeeds 
    // we have first to create AND put the TrajectorySeedCollection
    // in order to get a working Handle
    // if not there is a problem when inserting into the map

    pOutSeeds[i]->reserve(pSeeds[i]->size());  
    for (ElectronPixelSeedCollection::iterator is=pSeeds[i]->begin(); is!=pSeeds[i]->end();is++) {
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
      TrajectorySeed *seed = &(*is);
      pOutSeeds[i]->push_back(*seed);
    }
    const edm::OrphanHandle<TrajectorySeedCollection> refprod =  e.put(pOutSeeds[i],label_[i]);

    // now we can put the Ref-s into the associationmap
    unsigned int id=0;
    for (ElectronPixelSeedCollection::iterator is=pSeeds[i]->begin(); is!=pSeeds[i]->end();is++) {
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
      SuperClusterRef refsc = is->superCluster();
      edm::Ref<TrajectorySeedCollection> refseed(refprod,id++);
      LogDebug("")  <<" Adding scl ref with PID "<<refsc.id();
      pOutAssos[i]->insert(refseed,refsc);
    }
    e.put(pOutAssos[i],label_[i]);
  }

}


