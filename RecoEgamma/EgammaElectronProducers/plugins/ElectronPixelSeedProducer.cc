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
// $Id: ElectronPixelSeedProducer.cc,v 1.3 2007/10/12 11:30:42 uberthon Exp $
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
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "ElectronPixelSeedProducer.h"

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
					    iConfig.getParameter<double>("ZMax2"),
                                            iConfig.getParameter<bool>("dynamicPhiRoad") );
					      
  matcher_->setup(true); //always set to offline in our case!
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");

  //register your products
  produces<ElectronPixelSeedCollection>(label_[0]);
  produces<ElectronPixelSeedCollection>(label_[1]);
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
  pSeeds[0]=  std::auto_ptr<ElectronPixelSeedCollection>(new ElectronPixelSeedCollection);
  pSeeds[1]=  std::auto_ptr<ElectronPixelSeedCollection>(new ElectronPixelSeedCollection);
 
  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  
    // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    e.getByLabel(label_[i],instanceName_[i],clusters);
    matcher_->run(e,clusters,*pSeeds[i]);
  
    for (ElectronPixelSeedCollection::iterator is=pSeeds[i]->begin(); is!=pSeeds[i]->end();is++) {
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
    }
    e.put(pSeeds[i],label_[i]);
  }

}


