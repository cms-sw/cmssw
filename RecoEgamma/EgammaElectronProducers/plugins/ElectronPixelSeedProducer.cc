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
// $Id: ElectronPixelSeedProducer.cc,v 1.12 2007/12/17 16:58:41 uberthon Exp $
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
#include "RecoEgamma/EgammaElectronAlgos/interface/SubSeedGenerator.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "ElectronPixelSeedProducer.h"

#include <string>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{

  std::string algo = iConfig.getParameter<std::string>("SeedAlgo");
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  if (algo=="FilteredSeed") 
    matcher_= new SubSeedGenerator(pset);
    //    matcher_= new SubSeedGenerator(pset.getParameter<std::string>("initialSeedProducer"),pset.getParameter<std::string>("initialSeedLabel"));
  else matcher_ = new ElectronPixelSeedGenerator(pset.getParameter<double>("ePhiMin1"),
					    pset.getParameter<double>("ePhiMax1"),
					    pset.getParameter<double>("pPhiMin1"),
					    pset.getParameter<double>("pPhiMax1"),
					    pset.getParameter<double>("pPhiMin2"),
					    pset.getParameter<double>("pPhiMax2"),
						 //					    pset.getParameter<double>("ZMin1"),
						 //					    pset.getParameter<double>("ZMax1"),
					    pset.getParameter<double>("ZMin2"),
					    pset.getParameter<double>("ZMax2"),
                                            pset.getParameter<bool>("dynamicPhiRoad"),
						 pset.getParameter<double>("SCEtCut"));
					      
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");

  //register your products
  produces<ElectronPixelSeedCollection>();
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
}

void ElectronPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
     matcher_->setupES(iSetup);  
}

void ElectronPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("entering");
  LogDebug("")  <<"[ElectronPixelSeedProducer::produce] entering " ;

  ElectronPixelSeedCollection *seeds= new ElectronPixelSeedCollection;
  std::auto_ptr<ElectronPixelSeedCollection> pSeeds;
 
  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  
    // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    if (e.getByLabel(label_[i],instanceName_[i],clusters))     matcher_->run(e,iSetup,clusters,*seeds);
  }

  // store the accumulated result
  pSeeds=  std::auto_ptr<ElectronPixelSeedCollection>(seeds);
  for (ElectronPixelSeedCollection::iterator is=pSeeds->begin(); is!=pSeeds->end();is++) {
      LogDebug("")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
  }
  e.put(pSeeds);

}


