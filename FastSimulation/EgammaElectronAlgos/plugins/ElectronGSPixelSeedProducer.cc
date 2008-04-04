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

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"

#include <iostream>

//using namespace reco;
 
ElectronGSPixelSeedProducer::ElectronGSPixelSeedProducer(const edm::ParameterSet& iConfig)
{

  std::string algo = iConfig.getParameter<std::string>("SeedAlgo");
  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");

  /* This will wait for V00-09-01 in 18X / 20X ?
   if (algo=="FilteredSeed") 
     matcher_= 
     new SubSeedGenerator(pset);
   else
  */
   matcher_ = 
     new ElectronGSPixelSeedGenerator(pset.getParameter<double>("ePhiMin1"),
				      pset.getParameter<double>("ePhiMax1"),
				      pset.getParameter<double>("pPhiMin1"),
				      pset.getParameter<double>("pPhiMax1"),
				      pset.getParameter<double>("pPhiMin2"),
				      pset.getParameter<double>("pPhiMax2"),
				      pset.getParameter<double>("ZMin2"),
				      pset.getParameter<double>("ZMax2"),
				      pset.getParameter<bool>("dynamicPhiRoad"),
				      pset.getParameter<double>("SCEtCut"),
				      iConfig.getParameter<double>("pTMin"));
					      
  matcher_->setup(true); //always set to offline in our case!
 
 //  get labels from config'
  label_[0]=iConfig.getParameter<std::string>("superClusterBarrelProducer");
  instanceName_[0]=iConfig.getParameter<std::string>("superClusterBarrelLabel");
  label_[1]=iConfig.getParameter<std::string>("superClusterEndcapProducer");
  instanceName_[1]=iConfig.getParameter<std::string>("superClusterEndcapLabel");

  //register your products
  produces<reco::ElectronPixelSeedCollection>();

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

  std::auto_ptr<reco::ElectronPixelSeedCollection> pSeeds;
  reco::ElectronPixelSeedCollection *seeds= new reco::ElectronPixelSeedCollection;

  // loop over barrel + endcap
  for (unsigned int i=0; i<2; i++) {  

    // invoke algorithm
    edm::Handle<reco::SuperClusterCollection> clusters;
    e.getByLabel(label_[i],instanceName_[i],clusters);
    matcher_->run(e,clusters,*seeds);
  
  }

  pSeeds=  std::auto_ptr<reco::ElectronPixelSeedCollection>(seeds);
  e.put(pSeeds);
}


