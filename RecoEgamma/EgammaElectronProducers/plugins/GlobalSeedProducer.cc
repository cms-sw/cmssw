// -*- C++ -*-
//
// Package:    ElectronProducers
// Class:      GlobalSeedProducer
// 
/**\class GlobalSeedProducer RecoEgamma/ElectronProducers/src/GlobalSeedProducer.cc

 Description: EDProducer of ElectronPixelSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: GlobalSeedProducer.cc,v 1.21 2008/04/14 16:35:25 charlot Exp $
//
//

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//#include "Geometry/Records/interface/IdealGeometryRecord.h"
//#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "RecoEgamma/EgammaElectronAlgos/interface/SubSeedGenerator.h"

#include "DataFormats/EgammaReco/interface/ElectronPixelSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronPixelSeedFwd.h"

#include "GlobalSeedProducer.h"

#include <string>

using namespace reco;
 
GlobalSeedProducer::GlobalSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{

  //  edm::ParameterSet pset = iConfig.getParameter<edm::ParameterSet>("SeedConfiguration");
  //  SCEtCut_=pset.getParameter<double>("SCEtCut");
  //  maxHOverE_=pset.getParameter<double>("maxHOverE");

  matcher_= new SubSeedGenerator(iConfig);
 
  //  get collections from config'
  superClusters_[0]=iConfig.getParameter<edm::InputTag>("barrelSuperClusters");
  superClusters_[1]=iConfig.getParameter<edm::InputTag>("endcapSuperClusters");

  //register your products
  produces<ElectronPixelSeedCollection>();
}


GlobalSeedProducer::~GlobalSeedProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
}

void GlobalSeedProducer::beginJob(edm::EventSetup const&iSetup) 
{
}

void GlobalSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup) 
{
  LogDebug("GlobalSeedProducer");
  LogDebug("GlobalSeedProducer")  <<"[GlobalSeedProducer::produce] entering " ;

  ElectronPixelSeedCollection *seeds= new ElectronPixelSeedCollection;
  std::auto_ptr<ElectronPixelSeedCollection> pSeeds;

  // loop over barrel + endcap

  for (unsigned int i=0; i<2; i++) {  
   // invoke algorithm
    edm::Handle<SuperClusterCollection> clusters;
    if (e.getByLabel(superClusters_[i],clusters))     matcher_->run(e,iSetup,clusters,*seeds);
  }

  // store the accumulated result
  pSeeds=  std::auto_ptr<ElectronPixelSeedCollection>(seeds);
  for (ElectronPixelSeedCollection::iterator is=pSeeds->begin(); is!=pSeeds->end();is++) {
    LogDebug("ElectronPixelSeedProducer")  << "new seed with " << (*is).nHits() << " hits, charge " << (*is).getCharge() <<
	" and cluster energy " << (*is).superCluster()->energy() << " PID "<<(*is).superCluster().id();
  }
  e.put(pSeeds);
  }
