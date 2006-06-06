// -*- C++ -*-
//
// Package:    ElectronTrackSeedProducers
// Class:      ElectronPixelSeedProducer
// 
/**\class ElectronPixelSeedProducer RecoEgamma/ElectronTrackSeedProducers/src/ElectronPixelSeedProducer.cc

 Description: EDProducer of ElectronPixelSeed objects

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Ursula Berthon, Claude Charlot
//         Created:  Mon Mar 27 13:22:06 CEST 2006
// $Id: ElectronPixelSeedProducer.cc,v 1.1 2006/06/02 15:32:45 uberthon Exp $
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

#include <iostream>

using namespace reco;
 
ElectronPixelSeedProducer::ElectronPixelSeedProducer(const edm::ParameterSet& iConfig) : conf_(iConfig)
{
  //register your products
  produces<ElectronPixelSeedCollection>();

  //create top algorithm and initialize
  matcher_ = new ElectronPixelSeedGenerator();
  matcher_->setup(true); //always set to offline in our case!
}


ElectronPixelSeedProducer::~ElectronPixelSeedProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
      delete matcher_;
}

void ElectronPixelSeedProducer::beginJob(edm::EventSetup const&iSetup) {
     matcher_->setupES(iSetup,conf_);  
}

// ------------ method called to produce the data  ------------
void
ElectronPixelSeedProducer::produce(edm::Event& e, const edm::EventSetup& iSetup)
{
  // test
//   const MeasurementTracker* theMeasurementTracker = new MeasurementTracker(iSetup,conf_);
//   const GeometricSearchTracker * theGeometricSearchTracker=theMeasurementTracker->geometricSearchTracker();
//   edm::ESHandle<TrackerGeometry> pDD;
//   iSetup.get<TrackerDigiGeometryRecord> ().get (pDD);      

//   edm::Handle<SiPixelRecHitCollection> pixelHits;
//   e.getByType(pixelHits);  
//   edm::Handle<SuperClusterCollection> clusters;
//   e.getByType(clusters);
//   std::cout <<" =================> Treating event "<<e.id()<<" Number of hits "<<pixelHits.product()->size()<<" Number of clusters: "<<clusters.product()->size()<<std::endl;
//   for( SiPixelRecHitCollection::const_iterator MyP= (*pixelHits).begin(); MyP != (*pixelHits).end(); ++MyP) {
//     if ((*MyP).isValid()) {
//       std::cout <<" PixelHit   global  "<<pDD->idToDet((*MyP).geographicalId())->surface().toGlobal((*MyP).localPosition())<<endl;
//       const DetLayer *detl=theGeometricSearchTracker->detLayer((*MyP).geographicalId());

//       if ((*MyP).geographicalId().subdetId()==1) {
// 	const BarrelDetLayer *bdetl = dynamic_cast<const BarrelDetLayer *>(detl);
// 	if (bdetl) {
// 	  std::cout <<" radius "<<bdetl->specificSurface().radius()<<std::endl;
// 	}
// 	else  std::cout<<"Could not downcast!!"<<std::endl;
//       }else {
// 	const ForwardDetLayer *fdetl = dynamic_cast<const ForwardDetLayer *>(detl);
// 	if (!fdetl)
// 	  std::cout<<"Could not downcast!!"<<std::endl;
//       }
//     }
//   }
  //end test
 
 // Find the seeds
   
  std::auto_ptr<ElectronPixelSeedCollection> pOut(new ElectronPixelSeedCollection);
  
  // invoke algorithm
  matcher_->run(e,*pOut);

  // put result into the Event
  e.put(pOut);
}


