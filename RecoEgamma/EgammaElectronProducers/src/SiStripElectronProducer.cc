// -*- C++ -*-
//
// Package:     EgammaElectronProducers
// Class  :     SiStripElectronProducer
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Fri May 26 16:11:30 EDT 2006
// $Id: SiStripElectronProducer.cc,v 1.2 2006/06/02 22:43:13 pivarski Exp $
//

// system include files
#include <memory>

// user include files
#include "RecoEgamma/EgammaElectronProducers/interface/SiStripElectronProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DLocalPosCollection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiStripElectronProducer::SiStripElectronProducer(const edm::ParameterSet& iConfig)
{
   // register your products
   produces<reco::SiStripElectronCandidateCollection>("SiStripElectronCandidate");
   produces<TrackCandidateCollection>("");

   // get parameters
   siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
   siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
   siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");

   superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
   superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");
   
   algo_p = new SiStripElectronAlgo(
      iConfig.getParameter<int32_t>("maxHitsOnDetId"),
      iConfig.getParameter<double>("originUncertainty"),
      iConfig.getParameter<double>("phiBandWidth"),      // this is in radians
      iConfig.getParameter<int32_t>("minHits"),
      iConfig.getParameter<double>("maxReducedChi2"));
}


// SiStripElectronProducer::SiStripElectronProducer(const SiStripElectronProducer& rhs)
// {
//    // do actual copying here;
// }

SiStripElectronProducer::~SiStripElectronProducer()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   delete algo_p;
}

//
// assignment operators
//
// const SiStripElectronProducer& SiStripElectronProducer::operator=(const SiStripElectronProducer& rhs)
// {
//   //An exception safe implementation is
//   SiStripElectronProducer temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

// ------------ method called to produce the data  ------------
void
SiStripElectronProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Extract data from the event
   edm::ESHandle<TrackerGeometry> trackerHandle;
   iSetup.get<TrackerDigiGeometryRecord>().get(trackerHandle);

   edm::Handle<SiStripRecHit2DLocalPosCollection> rphiHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);

   edm::Handle<SiStripRecHit2DLocalPosCollection> stereoHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, stereoHitsHandle);

   edm::ESHandle<MagneticField> magneticFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);

   edm::Handle<reco::SuperClusterCollection> superClusterHandle;
   iEvent.getByLabel(superClusterProducer_, superClusterCollection_, superClusterHandle);

   // Set up SiStripElectronAlgo for this event
   algo_p->prepareEvent(trackerHandle.product(), rphiHitsHandle.product(), stereoHitsHandle.product(), magneticFieldHandle.product());

   // Prepare the output electron candidates and clouds to be filled
   std::auto_ptr<reco::SiStripElectronCandidateCollection> electronOut(new reco::SiStripElectronCandidateCollection);
   std::auto_ptr<TrackCandidateCollection> trackCandidateOut(new TrackCandidateCollection);

   // Loop over clusters
   edm::LogInfo("SiStripElectronProducer") << "Starting loop over superclusters." << std::endl;
   for (reco::SuperClusterCollection::const_iterator superClusterIter = superClusterHandle->begin();
	superClusterIter != superClusterHandle->end();
	++superClusterIter) {

      int electronCandidates = algo_p->findElectron(*electronOut, *trackCandidateOut, *superClusterIter);

      edm::LogInfo("SiStripElectronProducer") << "We found " << electronCandidates << " potential electrons associated with this supercluster." << std::endl;
   }
   edm::LogInfo("SiStripElectronProducer") << "Ending loop over superclusters." << std::endl;

   // Put the electron candidates and the tracking trajectories into the event
   iEvent.put(electronOut, "SiStripElectronCandidate");
   iEvent.put(trackCandidateOut, "");
}

//
// const member functions
//

//
// static member functions
//
