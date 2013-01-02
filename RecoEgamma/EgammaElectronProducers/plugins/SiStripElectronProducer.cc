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
// $Id: SiStripElectronProducer.cc,v 1.2 2007/08/28 01:42:29 ratnik Exp $
//

// system include files
#include <memory>
#include <sstream>

// user include files
#include "DataFormats/EgammaCandidates/interface/SiStripElectronFwd.h"
#include "SiStripElectronProducer.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/RoadSearchCloud/interface/RoadSearchCloudCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2DCollection.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 

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
   siStripElectronsLabel_ = iConfig.getParameter<std::string>("siStripElectronsLabel");
   trackCandidatesLabel_ = iConfig.getParameter<std::string>("trackCandidatesLabel");
   produces<reco::SiStripElectronCollection>(siStripElectronsLabel_);
   produces<TrackCandidateCollection>(trackCandidatesLabel_);

   // get parameters
   siHitProducer_ = iConfig.getParameter<std::string>("siHitProducer");
   siRphiHitCollection_ = iConfig.getParameter<std::string>("siRphiHitCollection");
   siStereoHitCollection_ = iConfig.getParameter<std::string>("siStereoHitCollection");
   siMatchedHitCollection_ = iConfig.getParameter<std::string>("siMatchedHitCollection");

   superClusterProducer_ = iConfig.getParameter<std::string>("superClusterProducer");
   superClusterCollection_ = iConfig.getParameter<std::string>("superClusterCollection");
   
   algo_p = new SiStripElectronAlgo(
      iConfig.getParameter<int32_t>("maxHitsOnDetId"),
      iConfig.getParameter<double>("originUncertainty"),
      iConfig.getParameter<double>("phiBandWidth"),      // this is in radians
      iConfig.getParameter<double>("maxNormResid"),
      iConfig.getParameter<int32_t>("minHits"),
      iConfig.getParameter<double>("maxReducedChi2"));

   LogDebug("") << " Welcome to SiStripElectronProducer " ;

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

   edm::Handle<SiStripRecHit2DCollection> rphiHitsHandle;
   iEvent.getByLabel(siHitProducer_, siRphiHitCollection_, rphiHitsHandle);

   edm::Handle<SiStripRecHit2DCollection> stereoHitsHandle;
   iEvent.getByLabel(siHitProducer_, siStereoHitCollection_, stereoHitsHandle);

   edm::Handle<SiStripMatchedRecHit2DCollection> matchedHitsHandle;
   iEvent.getByLabel(siHitProducer_, siMatchedHitCollection_, matchedHitsHandle);

   edm::ESHandle<MagneticField> magneticFieldHandle;
   iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);

   edm::Handle<reco::SuperClusterCollection> superClusterHandle;
   iEvent.getByLabel(superClusterProducer_, superClusterCollection_, superClusterHandle);

   // Set up SiStripElectronAlgo for this event
   algo_p->prepareEvent(trackerHandle, rphiHitsHandle, stereoHitsHandle, matchedHitsHandle, magneticFieldHandle);

   // Prepare the output electron candidates and clouds to be filled
   std::auto_ptr<reco::SiStripElectronCollection> electronOut(new reco::SiStripElectronCollection);
   std::auto_ptr<TrackCandidateCollection> trackCandidateOut(new TrackCandidateCollection);

   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHand;
   iSetup.get<IdealGeometryRecord>().get(tTopoHand);
   const TrackerTopology *tTopo=tTopoHand.product();

   // counter for electron candidates
   int siStripElectCands = 0 ;

   std::ostringstream str;


   // Loop over clusters
   str << "Starting loop over superclusters."<< "\n" << std::endl;
   for (unsigned int i = 0;  i < superClusterHandle.product()->size();  i++) {
      const reco::SuperCluster* sc = &(*reco::SuperClusterRef(superClusterHandle, i));
      double energy = sc->energy();

      if (algo_p->findElectron(*electronOut, *trackCandidateOut, reco::SuperClusterRef(superClusterHandle, i),tTopo)) {
	str << "Supercluster energy: " << energy << ", FOUND an electron." << "\n" << std::endl;
	 ++siStripElectCands ;
      }
      else {
	 str << "Supercluster energy: " << energy << ", DID NOT FIND an electron."<< "\n" << std::endl;
      }
   }
   str << "Ending loop over superclusters." << "\n" << std::endl;
   
   str << " Found " << siStripElectCands 
		    << " SiStripElectron Candidates before track fit " 
		    << "\n" << std::endl ;

   LogDebug("SiStripElectronProducer") << str.str();

   // Put the electron candidates and the tracking trajectories into the event
   iEvent.put(electronOut, siStripElectronsLabel_);
   iEvent.put(trackCandidateOut, trackCandidatesLabel_);
}

//
// const member functions
//

//
// static member functions
//
