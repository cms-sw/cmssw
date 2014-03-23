// -*- C++ -*-
//
// Package:    InterestingTrackEcalDetIdProducer
// Class:      InterestingTrackEcalDetIdProducer
// 
/**\class InterestingTrackEcalDetIdProducer InterestingTrackEcalDetIdProducer.cc Producers/InterestingTrackEcalDetIdProducer/src/InterestingTrackEcalDetIdProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  
//         Created:  Wed Sep 22 17:02:51 CEST 2010
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "DataFormats/TrackReco/interface/Track.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"


//
// class declaration
//

class InterestingTrackEcalDetIdProducer : public edm::EDProducer {
   public:
      explicit InterestingTrackEcalDetIdProducer(const edm::ParameterSet&);
      ~InterestingTrackEcalDetIdProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      void beginRun(edm::Run const&, const edm::EventSetup&) override;

      
      // ----------member data ---------------------------
	  edm::EDGetTokenT<reco::TrackCollection> trackCollectionToken_;
      edm::InputTag trackCollection_;
      edm::ParameterSet trackAssociatorPS_;

      double minTrackPt_;

      const CaloTopology* caloTopology_;
      TrackDetectorAssociator trackAssociator_; 
      TrackAssociatorParameters trackAssociatorParameters_; 


};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
InterestingTrackEcalDetIdProducer::InterestingTrackEcalDetIdProducer(const edm::ParameterSet& iConfig) :

  trackAssociatorPS_ (iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters")),
  minTrackPt_ (iConfig.getParameter<double>("MinTrackPt"))

{
  trackCollectionToken_=
	  consumes<reco::TrackCollection> (iConfig.getParameter<edm::InputTag>("TrackCollection"));	 
  trackAssociator_.useDefaultPropagator();
  trackAssociatorParameters_.loadParameters(trackAssociatorPS_);

  produces<DetIdCollection>(); 
}


InterestingTrackEcalDetIdProducer::~InterestingTrackEcalDetIdProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
InterestingTrackEcalDetIdProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   std::auto_ptr< DetIdCollection > interestingDetIdCollection( new DetIdCollection() ) ;

   // Get tracks from event
   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(trackCollectionToken_,tracks);

   // Loop over tracks
   for(reco::TrackCollection::const_iterator tkItr = tracks->begin(); tkItr != tracks->end(); ++tkItr)
   {
     if(tkItr->pt() < minTrackPt_)
       continue;

     TrackDetMatchInfo info = trackAssociator_.associate( iEvent, iSetup, 
         trackAssociator_.getFreeTrajectoryState(iSetup, *tkItr),
         trackAssociatorParameters_ );

     DetId centerId = info.findMaxDeposition(TrackDetMatchInfo::EcalRecHits);

     if(centerId.rawId()==0)
       continue;

     // Find 5x5 around max
     const CaloSubdetectorTopology* topology = caloTopology_->getSubdetectorTopology(DetId::Ecal,centerId.subdetId());
     const std::vector<DetId>& ids = topology->getWindow(centerId, 5, 5);
     for(std::vector<DetId>::const_iterator idItr = ids.begin(); idItr != ids.end(); ++idItr)
     {
       if(std::find(interestingDetIdCollection->begin(), interestingDetIdCollection->end(), *idItr)
           == interestingDetIdCollection->end())
         interestingDetIdCollection->push_back(*idItr);
     }

   }

   iEvent.put(interestingDetIdCollection);

}

void InterestingTrackEcalDetIdProducer::beginRun(edm::Run const& run, const edm::EventSetup & iSetup)  
{
  edm::ESHandle<CaloTopology> theCaloTopology;
  iSetup.get<CaloTopologyRecord>().get(theCaloTopology);
  caloTopology_ = &(*theCaloTopology); 
}

//define this as a plug-in
DEFINE_FWK_MODULE(InterestingTrackEcalDetIdProducer);
