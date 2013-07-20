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
// $Id: InterestingTrackEcalDetIdProducer.cc,v 1.2 2013/02/27 19:33:31 eulisse Exp $
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
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      void beginRun(edm::Run const&, const edm::EventSetup&);

      
      // ----------member data ---------------------------
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
  trackCollection_ (iConfig.getParameter<edm::InputTag>("TrackCollection")),
  trackAssociatorPS_ (iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters")),
  minTrackPt_ (iConfig.getParameter<double>("MinTrackPt"))

{
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
   iEvent.getByLabel(trackCollection_,tracks);

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

// ------------ method called once each job just before starting event loop  ------------
void 
InterestingTrackEcalDetIdProducer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
InterestingTrackEcalDetIdProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(InterestingTrackEcalDetIdProducer);
