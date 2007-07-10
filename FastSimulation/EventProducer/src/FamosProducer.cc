#include "FWCore/PluginManager/interface/PluginManager.h"

//#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/Framework/interface/EventSetup.h"
//#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

#include "FastSimulation/EventProducer/interface/FamosProducer.h"
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
//#include "HepMC/FourVector.h"

#include <iostream>
#include <memory>
#include <vector>

FamosProducer::FamosProducer(edm::ParameterSet const & p)      
{    
    produces<edm::HepMCProduct>();
    produces<edm::SimTrackContainer>();
    produces<edm::SimVertexContainer>();
    produces<edm::PSimHitContainer>("TrackerHits");
    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
    produces<edm::PCaloHitContainer>("EcalHitsES");
    produces<edm::PCaloHitContainer>("HcalHits");
    // Temporary facility to allow for the crossing frame to work...
    simulateMuons = p.getParameter<bool>("SimulateMuons");
    if ( simulateMuons ) produces<edm::SimTrackContainer>("MuonSimTracks");
    famosManager_ = new FamosManager(p);

}

FamosProducer::~FamosProducer() 
{ if ( famosManager_ ) delete famosManager_; }

void FamosProducer::beginJob(const edm::EventSetup & es)
{
    std::cout << " FamosProducer initializing " << std::endl;
    famosManager_->setupGeometryAndField(es);
    //    famosManager_->initEventReader();
}
 
void FamosProducer::endJob()
{ 
    std::cout << " FamosProducer terminating " << std::endl; 
}
 
void FamosProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{
   using namespace edm;

   // Get the generated event(s) from the edm::Event
   // 1. Check if a HepMCProduct exists
   //    a. Take the VtxSmeared if it exists
   //    b. Take the source  otherwise
   // 2. Otherwise go for the CandidateCollection
   Handle<HepMCProduct> theHepMCProduct;
   bool genPart = false;
   bool source = false;
   bool vtxSmeared = false;
   std::vector< Handle<reco::CandidateCollection> > genEvts;
   const reco::CandidateCollection* myGenParticles = 0;
   const HepMC::GenEvent* myGenEvent = 0;

   // Look for the GenEvent
   std::vector< Handle<HepMCProduct> > evts; 
   iEvent.getManyByType(evts);
   for ( unsigned i=0; i<evts.size(); ++i ) {
     if (!vtxSmeared && evts[i].provenance()->moduleLabel()=="VtxSmeared") {
       vtxSmeared = true;      
       theHepMCProduct = evts[i];
       break;
     } else if (!source &&  evts[i].provenance()->moduleLabel()=="source") {
       source = true;
       theHepMCProduct = evts[i];
     }
   }
   
   // Take the VtxSmeared if it exists, the source otherwise
   // (The vertex smearing is done in Famos only in the latter case)
   /*
   if ( vtxSmeared ) {
     myGenEvent = evtVtxSmeared->GetEvent();
   } else if ( source ) {
     myGenEvent = evtSource->GetEvent();
   }
   */
   if ( vtxSmeared || source ) myGenEvent = theHepMCProduct->GetEvent();

   if ( !myGenEvent ) { 
     // Look for the particle CandidateCollection
     iEvent.getManyByType(genEvts);
     if ( genEvts.size() ) { 
       for ( unsigned i=0; i<genEvts.size(); ++i ) {
	 if ( genEvts[i].provenance()->moduleLabel()=="genParticleCandidates" )
	   {
	     genPart= true;
	     myGenParticles = &(*genEvts[i]);
	   }
       }
     }
   }

   // .and pass the event to the Famos Manager
   if ( myGenEvent || myGenParticles ) 
     famosManager_->reconstruct(myGenEvent,myGenParticles);
   
   // Put info on to the end::Event
   FSimEvent* fevt = famosManager_->simEvent();
   
   // Set the vertex back to the HepMCProduct (except if it was smeared already)
   if ( myGenEvent ) { 
     HepMC::GenVertex* primaryVertex =  *(myGenEvent->vertices_begin());
     if ( primaryVertex && fabs(primaryVertex->position().z()) > 1e-9 ) {  
       HepMC::FourVector theVertex(fevt->filter().vertex().X()*10.,
				   fevt->filter().vertex().Y()*10.,
				   fevt->filter().vertex().Z()*10.,
				   fevt->filter().vertex().T()*10.);
       theHepMCProduct->applyVtxGen( &theVertex );
     }
   }
   
   CalorimetryManager * calo = famosManager_->calorimetryManager();
   TrajectoryManager * tracker = famosManager_->trackerManager();

   std::auto_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimTrackContainer> m1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
   std::auto_ptr<edm::PSimHitContainer> p3(new edm::PSimHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p6(new edm::PCaloHitContainer); 
   std::auto_ptr<edm::PCaloHitContainer> p7(new edm::PCaloHitContainer);

   fevt->load(*p1,*m1);
   fevt->load(*p2);
   //   fevt->print();
   tracker->loadSimHits(*p3);

   if ( calo ) {  
     calo->loadFromEcalBarrel(*p4);
     calo->loadFromEcalEndcap(*p5);
     calo->loadFromPreshower(*p6);
     calo->loadFromHcal(*p7);
   }

   // Write muon first, to allow tracking particles to work... (pending MixingModule fix)
   if ( simulateMuons ) iEvent.put(m1,"MuonSimTracks");
   iEvent.put(p1);
   iEvent.put(p2);
   iEvent.put(p3,"TrackerHits");
   iEvent.put(p4,"EcalHitsEB");
   iEvent.put(p5,"EcalHitsEE");
   iEvent.put(p6,"EcalHitsES");
   iEvent.put(p7,"HcalHits");

}

DEFINE_FWK_MODULE(FamosProducer);
