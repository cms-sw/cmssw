#include "PluginManager/PluginManager.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FastSimulation/EventProducer/interface/FamosProducer.h"
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"

#include "CLHEP/HepMC/GenEvent.h"

#include <iostream>
#include <memory>

FamosProducer::FamosProducer(edm::ParameterSet const & p)      
{    
    produces<edm::HepMCProduct>();
    produces<edm::SimTrackContainer>();
    produces<edm::SimVertexContainer>();
    produces<edm::PSimHitContainer>("TrackerHits");
    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
    produces<edm::PCaloHitContainer>("HcalHits");
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
   using namespace std;

   // Get the generated event(s) from the edm::Event
   Handle<HepMCProduct> evtSource;
   Handle<HepMCProduct> evtVtxSmeared;
   bool source = false;
   bool vtxSmeared = false;
   vector< Handle<HepMCProduct> > evts; 
   iEvent.getManyByType(evts);
   for ( unsigned i=0; i<evts.size(); ++i ) {
     if ( !vtxSmeared && evts[i].provenance()->moduleLabel()=="VtxSmeared" ) {
       vtxSmeared = true;      
       evtVtxSmeared = evts[i];
       break;
     } else if ( !source &&  evts[i].provenance()->moduleLabel()=="source" ) {
       source = true;
       evtSource = evts[i];
     }
   }

   // Take the VtxSmeared if it exists, the source otherwise
   // (The vertex smearing is done in Famos only in the latter case)
   const HepMC::GenEvent* myGenEvent;
   if ( vtxSmeared ) {
     myGenEvent = evtVtxSmeared->GetEvent();
   } else if ( source ) {
     myGenEvent = evtSource->GetEvent();
   } else {
     myGenEvent = 0;
   }

   // .and pass it to the Famos Manager (should be EventManager)
   if ( myGenEvent ) famosManager_->reconstruct(myGenEvent);

   // Put info on to the end::Event
   FSimEvent* fevt = famosManager_->simEvent();

   CalorimetryManager * calo = famosManager_->calorimetryManager();
   TrajectoryManager * tracker = famosManager_->trackerManager();

   std::auto_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
   std::auto_ptr<edm::PSimHitContainer> p3(new edm::PSimHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);
   // For the preshower (not implemented yet)
   //   std::auto_ptr<edm::PCaloHitContainer> p6(new edm::PCaloHitContainer); 
   std::auto_ptr<edm::PCaloHitContainer> p7(new edm::PCaloHitContainer);

   fevt->load(*p1);
   fevt->load(*p2);

   tracker->loadSimHits(*p3);

   if ( calo ) {  
     calo->loadFromEcalBarrel(*p4);
     calo->loadFromEcalEndcap(*p5);
     // calo->loadFromPreshower(*p6);
     calo->loadFromHcal(*p7);
   }

   iEvent.put(p1);
   iEvent.put(p2);
   iEvent.put(p3,"TrackerHits");
   iEvent.put(p4,"EcalHitsEB");
   iEvent.put(p5,"EcalHitsEE");
   // preshower 
   iEvent.put(p7,"HcalHits");

}

DEFINE_FWK_MODULE(FamosProducer);
 
