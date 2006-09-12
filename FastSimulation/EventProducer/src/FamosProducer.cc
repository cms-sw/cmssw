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

   // Get the generated event from the edm::Event
   Handle<HepMCProduct> evt;
   iEvent.getByType(evt);
   const HepMC::GenEvent* myGenEvent = evt->GetEvent();

   // .and pass it to the Famos Manager (should be EventManager)
   famosManager_->reconstruct(myGenEvent);

   // Put info on to the end::Event
   FSimEvent* fevt = famosManager_->simEvent();

   CalorimetryManager * calo = famosManager_->calorimetryManager();
   TrajectoryManager * tracker = famosManager_->trackerManager();

   std::auto_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
   std::auto_ptr<edm::PSimHitContainer> p3(new edm::PSimHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);

   fevt->load(*p1);
   fevt->load(*p2);

   tracker->loadSimHits(*p3);

   if ( calo ) {  
     calo->loadFromEcalBarrel(*p4);
     calo->loadFromHcal(*p5);
   }

   iEvent.put(p1);
   iEvent.put(p2);
   iEvent.put(p3,"TrackerHits");
   iEvent.put(p4,"EcalHitsEB");
   iEvent.put(p5,"HcalHits");

}

DEFINE_FWK_MODULE(FamosProducer)
 
