#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "FastSimDataFormats/NuclearInteractions/interface/FSimVertexTypeFwd.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Provenance/interface/EventID.h"

#include "FastSimulation/EventProducer/interface/FamosProducer.h"
#include "FastSimulation/EventProducer/interface/FamosManager.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/KineParticleFilter.h"
#include "FastSimulation/Event/interface/PrimaryVertexGenerator.h"
#include "FastSimulation/Calorimetry/interface/CalorimetryManager.h"
#include "FastSimulation/TrajectoryManager/interface/TrajectoryManager.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"

#include <iostream>
#include <memory>
#include <vector>

FamosProducer::FamosProducer(edm::ParameterSet const & p)      
{    

    produces<edm::SimTrackContainer>();
    produces<edm::SimVertexContainer>();
    produces<FSimVertexTypeCollection>("VertexTypes");
    produces<edm::PSimHitContainer>("TrackerHits");
    produces<edm::PCaloHitContainer>("EcalHitsEB");
    produces<edm::PCaloHitContainer>("EcalHitsEE");
    produces<edm::PCaloHitContainer>("EcalHitsES");
    produces<edm::PCaloHitContainer>("HcalHits");
    // Temporary facility to allow for the crossing frame to work...
    simulateMuons = p.getParameter<bool>("SimulateMuons");
    if ( simulateMuons ) produces<edm::SimTrackContainer>("MuonSimTracks");

    // The generator input label
    theSourceLabel = p.getParameter<edm::InputTag>("SourceLabel");
    theGenParticleLabel = p.getParameter<edm::InputTag>("GenParticleLabel");
    theBeamSpotLabel = p.getParameter<edm::InputTag>("BeamSpotLabel");

    famosManager_ = new FamosManager(p);
  
}

FamosProducer::~FamosProducer() 
{ if ( famosManager_ ) delete famosManager_; }

void
FamosProducer::beginRun(edm::Run & run, const edm::EventSetup & es) {
  famosManager_->setupGeometryAndField(run,es);
}
 
void FamosProducer::endJob()
{ 
}
 
void FamosProducer::produce(edm::Event & iEvent, const edm::EventSetup & es)
{
   using namespace edm;

  //  // The beam spot position
   edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
   iEvent.getByLabel(theBeamSpotLabel,recoBeamSpotHandle); 
   math::XYZPoint BSPosition_ = recoBeamSpotHandle->position();

   //Retrieve tracker topology from geometry
   edm::ESHandle<TrackerTopology> tTopoHand;
   es.get<IdealGeometryRecord>().get(tTopoHand);
   const TrackerTopology *tTopo=tTopoHand.product();

      
   const HepMC::GenEvent* myGenEvent = 0;
   FSimEvent* fevt = famosManager_->simEvent();
   //   fevt->setBeamSpot(BSPosition_);     
   
   // Get the generated event(s) from the edm::Event
   // 1. Check if a HepMCProduct exists
   //    a. Take the VtxSmeared if it exists
   //    b. Take the source  otherwise
   // 2. Otherwise go for the CandidateCollection
   
   Handle<HepMCProduct> theHepMCProduct;
   
   PrimaryVertexGenerator* theVertexGenerator = fevt->thePrimaryVertexGenerator();
   
   
   const reco::GenParticleCollection* myGenParticlesXF = 0;
   const reco::GenParticleCollection* myGenParticles = 0;
   const HepMC::GenEvent* thePUEvents = 0;

   Handle<CrossingFrame<HepMCProduct> > theHepMCProductCrossingFrame;
   bool isPileUpXF = iEvent.getByLabel("mixGenPU","generator",theHepMCProductCrossingFrame);

   if (isPileUpXF){// take the GenParticle from crossingframe event collection, if it exists 
     Handle<reco::GenParticleCollection> genEvtXF;
     bool genPartXF = iEvent.getByLabel("genParticlesFromMixingModule",genEvtXF);
     if(genPartXF) myGenParticlesXF = &(*genEvtXF);
   }
   else{// otherwise, use the old famos PU     
     // Get the generated signal event
     bool source = iEvent.getByLabel(theSourceLabel,theHepMCProduct);
     if ( source ) { 
       myGenEvent = theHepMCProduct->GetEvent();
       // First rotate in case of beam crossing angle (except if done already)
       if ( theVertexGenerator ) { 
	 TMatrixD* boost = theVertexGenerator->boost();
	 if ( boost ) theHepMCProduct->boostToLab(boost,"momentum");
       }          
       myGenEvent = theHepMCProduct->GetEvent();
     } 

     fevt->setBeamSpot(BSPosition_);
       
     //     In case there is no HepMCProduct, seek a genParticle Candidate Collection
     bool genPart = false;
     if ( !myGenEvent ) { 
       // Look for the particle CandidateCollection
       Handle<reco::GenParticleCollection> genEvt;
       genPart = iEvent.getByLabel(theGenParticleLabel,genEvt);
       if ( genPart ) myGenParticles = &(*genEvt);
     }
       
     if ( !myGenEvent && !genPart )
       std::cout << "There is no generator input for this event, under " 
		 << "any form (HepMCProduct, genParticles)" << std::endl
		 << "Please check SourceLabel or GenParticleLabel" << std::endl;
       
     // Get the pile-up events from the pile-up producer
     // There might be no pile-up events, by the way, in that case, just continue
     Handle<HepMCProduct> thePileUpEvents;
     bool isPileUp = iEvent.getByLabel("famosPileUp","PileUpEvents",thePileUpEvents);
     thePUEvents = isPileUp ? thePileUpEvents->GetEvent() : 0;
      
   }//end else

   // pass the event to the Famos Manager for propagation and simulation
   if (myGenParticlesXF) {
     famosManager_->reconstruct(myGenParticlesXF,tTopo);
   } else {
     famosManager_->reconstruct(myGenEvent,myGenParticles,thePUEvents,tTopo);
   }

   // Set the vertex back to the HepMCProduct (except if it was smeared already)
   if ( myGenEvent ) { 
     if ( theVertexGenerator ) { 
       HepMC::FourVector theVertex(
				   (theVertexGenerator->X()-theVertexGenerator->beamSpot().X()+BSPosition_.X())*10.,
				   (theVertexGenerator->Y()-theVertexGenerator->beamSpot().Y()+BSPosition_.Y())*10.,
				   (theVertexGenerator->Z()-theVertexGenerator->beamSpot().Z()+BSPosition_.Z())*10.,
				   0.);
       if ( fabs(theVertexGenerator->Z()) > 1E-10 ) theHepMCProduct->applyVtxGen( &theVertex );
     }
   }
   
   CalorimetryManager * calo = famosManager_->calorimetryManager();
   TrajectoryManager * tracker = famosManager_->trackerManager();
   
   // Save everything in the edm::Event
   std::auto_ptr<edm::SimTrackContainer> p1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimTrackContainer> m1(new edm::SimTrackContainer);
   std::auto_ptr<edm::SimVertexContainer> p2(new edm::SimVertexContainer);
   std::auto_ptr<FSimVertexTypeCollection> v1(new FSimVertexTypeCollection);
   std::auto_ptr<edm::PSimHitContainer> p3(new edm::PSimHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p4(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p5(new edm::PCaloHitContainer);
   std::auto_ptr<edm::PCaloHitContainer> p6(new edm::PCaloHitContainer); 
   std::auto_ptr<edm::PCaloHitContainer> p7(new edm::PCaloHitContainer);
   
   fevt->load(*p1,*m1);
   fevt->load(*p2);
   fevt->load(*v1);
   //   fevt->print();
   tracker->loadSimHits(*p3);

   //   fevt->print();

   if ( calo ) {  
     calo->loadFromEcalBarrel(*p4);
     calo->loadFromEcalEndcap(*p5);
     calo->loadFromPreshower(*p6);
     calo->loadFromHcal(*p7);
     // update the muon SimTracks
     calo->loadMuonSimTracks(*m1);
   }

   // Write muon first, to allow tracking particles to work... (pending MixingModule fix)
   if ( simulateMuons ) iEvent.put(m1,"MuonSimTracks");
   iEvent.put(p1);
   iEvent.put(p2);
   iEvent.put(p3,"TrackerHits");
   iEvent.put(v1,"VertexTypes");
   iEvent.put(p4,"EcalHitsEB");
   iEvent.put(p5,"EcalHitsEE");
   iEvent.put(p6,"EcalHitsES");
   iEvent.put(p7,"HcalHits");

}

DEFINE_FWK_MODULE(FamosProducer);

