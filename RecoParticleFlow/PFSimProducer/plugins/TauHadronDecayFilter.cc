// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
// #include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoParticleFlow/PFSimProducer/plugins/TauHadronDecayFilter.h"


#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


using namespace edm;
using namespace std;

TauHadronDecayFilter::TauHadronDecayFilter(const edm::ParameterSet& iConfig) {
  //now do what ever initialization is needed

  vertexGenerator_ = iConfig.getParameter<edm::ParameterSet>
    ( "VertexGenerator" );   
  particleFilter_ = iConfig.getParameter<edm::ParameterSet>
    ( "ParticleFilter" );   

  // mySimEvent =  new FSimEvent(vertexGenerator_, particleFilter_);
  mySimEvent =  new FSimEvent( particleFilter_);
}


TauHadronDecayFilter::~TauHadronDecayFilter() {
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  delete mySimEvent;

}


bool
TauHadronDecayFilter::filter(edm::Event& iEvent, 
			     const edm::EventSetup& iSetup) {

  
  Handle<vector<SimTrack> > simTracks;
  iEvent.getByLabel("g4SimHits",simTracks);
  Handle<vector<SimVertex> > simVertices;
  iEvent.getByLabel("g4SimHits",simVertices);
   
  mySimEvent->fill( *simTracks, *simVertices );
 
  if( mySimEvent->nTracks() >= 2 ) {
    FSimTrack& gene = mySimEvent->track(0); 
    if( abs(gene.type()) != 15) { 
      // first particle is not a tau. 
      // -> do not filter
      return true;
    }
    
    FSimTrack& decayproduct = mySimEvent->track(1);
    switch( abs(decayproduct.type() ) ) {
    case 11: // electrons
    case 13: // muons 
      LogWarning("PFProducer")
	<<"TauHadronDecayFilter: selecting single tau events with hadronic decay."<<endl;
      // mySimEvent->print();
      return false;
    default:
      return true;
    }
  }
  
  // more than 2 particles
  return true;
}

void
TauHadronDecayFilter::beginRun(const edm::Run& run,
			       const edm::EventSetup& es) {
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  // edm::ESHandle < DefaultConfig::ParticleDataTable > pdt;

  es.getData(pdt);
  if ( !ParticleTable::instance() ) 
    ParticleTable::instance(&(*pdt));
  mySimEvent->initializePdt(&(*pdt));

}

