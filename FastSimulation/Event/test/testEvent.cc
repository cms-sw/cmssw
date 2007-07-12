// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimulation/Utilities/interface/Histos.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>
#include "TH2.h"
#include "TFile.h"
#include "TCanvas.h"

class testEvent : public edm::EDAnalyzer {
public :
  explicit testEvent(const edm::ParameterSet&);
  ~testEvent();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginJob(const edm::EventSetup & c);
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  bool isGeant;
  edm::ParameterSet particleFilter_;
  std::vector<FSimEvent*> mySimEvent;

};

testEvent::testEvent(const edm::ParameterSet& p) :
  isGeant(true),
  mySimEvent(2, static_cast<FSimEvent*>(0))
{
  
  particleFilter_ = p.getParameter<edm::ParameterSet> ( "ParticleFilter" );
  isGeant = p.getParameter<bool>("GeantInfo");

  // For the full sim
  if ( isGeant) mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);
  
								
}

testEvent::~testEvent()
{
}

void testEvent::beginJob(const edm::EventSetup & es)
{
  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  if ( !ParticleTable::instance() ) ParticleTable::instance(&(*pdt));
  if ( isGeant ) mySimEvent[0]->initializePdt(&(*pdt));
  mySimEvent[1]->initializePdt(&(*pdt));

}

void
testEvent::analyze( const edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  if ( isGeant ) { 
    std::cout << "Fill full event " << std::endl;
    edm::Handle<std::vector<SimTrack> > fullSimTracks;
    iEvent.getByLabel("g4SimHits","",fullSimTracks);
    edm::Handle<std::vector<SimVertex> > fullSimVertices;
    iEvent.getByLabel("g4SimHits","",fullSimVertices);
    mySimEvent[0]->fill( *fullSimTracks, *fullSimVertices );
  }

  //  for ( unsigned i=0; i< (*fullSimTracks).size(); ++i ) { 
  //    std::cout << (*fullSimTracks)[i] << std::endl;
  //  }
  
  /* */
  std::cout << "Fill fast event " << std::endl;
  edm::Handle<std::vector<SimTrack> > fastSimTracks;
  iEvent.getByLabel("famosSimHits","",fastSimTracks);
  edm::Handle<std::vector<SimVertex> > fastSimVertices;
  iEvent.getByLabel("famosSimHits","",fastSimVertices);

  mySimEvent[1]->fill( *fastSimTracks, *fastSimVertices );
  /* */
  
  for ( unsigned ievt=0; ievt<2; ++ievt ) {

    if ( isGeant || ievt == 1 ) { 
      if ( isGeant ) std::cout << "Event number " << ievt << std::endl;
      mySimEvent[ievt]->print();
    }

  }

}

//define this as a plug-in
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(testEvent);
