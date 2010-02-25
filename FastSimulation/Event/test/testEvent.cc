// user include files
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

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <vector>
#include <string>

class testEvent : public edm::EDAnalyzer {
public :
  explicit testEvent(const edm::ParameterSet&);
  ~testEvent();

  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  virtual void beginRun(edm::Run const&, edm::EventSetup const&  );
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  bool isGeant;
  edm::ParameterSet particleFilter_;
  std::vector<FSimEvent*> mySimEvent;

  // Histograms
  DQMStore * dbe;
  std::vector<MonitorElement*> PIDs;
  std::vector<MonitorElement*> Energies;

};

testEvent::testEvent(const edm::ParameterSet& p) :
  isGeant(true),
  mySimEvent(2, static_cast<FSimEvent*>(0)),
  PIDs(2,static_cast<MonitorElement*>(0)),
  Energies(2,static_cast<MonitorElement*>(0))
{
  
  particleFilter_ = p.getParameter<edm::ParameterSet> ( "ParticleFilter" );
  isGeant = p.getParameter<bool>("GeantInfo");

  // For the full sim
  if ( isGeant) mySimEvent[0] = new FSimEvent(particleFilter_);
  // For the fast sim
  mySimEvent[1] = new FSimEvent(particleFilter_);
  
  dbe = edm::Service<DQMStore>().operator->();
  PIDs[0] = dbe->book1D("PIDFull", "Particle ID distribution (full)",6000,-6000.,6000.);
  PIDs[1] = dbe->book1D("PIDFast", "Particle ID distribution (fast)",6000,-6000.,6000.);
  Energies[0] = dbe->book1D("EneFull", "Energy distribution (full)",20,0.,20.);
  Energies[1] = dbe->book1D("EneFast", "Energy distribution (fast)",20,0.,20.);
								
}

testEvent::~testEvent()
{
  dbe->save("testEvent.root");
}

void testEvent::beginRun(edm::Run const&, edm::EventSetup const& es)
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


      for ( unsigned fsimi=0; fsimi < mySimEvent[ievt]->nTracks(); ++fsimi ) {
	const FSimTrack& theTrack = mySimEvent[ievt]->track(fsimi);
	
	if ( theTrack.momentum().Perp2() < 1. ) continue;
	if ( fabs(theTrack.momentum().Eta()) > 3. ) continue;
	if ( theTrack.noEndVertex() || theTrack.endVertex().position().Perp2() > 5. ) { 
	  int Type = theTrack.type();
	  PIDs[ievt]->Fill(Type);
	  double bin;
	  double en = theTrack.momentum().E();
	  if      ( en <   0.8 ) bin =  0.5; 
	  else if ( en <   1.5 ) bin =  1.5; 
	  else if ( en <   2.5 ) bin =  2.5; 
	  else if ( en <   3.5 ) bin =  3.5; 
	  else if ( en <   4.5 ) bin =  4.5; 
	  else if ( en <   6.0 ) bin =  5.5; 
	  else if ( en <   8.0 ) bin =  6.5; 
	  else if ( en <  10.5 ) bin =  7.5; 
	  else if ( en <  13.5 ) bin =  8.5; 
	  else if ( en <  17.5 ) bin =  9.5; 
	  else if ( en <  25.0 ) bin = 10.5; 
	  else if ( en <  40.0 ) bin = 11.5; 
	  else if ( en <  75.0 ) bin = 12.5; 
	  else if ( en < 150.0 ) bin = 13.5; 
	  else if ( en < 250.0 ) bin = 14.5; 
	  else if ( en < 400.0 ) bin = 15.5; 
	  else if ( en < 600.0 ) bin = 16.5; 
	  else if ( en < 850.0 ) bin = 17.5; 
	  else if ( en < 1500. ) bin = 18.5; 
	  else                   bin = 19.5; 
	  Energies[ievt]->Fill(bin);
	}
      }

    }

  }

}

//define this as a plug-in

DEFINE_FWK_MODULE(testEvent);
