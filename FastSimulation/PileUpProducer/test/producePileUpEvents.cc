// user include files
//#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "FastSimulation/Event/interface/FSimEvent.h"
#include "FastSimulation/Event/interface/FSimTrack.h"
//#include "FastSimulation/Event/interface/FSimVertex.h"
#include "FastSimulation/Particle/interface/ParticleTable.h"
#include "FastSimDataFormats/PileUpEvents/interface/PUEvent.h"

#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TProcessID.h"

class producePileUpEvents : public edm::stream::EDProducer <>{

public :
  explicit producePileUpEvents(const edm::ParameterSet&);
  ~producePileUpEvents();

  virtual void produce(edm::Event&, const edm::EventSetup& ) override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const& ) override;
private:
  
  // See RecoParticleFlow/PFProducer/interface/PFProducer.h
  edm::ParameterSet particleFilter_;
  FSimEvent* mySimEvent;
  PUEvent* puEvent;
  TTree* puTree;
  TFile* outFile;
  int ObjectNumber;
  //  DaqMonitorBEInterface * dbe;
  std::string PUEventFileName;
  bool savePU;
  unsigned sizePU;
  int totalPU;

};

producePileUpEvents::producePileUpEvents(const edm::ParameterSet& p) :
  mySimEvent(0),
  savePU(false),
  sizePU(0),
  totalPU(0)
{
  
  // This producer produce a vector of SimTracks (actually not, but the line is needed)
  produces<edm::SimTrackContainer>();

  // Let's just initialize the SimEvent's
  particleFilter_ = p.getParameter<edm::ParameterSet> ( "PUParticleFilter" );   

  // Do we save the minbias events?
  savePU = p.getParameter<bool>("SavePileUpEvents");
  if ( savePU ) { 
    std::cout << "Pile-Up Events will be saved ! ";
    sizePU = p.getParameter<unsigned int>("BunchPileUpEventSize");
    std::cout << " by bunch of " << sizePU << " events" << std::endl;
  } else {
    std::cout << "Pile-Up Events won't be saved ! " << std::endl;
  }

  // For a proper format
  mySimEvent = new FSimEvent(particleFilter_);

  // Where the pile-up events are saved;
  PUEventFileName = "none";
  if ( savePU ) { 

    puEvent = new PUEvent();
  
    PUEventFileName = 
      p.getUntrackedParameter<std::string>("PUEventFile","MinBiasEventsTest.root");
    outFile = new TFile(PUEventFileName.c_str(),"recreate");

    // Open the tree
    puTree = new TTree("MinBiasEvents","");
    puTree->Branch("puEvent","PUEvent",&puEvent,32000,99);

  }

  // ObjectNumber
  ObjectNumber = -1;
    
  // ... and the histograms
  //  dbe = edm::Service<DaqMonitorBEInterface>().operator->();
								
}

producePileUpEvents::~producePileUpEvents()
{
  //  dbe->save(outputFileName);

  if ( savePU ) {
 
    outFile->cd();
    // Fill the last (incomplete) puEvent
    if ( puEvent->nMinBias() != 0 ) { 
      puTree->Fill();
      std::cout << "Saved " << puEvent->nMinBias() 
		<< " MinBias Event(s) with " << puEvent->nParticles()
		<< " Particles in total " << std::endl;
    }
    // Conclude the writing on disk
    puTree->Write();
    // Print information
    puTree->Print();
    // And tidy up everything!
    //  outFile->Close();
    delete puEvent;
    delete puTree;
    delete outFile;

  }
  
  //  delete mySimEvent;
}

void producePileUpEvents::beginRun(edm::Run const&, edm::EventSetup const& es)
{

  // init Particle data table (from Pythia)
  edm::ESHandle < HepPDT::ParticleDataTable > pdt;
  es.getData(pdt);
  
  mySimEvent->initializePdt(&(*pdt));

}

void
producePileUpEvents::produce(edm::Event& iEvent, const edm::EventSetup& iSetup )
{
  ParticleTable::Sentry(mySimEvent->theTable());
  ++totalPU;
  if ( totalPU/1000*1000 == totalPU ) 
    std::cout << "Number of events produced "
	      << totalPU << std::endl; 
  
  // Get the generated event(s) from the edm::Event
  edm::Handle< edm::HepMCProduct> evtSource;
  std::vector< edm::Handle< edm::HepMCProduct> > evts; 
  iEvent.getManyByType(evts);
  for ( unsigned i=0; i<evts.size(); ++i ) 
    if ( evts[i].provenance()->moduleLabel()=="generator" ) evtSource = evts[i];
  
  // Take the VtxSmeared if it exists, the source otherwise
  // (The vertex smearing is done in Famos only in the latter case)
  const HepMC::GenEvent* myGenEvent = evtSource->GetEvent();
  edm::EventID id(1,1,totalPU);

  mySimEvent->fill(*myGenEvent,id);
  
  //  mySimEvent->print();
  
  if ( savePU ) { 

    PUEvent::PUMinBiasEvt minBiasEvt;
    minBiasEvt.first = puEvent->nParticles();
    minBiasEvt.size = mySimEvent->nTracks();
    puEvent->addPUMinBiasEvt(minBiasEvt);
    //    std::cout << "Minbias number " << puEvent->nMinBias() << std::endl;

    for ( unsigned itrack=0; itrack<mySimEvent->nTracks(); ++itrack ) { 

      FSimTrack& myTrack = mySimEvent->track(itrack);
      PUEvent::PUParticle particle;
      particle.px = myTrack.momentum().px();
      particle.py = myTrack.momentum().py();
      particle.pz = myTrack.momentum().pz();
      particle.mass = myTrack.particleInfo()->mass().value();
      particle.id = myTrack.type();
      puEvent->addPUParticle(particle);
      //      std::cout << "Particle number " << puEvent->nParticles() << std::endl;

    }


    // Save the object number count for a new NUevent
    if ( ObjectNumber == -1 || puEvent->nMinBias() == sizePU ) {
      ObjectNumber = TProcessID::GetObjectCount();
    }
    
    
    // Save the tracks from the minbias event bunch
    //      std::cout << "Number of MinBias event in puEvent = "
    //		<< puEvent->nMinBias() << std::endl;
    if ( puEvent->nMinBias() == sizePU ) { 
      // Reset Event object count to avoid memory overflows
      TProcessID::SetObjectCount(ObjectNumber);
      // Save the puEvent
      std::cout << "Saved " << puEvent->nMinBias() 
		<< " MinBias Event(s) with " << puEvent->nParticles()
		<< " Particles in total " << std::endl;
      outFile->cd(); 
      puTree->Fill();
      //	puTree->Print();
      puEvent->reset();
      
    }
    
  }


}

//define this as a plug-in

DEFINE_FWK_MODULE(producePileUpEvents);
