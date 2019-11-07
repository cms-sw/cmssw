#include "GeneratorInterface/GenFilters/plugins/MCDijetResonance.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace edm;
using namespace std;

MCDijetResonance::MCDijetResonance(const edm::ParameterSet& iConfig)
    : token_(consumes<edm::HepMCProduct>(
          edm::InputTag(iConfig.getUntrackedParameter("moduleLabel", std::string("generator")), "unsmeared"))) {
  //here do whatever other initialization is needed

  //Get dijet process type we will select
  dijetProcess = iConfig.getUntrackedParameter<string>("dijetProcess");
  cout << "Dijet Resonance Filter Selecting Process = " << dijetProcess << endl;

  maxQuarkID = 4;  //Maximum |ID| of Light Quark = charm quark gives  u, d, s, and c decays of Zprime.
  bosonID = 21;    //Gluon

  //Number of events and Number Accepted
  nEvents = 0;
  nAccepted = 0;
}

MCDijetResonance::~MCDijetResonance() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void MCDijetResonance::endJob() {
  edm::LogVerbatim("MCDijetResonanceInfo")
      << "================MCDijetResonance report========================================\n"
      << "Events read " << nEvents << " Events accepted " << nAccepted << "\nEfficiency "
      << ((double)nAccepted) / ((double)nEvents)
      << "\n====================================================================" << std::endl;
}

// ------------ method called to skim the data  ------------
bool MCDijetResonance::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  nEvents++;
  cout << endl << "Event=" << nEvents << endl;
  using namespace edm;
  Handle<HepMCProduct> evt;
  iEvent.getByToken(token_, evt);
  const HepMC::GenEvent* myGenEvent = evt->GetEvent();

  //If process is not the desired primary process, cleanup and reject the event.
  if (dijetProcess == "ZprimeLightQuarks" && myGenEvent->signal_process_id() != 141) {
    // Wanted a Z' but didn't find it, so reject event.
    delete myGenEvent;
    return false;
  }
  if (dijetProcess == "QstarQuarkGluon" && myGenEvent->signal_process_id() != 147 &&
      myGenEvent->signal_process_id() != 148) {
    // Wanted a q* but didn't find it, so reject event.
    delete myGenEvent;
    return false;
  }

  //found a dijet resonance

  //debug
  // int count = 0;
  //for ( HepMC::GenEvent::particle_iterator p = myGenEvent->particles_begin();
  //	p != myGenEvent->particles_end() & count<13; ++p )
  // {
  //std::cout << count << ": ID=" << (*p)->pdg_id() << ", status=" << (*p)->status() << ", mass=" << (*p)->momentum().invariantMass() << ", pt=" <<(*p)->momentum().perp() << std::endl;
  //count++;
  //}

  for (HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end();
       ++p) {
    //Find resonance particle and check that it is part of hard collision
    if ((*p)->status() == 3 && ((dijetProcess == "ZprimeLightQuarks" && (*p)->pdg_id() == 32) ||
                                (dijetProcess == "QstarQuarkGluon" && abs((*p)->pdg_id()) == 4000001) ||
                                (dijetProcess == "QstarQuarkGluon" && abs((*p)->pdg_id()) == 4000002))) {
      // The next two particles are the outgoing particles from the resonance decay
      p++;
      int ID1 = (*p)->pdg_id();
      p++;
      int ID2 = (*p)->pdg_id();

      //Check for the process we want
      if ((dijetProcess == "ZprimeLightQuarks" && abs(ID1) <= maxQuarkID && abs(ID2) <= maxQuarkID) ||
          (dijetProcess == "QstarQuarkGluon" && (ID1 == bosonID || ID2 == bosonID))) {
        //cout << "dijet resonance " << dijetProcess << " found " << endl;
        nAccepted++;
        delete myGenEvent;
        return true;
      } else {
        delete myGenEvent;
        return false;
      }
    }
  }

  delete myGenEvent;
  return false;
}
