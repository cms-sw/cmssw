#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/EventHeader.h"

EventHeader::EventHeader() {

  //set parameter defaults 
  _Debug=false;
}

EventHeader::~EventHeader() {

}

/*  Setup the analysis to put the branch-variables into the tree. */
void EventHeader::setup(TTree* HltTree) {

  fRun = -1;
  fEvent = -1;
  fLumiBlock=-1;
  fBx = -1;
  fOrbit = -1;

  HltTree->Branch("Run",       &fRun,         "Run/I");
  HltTree->Branch("Event",     &fEvent,       "Event/I");
  HltTree->Branch("LumiBlock", &fLumiBlock,   "LumiBlock/I"); 
  HltTree->Branch("Bx",        &fBx,          "Bx/I"); 
  HltTree->Branch("Orbit",     &fOrbit,       "Orbit/I"); 
}

/* **Analyze the event** */
void EventHeader::analyze(edm::Event const& iEvent, TTree* HltTree) {
					
  fRun 		= iEvent.id().run();
  fEvent 	= iEvent.id().event();
  fLumiBlock    = iEvent.luminosityBlock();
  fBx           = iEvent.bunchCrossing();
  fOrbit        = iEvent.orbitNumber();
  
  if (_Debug) {	
    std::cout << "EventHeader -- run   = "          << fRun       << std::endl;
    std::cout << "EventHeader -- event = "          << fEvent     << std::endl;
    std::cout << "EventHeader -- lumisection = "    << fLumiBlock << std::endl; 
    std::cout << "EventHeader -- bunch crossing = " << fBx        << std::endl; 
    std::cout << "EventHeader -- orbit number = "   << fOrbit     << std::endl; 
  }

}
