#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/EventHeader.h"

EventHeader::EventHeader() :
  fEvent( 0 ),
  fLumiBlock( -1 ),
  fRun( -1 ),
  fBx( -1 ),
  fOrbit( -1 ),
  fAvgInstDelLumi( -999. ),
  _Debug( false )
{ }

EventHeader::~EventHeader() {

}

/*  Setup the analysis to put the branch-variables into the tree. */
void EventHeader::setup(edm::ConsumesCollector && iC, TTree* HltTree) {

  fEvent = 0;
  fLumiBlock=-1;
  fRun = -1;
  fBx = -1;
  fOrbit = -1;
  fAvgInstDelLumi = -999.; 

  HltTree->Branch("Event",     &fEvent,       "Event/l");
  HltTree->Branch("LumiBlock", &fLumiBlock,   "LumiBlock/I"); 
  HltTree->Branch("Run",       &fRun,         "Run/I");
  HltTree->Branch("Bx",        &fBx,          "Bx/I"); 
  HltTree->Branch("Orbit",     &fOrbit,       "Orbit/I"); 
  HltTree->Branch("AvgInstDelLumi", &fAvgInstDelLumi, "AvgInstDelLumi/D");

  lumi_Token = iC.consumes<LumiSummary,edm::InLumi>(edm::InputTag("lumiProducer")); 
}

/* **Analyze the event** */
void EventHeader::analyze(edm::Event const& iEvent, TTree* HltTree) {
					
  fEvent 	= iEvent.id().event();
  fLumiBlock    = iEvent.luminosityBlock();
  fRun 		= iEvent.id().run();
  fBx           = iEvent.bunchCrossing();
  fOrbit        = iEvent.orbitNumber();
  
 
  bool lumiException = false;
  const edm::LuminosityBlock& iLumi = iEvent.getLuminosityBlock(); 
  edm::Handle<LumiSummary> lumiSummary; 
  try{
    iLumi.getByToken(lumi_Token, lumiSummary);
    lumiSummary->isValid();
  }
  catch(cms::Exception&){
    lumiException = true;
  }
  if(!lumiException)
    fAvgInstDelLumi = lumiSummary->avgInsDelLumi(); 
  else 
    fAvgInstDelLumi = -999.; 
  
  
  if (_Debug) {	
    std::cout << "EventHeader -- event = "          << fEvent     << std::endl;
    std::cout << "EventHeader -- lumisection = "    << fLumiBlock << std::endl;
    std::cout << "EventHeader -- run   = "          << fRun       << std::endl;
    std::cout << "EventHeader -- bunch crossing = " << fBx        << std::endl; 
    std::cout << "EventHeader -- orbit number = "   << fOrbit     << std::endl; 
  }

}
