#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/EventHeader.h"

EventHeader::EventHeader() :
  fRun( -1 ),
  fEvent( -1 ),
  fLumiBlock( -1 ),
  fBx( -1 ),
  fOrbit( -1 ),
  fAvgInstDelLumi( -999. ),
  _Debug( false )
{ }

EventHeader::~EventHeader() {

}

/*  Setup the analysis to put the branch-variables into the tree. */
void EventHeader::setup(TTree* HltTree) {

  fRun = -1;
  fEvent = -1;
  fLumiBlock=-1;
  fBx = -1;
  fOrbit = -1;
  fAvgInstDelLumi = -999.; 

  HltTree->Branch("Run",       &fRun,         "Run/I");
  HltTree->Branch("Event",     &fEvent,       "Event/I");
  HltTree->Branch("LumiBlock", &fLumiBlock,   "LumiBlock/I"); 
  HltTree->Branch("Bx",        &fBx,          "Bx/I"); 
  HltTree->Branch("Orbit",     &fOrbit,       "Orbit/I"); 
  HltTree->Branch("AvgInstDelLumi", &fAvgInstDelLumi, "AvgInstDelLumi/D");
}

/* **Analyze the event** */
void EventHeader::analyze(edm::Event const& iEvent, const edm::ESHandle<LumiCorrectionParam> & lumicorrdatahandle, TTree* HltTree) {
  fRun 		= iEvent.id().run();
  fEvent 	= iEvent.id().event();
  fLumiBlock    = iEvent.luminosityBlock();
  fBx           = iEvent.bunchCrossing();
  fOrbit        = iEvent.orbitNumber();
  
 
  bool lumiException = false;
  const edm::LuminosityBlock& iLumi = iEvent.getLuminosityBlock(); 
  edm::Handle<LumiSummary> lumiSummary; 
  try{
    iLumi.getByLabel("lumiProducer", lumiSummary); 
    lumiSummary->isValid();
  }
  catch(cms::Exception&){
    lumiException = true;
  }
  if(!lumiException)
    {
      // Raw delivered lumi
      fAvgInstDelLumi = lumiSummary->avgInsDelLumi(); 

      // Now apply lumi corrections per LumiCalc#Luminosity_Objects_in_EDM_and_lu twiki
      float instlumi = fAvgInstDelLumi;
      float corrfac=1.;
      if(lumicorrdatahandle.isValid()){
	const LumiCorrectionParam* mydata=lumicorrdatahandle.product();
	corrfac=mydata->getCorrection(instlumi);
	fAvgInstDelLumi= corrfac * instlumi;
      }
    }
  else 
    fAvgInstDelLumi = -999.; 
  
  if (_Debug) {	
    std::cout << "EventHeader -- run   = "          << fRun       << std::endl;
    std::cout << "EventHeader -- event = "          << fEvent     << std::endl;
    std::cout << "EventHeader -- lumisection = "    << fLumiBlock << std::endl; 
    std::cout << "EventHeader -- bunch crossing = " << fBx        << std::endl; 
    std::cout << "EventHeader -- orbit number = "   << fOrbit     << std::endl; 
  }

}
