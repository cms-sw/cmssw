// -*- C++ -*-
//
// Class:      HLTEffCalculator
// 
/**\class HLTEffCalculator HLTEffCalculator.cc DQM/HLTEffCalculator/src/HLTEffCalculator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author: Abideh Jafari 
//         Created:  Wed Aug 01 09:06:32 CEST 2012
// $Id: HLTEfficiencyCalculator.cc,v 1.1 2012/08/01 08:26:04 ajafari Exp $
//
//


#include "HLTriggerOffline/Top/interface/HLTEfficiencyCalculator.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "TFile.h"


HLTEffCalculator::HLTEffCalculator(const edm::ParameterSet& iConfig)

{
  
   
     outputFileName      = iConfig.getParameter<std::string>("OutputFileName");
     HLTresCollection           = iConfig.getParameter<edm::InputTag>("TriggerResCollection");
     verbosity =  iConfig.getUntrackedParameter<int>("verbosity",0);	
     myEffHandler = new  EfficiencyHandler("TopHLTs", iConfig.getParameter<std::vector<std::string> >("hltPaths"), verbosity);   
}

HLTEffCalculator::~HLTEffCalculator()
{
  
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  
}


//
// member functions
//


// ------------ method called to for each event  ------------
void
HLTEffCalculator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
 
  // Trigger 
  Handle<TriggerResults> trh;
  iEvent.getByLabel(HLTresCollection,trh);
  if( ! trh.isValid() ) {
    LogDebug("") << "HL TriggerResults with label ["+HLTresCollection.encode()+"] not found!";
    return;
  }  
  myEffHandler->Fill(iEvent,*trh);
  
}



// ------------ method called once each job just before starting event loop  ------------
void 
HLTEffCalculator::beginJob()
{}

// ------------ method called once each job just after ending the event loop  ------------
void 
HLTEffCalculator::endJob() {  
	TFile * F = new TFile(outputFileName.c_str(),"recreate");
	myEffHandler->WriteAll(F);
	F->Save();
}

