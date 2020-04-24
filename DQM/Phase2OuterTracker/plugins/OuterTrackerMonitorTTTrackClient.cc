// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerMonitorTTTrackClient.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorTTTrackClient.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Mon, 10 Feb 2014 13:57:08 GMT
// 

// system include files
#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorTTTrackClient.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


//
// constructors and destructor
//
OuterTrackerMonitorTTTrackClient::OuterTrackerMonitorTTTrackClient(const edm::ParameterSet& iConfig)
{
 
}


OuterTrackerMonitorTTTrackClient::~OuterTrackerMonitorTTTrackClient()
{
 

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMonitorTTTrackClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
OuterTrackerMonitorTTTrackClient::beginRun(const edm::Run& run, const edm::EventSetup& es)
{


}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorTTTrackClient::endJob(void) 
{

}

DEFINE_FWK_MODULE(OuterTrackerMonitorTTTrackClient);
