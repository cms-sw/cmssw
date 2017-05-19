// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerMonitorTTStubClient.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorTTStubClient.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Isabelle Helena J De Bruyn
//         Created:  Mon, 14 Nov 2014 12:07:38 GMT
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
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorTTStubClient.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"


//
// constructors and destructor
//
OuterTrackerMonitorTTStubClient::OuterTrackerMonitorTTStubClient(const edm::ParameterSet& iConfig)
{
 
}


OuterTrackerMonitorTTStubClient::~OuterTrackerMonitorTTStubClient()
{
 

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMonitorTTStubClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
OuterTrackerMonitorTTStubClient::beginRun(const edm::Run& run, const edm::EventSetup& es)
{


}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorTTStubClient::endJob(void) 
{

}

DEFINE_FWK_MODULE(OuterTrackerMonitorTTStubClient);
