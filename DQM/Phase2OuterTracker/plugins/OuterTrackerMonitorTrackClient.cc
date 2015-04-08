// -*- C++ -*-
//
// Package:    Phase2OuterTracker
// Class:      Phase2OuterTracker
// 
/**\class Phase2OuterTracker OuterTrackerMonitorTrackClient.cc DQM/Phase2OuterTracker/plugins/OuterTrackerMonitorTrackClient.cc

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
#include <fstream>
#include <math.h>
#include "TNamed.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/Phase2OuterTracker/interface/OuterTrackerMonitorTrackClient.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

#include "TMath.h"
#include <iostream>

//
// constructors and destructor
//
OuterTrackerMonitorTrackClient::OuterTrackerMonitorTrackClient(const edm::ParameterSet& iConfig)
{
 
}


OuterTrackerMonitorTrackClient::~OuterTrackerMonitorTrackClient()
{
 

}


//
// member functions
//

// ------------ method called for each event  ------------
void
OuterTrackerMonitorTrackClient::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
}


// ------------ method called once each job just before starting event loop  ------------
void 
OuterTrackerMonitorTrackClient::beginRun(const edm::Run& run, const edm::EventSetup& es)
{


}//end of method


// ------------ method called once each job just after ending the event loop  ------------
void 
OuterTrackerMonitorTrackClient::endJob(void) 
{

}

DEFINE_FWK_MODULE(OuterTrackerMonitorTrackClient);
