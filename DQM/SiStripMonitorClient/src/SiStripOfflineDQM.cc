// -*- C++ -*-
//
// Package:    SiStripMonitorCluster
// Class:      SiStripOfflineDQM
// 
/**\class SiStripOfflineDQM SiStripOfflineDQM.cc DQM/SiStripMonitorCluster/src/SiStripOfflineDQM.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct  5 16:42:34 CET 2006
// $Id: SiStripOfflineDQM.cc,v 1.4 2007/02/24 15:39:23 samvel Exp $
//
//

// Root UI that is used by original Client's SiStripActionExecuter
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "DQM/SiStripMonitorClient/interface/SiStripOfflineDQM.h"

using edm::LogInfo;

SiStripOfflineDQM::SiStripOfflineDQM( const edm::ParameterSet &roPARAMETER_SET)
  : bVerbose( roPARAMETER_SET.getUntrackedParameter<bool>( "bVerbose")),
    bSaveInFile( roPARAMETER_SET.getUntrackedParameter<bool>( "bOutputMEsInRootFile")),
    oFILE_NAME( roPARAMETER_SET.getUntrackedParameter<std::string>( "oOutputFile")),
    //    nQTEST_EVENTS_DELAY_( roPARAMETER_SET.getUntrackedParameter<int>( "nQTestEventsDelay")),
    poMui( new MonitorUIRoot()) {

  // Create MessageSender
  LogInfo( "SiStripOfflineDQM");
}

SiStripOfflineDQM::~SiStripOfflineDQM() {
  delete poMui;
}

void SiStripOfflineDQM::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Essential: creates some object that are used in createSummary
  oActionExecutor_.readConfiguration();
  //  nQTestEventsPassed_ = 0;

  if( bVerbose) {
    LogInfo( "SiStripOfflineDQM") << "[beginJob] done";
  }
}

void SiStripOfflineDQM::analyze( const edm::Event      &roEVENT, 
				                         const edm::EventSetup &roEVENT_SETUP) {

  //  if( nQTEST_EVENTS_DELAY_ < nQTestEventsPassed_) {
  //    oActionExecutor_.setupQTests( poMui);
  //    nQTestEventsPassed_ = 0;
  //  }

  //  ++nQTestEventsPassed_;

  if( bVerbose) {
    LogInfo( "SiStripOfflineDQM") << "[analyze] done";
  }
}


void SiStripOfflineDQM::endJob() {
  if( bVerbose) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] start";
  }

  oActionExecutor_.setupQTests( poMui);
  oActionExecutor_.createSummary( poMui);

  if( bSaveInFile) {
    oActionExecutor_.saveMEs( poMui, oFILE_NAME);
  }

  if( bVerbose) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] done";
  }
}

