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
// $Id: SiStripOfflineDQM.cc,v 1.10 2007/04/05 21:06:44 samvel Exp $
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

/** 
* @brief 
*   Construct object
* 
* @param roPARAMETER_SET 
*   Regular Parameter Set that represent read configuration file
*/
SiStripOfflineDQM::SiStripOfflineDQM( const edm::ParameterSet &roPARAMETER_SET)
  : bVERBOSE_( roPARAMETER_SET.getUntrackedParameter<bool>( "bVerbose")),
    bSAVE_IN_FILE_( roPARAMETER_SET.getUntrackedParameter<bool>( "bOutputMEsInRootFile")),
    oOUT_FILE_NAME_( roPARAMETER_SET.getUntrackedParameter<std::string>( "oOutputFile")),
    poMui_( new MonitorUIRoot()),
    oActionExecutor_() {

  // Create MessageSender
  LogInfo( "SiStripOfflineDQM");
}

SiStripOfflineDQM::~SiStripOfflineDQM() {
  delete poMui_;
}

/** 
* @brief 
*   Executed everytime once all events are processed
* 
* @param roEVENT_SETUP 
*   Event Setup object
*/
void SiStripOfflineDQM::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Essential: creates some object that are used in createSummary
  oActionExecutor_.readConfiguration();
  oActionExecutor_.setupQTests( poMui_);

  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[beginJob] done";
  }
}

/** 
* @brief 
* 
* @param roEVENT 
*   Event Object that holds all collections
* @param roEVENT_SETUP 
*   Event Setup Object with Geometry, Magnetic Field, etc.
*/
void SiStripOfflineDQM::analyze( const edm::Event      &roEVENT, 
				                         const edm::EventSetup &roEVENT_SETUP) {

  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[analyze] done";
  }
}


void SiStripOfflineDQM::endJob() {
  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] start";
  }

  poMui_->runQTests();

  LogInfo( "SiStripOfflineDQM")
    << "Summary";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummary( poMui_->getBEInterface());

  LogInfo( "SiStripOfflineDQM")
    << "SummaryLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryLite( poMui_->getBEInterface());

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXML";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXML( poMui_->getBEInterface());

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXMLLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXMLLite( poMui_->getBEInterface());

  oActionExecutor_.createSummary( poMui_->getBEInterface());

  if( bSAVE_IN_FILE_) {
    oActionExecutor_.saveMEs( poMui_->getBEInterface(), oOUT_FILE_NAME_);
  }

  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] done";
  }
}

