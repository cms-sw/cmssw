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
// $Id: SiStripOfflineDQM.cc,v 1.16 2008/02/21 23:17:49 dutta Exp $
//
//

// Root UI that is used by original Client's SiStripActionExecuter
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

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
    bCreateSummary_(false),
    oActionExecutor_() {

  // Create MessageSender
  LogInfo( "SiStripOfflineDQM");

  poDQMStore_ = edm::Service<DQMStore>().operator->();

}

SiStripOfflineDQM::~SiStripOfflineDQM() {
}

/** 
* @brief 
*   Executed everytime once all events are processed
* 
* @param roEVENT_SETUP 
*   Event Setup object
*/
void SiStripOfflineDQM::beginJob( const edm::EventSetup &roEVENT_SETUP) {
  // Essential: reads xml file to get the histogram names to create summary
  if (oActionExecutor_.readConfiguration()) bCreateSummary_ = true;

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
void SiStripOfflineDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  if (bCreateSummary_) { 
    oActionExecutor_.createSummary( poDQMStore_);
  }
}
void SiStripOfflineDQM::endJob() {
  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] start";
  }


  LogInfo( "SiStripOfflineDQM")
    << "Summary";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummary( poDQMStore_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryLite( poDQMStore_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXML";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXML( poDQMStore_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXMLLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXMLLite( poDQMStore_);


  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] done";
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);

