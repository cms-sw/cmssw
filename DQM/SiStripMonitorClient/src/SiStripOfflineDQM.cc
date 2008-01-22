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
// $Id: SiStripOfflineDQM.cc,v 1.14 2007/12/19 21:14:44 dutta Exp $
//
//

// Root UI that is used by original Client's SiStripActionExecuter
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


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

  poBei_ = edm::Service<DaqMonitorBEInterface>().operator->();

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
    oActionExecutor_.createSummary( poBei_);
  }
}
void SiStripOfflineDQM::endJob() {
  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] start";
  }


  LogInfo( "SiStripOfflineDQM")
    << "Summary";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummary( poBei_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryLite( poBei_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXML";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXML( poBei_);

  LogInfo( "SiStripOfflineDQM")
    << "SummaryXMLLite";
  LogInfo( "SiStripOfflineDQM")
    << oActionExecutor_.getQTestSummaryXMLLite( poBei_);


  if( bVERBOSE_) {
    LogInfo( "SiStripOfflineDQM") << "[endJob] done";
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);

