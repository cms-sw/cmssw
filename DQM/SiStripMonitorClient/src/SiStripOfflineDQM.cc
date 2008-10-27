
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
// $Id: SiStripOfflineDQM.cc,v 1.19 2008/08/29 10:08:51 dutta Exp $
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include "DQM/SiStripMonitorClient/interface/SiStripOfflineDQM.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

/** 
* @brief 
*   Construct object
* 
* @param roPARAMETER_SET 
*   Regular Parameter Set that represent read configuration file
*/
SiStripOfflineDQM::SiStripOfflineDQM(edm::ParameterSet const& pSet) {
  // Create MessageSender
  edm::LogInfo( "SiStripOfflineDQM") << "SiStripOfflineDQM::Deleting SiStripOfflineDQM ";

  // Action Executor
  actionExecutor_ = new SiStripActionExecutor();

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();

  createSummary_       = pSet.getUntrackedParameter<bool>("CreateSummary",true);
  inputFileName_       = pSet.getUntrackedParameter<std::string>("InputFileName","");
  outputFileName_      = pSet.getUntrackedParameter<std::string>("OutputFileName","");
  globalStatusFilling_ = pSet.getUntrackedParameter<int>("GlobalStatusFilling", 1);

  nEvents_  = 0;
}
/** 
* @brief 
*   Destructor
* 
*/
SiStripOfflineDQM::~SiStripOfflineDQM() {
  edm::LogInfo("SiStripOfflineDQM") << "SiStripOfflineDQM::Deleting SiStripOfflineDQM ";

}
/** 
* @brief 
*   Executed at the begining of application
* 
* @param eSetup
*   Event Setup object
*/
void SiStripOfflineDQM::beginJob( const edm::EventSetup &eSetup) {

  // Essential: reads xml file to get the histogram names to create summary
  // Read the summary configuration file
  if (createSummary_) {  
    if (!actionExecutor_->readConfiguration()) {
      edm::LogInfo ("SiStripOfflineDQM") <<"SiStripOfflineDQM:: Error to read configuration file!! Summary will not be produced!!!";
      createSummary_ = false;
    }
  }
  if (globalStatusFilling_) actionExecutor_->bookGlobalStatus(dqmStore_);

  edm::LogInfo("SiStripOfflineDQM") << "SiStripOfflineDQM::beginJob done";
}
/** 
* @brief 
*   Executed at the begining a Run
* 
* @param run
*   Run  object
* @param eSetup
*  Event Setup object with Geometry, Magnetic Field, etc.
*/
//
// -- Begin Run
//
void SiStripOfflineDQM::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripOfflineDQM") <<"SiStripOfflineDQM:: Begining of Run";

  if (!openInputFile()) createSummary_ = false;
}
/** 
 * @brief
 *
 *  Executed at every Event
 *
 * @param Event                             
 *   Event  
 *                                                                                                                                                      
 * @param eSetup                                                                                                                                                      *  Event Setup object with Geometry, Magnetic Field, etc.    
 */
void SiStripOfflineDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  nEvents_++;  
}
/** 
* @brief 
* 
* End Job
*
*/
void SiStripOfflineDQM::endJob() {
  edm::LogInfo( "SiStripOfflineDQM") << "SiStripOfflineDQM::EndJob";
  // create Summary Plots
  if (createSummary_)       actionExecutor_->createSummaryOffline(dqmStore_);

  // Fill Global Status
  if (globalStatusFilling_ == 1) actionExecutor_->fillGlobalStatusFromModule(dqmStore_);
  if (globalStatusFilling_ == 2) actionExecutor_->fillGlobalStatusFromLayer(dqmStore_);

  // Save Output file
  //  std::string outputFileName = inputFileName_.replace(inputFileName_.find("-standAlone"), 11, "");
  dqmStore_->cd();
  dqmStore_->save(outputFileName_, "","","");
}
/** 
* @brief 
* 
* Open Input File
*
*/
bool SiStripOfflineDQM::openInputFile() { 
  if (inputFileName_.size() == 0) return false;
  edm::LogInfo("SiStripOfflineDQM") <<  "SiStripOfflineDQM::openInputFile: Accessing root File" << inputFileName_;
  dqmStore_->setVerbose(0);
  dqmStore_->open(inputFileName_, false); 
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);
