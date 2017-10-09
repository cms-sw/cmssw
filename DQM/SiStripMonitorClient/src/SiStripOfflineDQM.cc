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
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"

#include "DQM/SiStripMonitorClient/interface/SiStripOfflineDQM.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
// Cabling
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
//#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

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
SiStripOfflineDQM::SiStripOfflineDQM(edm::ParameterSet const& pSet) : configPar_(pSet) {
  // Create MessageSender

  // Action Executor
  actionExecutor_ = new SiStripActionExecutor(pSet);

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();

  usedWithEDMtoMEConverter_= configPar_.getUntrackedParameter<bool>("UsedWithEDMtoMEConverter",false); 
  createSummary_           = configPar_.getUntrackedParameter<bool>("CreateSummary",false);
  createTkInfoFile_        = configPar_.getUntrackedParameter<bool>("CreateTkInfoFile",false);
  inputFileName_           = configPar_.getUntrackedParameter<std::string>("InputFileName","");
  outputFileName_          = configPar_.getUntrackedParameter<std::string>("OutputFileName","");
  globalStatusFilling_     = configPar_.getUntrackedParameter<int>("GlobalStatusFilling", 1);
  printFaultyModuleList_   = configPar_.getUntrackedParameter<bool>("PrintFaultyModuleList", false);

  nEvents_  = 0;

  tkinfoTree_ = nullptr;

  if(createTkInfoFile_) {
    edm::Service<TFileService> fs;
    tkinfoTree_ = fs->make<TTree>("TkDetIdInfo", "");
  }

}
/** 
* @brief 
*   Destructor
* 
*/
SiStripOfflineDQM::~SiStripOfflineDQM() {

}
/** 
* @brief 
*   Executed at the begining of application
* 
* @param eSetup
*   Event Setup object
*/
void SiStripOfflineDQM::beginJob() {

  // Essential: reads xml file to get the histogram names to create summary
  // Read the summary configuration file
  if (createSummary_) {  
    if (!actionExecutor_->readConfiguration()) {
      edm::LogInfo ("ReadConfigurationProblem") <<"SiStripOfflineDQM:: Error to read configuration file!! Summary will not be produced!!!";
      createSummary_ = false;
    }
  }
  edm::LogInfo("BeginJobDone") << "SiStripOfflineDQM::beginJob done";
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
void SiStripOfflineDQM::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("BeginRun") <<"SiStripOfflineDQM:: Begining of Run";

  int nFEDs = 0;
  edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType("RunInfoRcd"));
  if( eSetup.find( recordKey ) != 0) {

    edm::ESHandle<RunInfo> sumFED;
    eSetup.get<RunInfoRcd>().get(sumFED);    
    if ( sumFED.isValid() ) {

      const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
      const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
      
      std::vector<int> FedsInIds= sumFED->m_fed_in;   
      for(unsigned int it = 0; it < FedsInIds.size(); ++it) {
        int fedID = FedsInIds[it];     
        
        if(fedID>=siStripFedIdMin &&  fedID<=siStripFedIdMax)  ++nFEDs;
      }
    }
  }
  if (nFEDs > 0) trackerFEDsFound_ = true;
  else trackerFEDsFound_ = false;
  if (!usedWithEDMtoMEConverter_) {
    if (!openInputFile()) createSummary_ = false;
  }
  if (globalStatusFilling_ > 0) actionExecutor_->createStatus(dqmStore_);
}
/** 
 * @brief
 *
 *  Executed at every Event
 *
 * @param Event                             
 *   Event  
 *                 
 * @param eSetup 
 *  Event Setup object with Geometry, Magnetic Field, etc.    
 */
void SiStripOfflineDQM::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  nEvents_++;  
}
/** 
 * @brief 
 * 
 * End Lumi
 *
*/
void SiStripOfflineDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {
  edm::LogInfo( "EndLumiBlock") << "SiStripOfflineDQM::endLuminosityBlock";
  if (trackerFEDsFound_) {
    if (globalStatusFilling_ > 0) actionExecutor_->fillStatusAtLumi(dqmStore_);
  }
}
/** 
 * @brief 
 * 
 * End Run
 *
*/
void SiStripOfflineDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup){

  edm::LogInfo( "EndOfRun") << "SiStripOfflineDQM::endRun";

  // Access Cabling
  edm::ESHandle< SiStripDetCabling > det_cabling;
  eSetup.get<SiStripDetCablingRcd>().get(det_cabling);
  //  edm::ESHandle< SiStripFedCabling > fed_cabling;
  //  eSetup.get<SiStripFedCablingRcd>().get(fed_cabling);
  if (globalStatusFilling_ > 0) actionExecutor_->createStatus(dqmStore_);

  if (!trackerFEDsFound_) {
    if (globalStatusFilling_ > 0)  actionExecutor_->fillDummyStatus();
    return;
  }

  // Fill Global Status
  if (globalStatusFilling_ > 0) actionExecutor_->fillStatus(dqmStore_, det_cabling, eSetup);

  if (!usedWithEDMtoMEConverter_) {

    // create Summary Plots
    if (createSummary_)  actionExecutor_->createSummaryOffline(dqmStore_);

    // Create TrackerMap
    bool create_tkmap    = configPar_.getUntrackedParameter<bool>("CreateTkMap",false); 
    if (create_tkmap) {
      std::vector<edm::ParameterSet> tkMapOptions = configPar_.getUntrackedParameter< std::vector<edm::ParameterSet> >("TkMapOptions" );
      if (actionExecutor_->readTkMapConfiguration(eSetup)) {
        std::vector<std::string> map_names;
        
        for(std::vector<edm::ParameterSet>::iterator it = tkMapOptions.begin(); it != tkMapOptions.end(); ++it) {
          edm::ParameterSet tkMapPSet = *it;
          std::string map_type = it->getUntrackedParameter<std::string>("mapName","");
          map_names.push_back(map_type);
          tkMapPSet.augment(configPar_.getUntrackedParameter<edm::ParameterSet>("TkmapParameters"));
          edm::LogInfo("TkMapParameters") << tkMapPSet;
          actionExecutor_->createOfflineTkMap(tkMapPSet, dqmStore_, map_type, eSetup); 
        }
        if(createTkInfoFile_) {
          actionExecutor_->createTkInfoFile(map_names, tkinfoTree_, dqmStore_);
        }
      }
    } 
  }
}
/** 
* @brief 
* 
* End Job
*
*/
void SiStripOfflineDQM::endJob() {

  edm::LogInfo( "EndOfJob") << "SiStripOfflineDQM::endJob";
  if (!usedWithEDMtoMEConverter_) {
    if (printFaultyModuleList_) {
      std::ostringstream str_val;
      actionExecutor_->printFaultyModuleList(dqmStore_, str_val);
      std::cout << str_val.str() << std::endl;
    }  
  }
}
/** 
* @brief 
* 
* Open Input File
*
*/
bool SiStripOfflineDQM::openInputFile() { 
  if (inputFileName_.size() == 0) return false;
  edm::LogInfo("OpenFile") <<  "SiStripOfflineDQM::openInputFile: Accessing root File" << inputFileName_;
  dqmStore_->open(inputFileName_, false); 
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);
