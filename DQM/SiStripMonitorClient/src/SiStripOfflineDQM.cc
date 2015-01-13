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

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

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
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

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

  usedWithEDMtoMEConverter_= configPar_.getUntrackedParameter<bool>("UsedWithEDMtoMEConverter",false); 
  createSummary_           = configPar_.getUntrackedParameter<bool>("CreateSummary",false);
  globalStatusFilling_     = configPar_.getUntrackedParameter<int>("GlobalStatusFilling", 1);
  printFaultyModuleList_   = configPar_.getUntrackedParameter<bool>("PrintFaultyModuleList", false);
  useSSQuality_            = configPar_.getUntrackedParameter<bool>("useSSQuality",false); //need to define
  ssqLabel_                = configPar_.getUntrackedParameter<std::string>("ssqLabel",""); //need to define

  // Essential: reads xml file to get the histogram names to create summary
  // Read the summary configuration file
  if (createSummary_) {  
    if (!actionExecutor_->readConfiguration()) {
      edm::LogInfo ("ReadConfigurationProblem") <<"SiStripOfflineDQM:: Error to read configuration file!! Summary will not be produced!!!";
      createSummary_ = false;
    }
  }

  configRead = false;
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

  edm::ESHandle<TrackerTopology> tTopoHandle;
  eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
  tTopo = tTopoHandle.product();
  
  eSetup.get<SiStripQualityRcd>().get(ssqLabel_,ssq);
}
/** 
 * @brief 
 * 
 * End Lumi
 *
*/
void SiStripOfflineDQM::dqmEndLuminosityBlock(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {
  edm::LogInfo( "EndLumiBlock") << "SiStripOfflineDQM::endLuminosityBlock";

  if (globalStatusFilling_ > 0) actionExecutor_->createStatus(ibooker , igetter);

  if (trackerFEDsFound_) {
    if (globalStatusFilling_ > 0) actionExecutor_->fillStatusAtLumi(ibooker , igetter);
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
  eSetup.get<SiStripDetCablingRcd>().get(det_cabling);

  if (!trackerFEDsFound_) {
    if (globalStatusFilling_ > 0)  actionExecutor_->fillDummyStatus();
    return;
  }

  if (!usedWithEDMtoMEConverter_) {

    // create Summary Plots
    //if (createSummary_)  actionExecutor_->createSummaryOffline(ibooker , igetter);

    // Create TrackerMap
    bool create_tkmap    = configPar_.getUntrackedParameter<bool>("CreateTkMap",false); 
    if (create_tkmap) {
      //std::vector<edm::ParameterSet> tkMapOptions = configPar_.getUntrackedParameter< std::vector<edm::ParameterSet> >("TkMapOptions" );
      if (actionExecutor_->readTkMapConfiguration(eSetup)) configRead = true;
      else configRead = false;
      /*      
	for(std::vector<edm::ParameterSet>::iterator it = tkMapOptions.begin(); it != tkMapOptions.end(); ++it) {
	  edm::ParameterSet tkMapPSet = *it;
	  std::string map_type = it->getUntrackedParameter<std::string>("mapName","");
	  tkMapPSet.augment(configPar_.getUntrackedParameter<edm::ParameterSet>("TkmapParameters"));
	  edm::LogInfo("TkMapParameters") << tkMapPSet;
	  actionExecutor_->createOfflineTkMap(tkMapPSet, ibooker , igetter , map_type, ssq); 
	}
	*/
    } 
  }
}
/** 
* @brief 
* 
* End Job
*
*/
void SiStripOfflineDQM::dqmEndJob(DQMStore::IBooker& ibooker, DQMStore::IGetter& igetter) {

  edm::LogInfo( "EndOfJob") << "SiStripOfflineDQM::endJob";
  
  if (globalStatusFilling_ > 0) actionExecutor_->createStatus(ibooker , igetter);

  if (!trackerFEDsFound_) {
    if (globalStatusFilling_ > 0)  actionExecutor_->fillDummyStatus();
    return;
  }
  
  if (globalStatusFilling_ > 0) actionExecutor_->fillStatus(ibooker , igetter , det_cabling, tTopo);

  if (!usedWithEDMtoMEConverter_) {
    if (createSummary_)  actionExecutor_->createSummaryOffline(ibooker , igetter);                                                                                             
  }

  if (configRead)
    {
      std::vector<edm::ParameterSet> tkMapOptions = configPar_.getUntrackedParameter< std::vector<edm::ParameterSet> >("TkMapOptions" );
      for(std::vector<edm::ParameterSet>::iterator it = tkMapOptions.begin(); it != tkMapOptions.end(); ++it) {
	edm::ParameterSet tkMapPSet = *it;
	std::string map_type = it->getUntrackedParameter<std::string>("mapName","");
	tkMapPSet.augment(configPar_.getUntrackedParameter<edm::ParameterSet>("TkmapParameters"));
	edm::LogInfo("TkMapParameters") << tkMapPSet;
	actionExecutor_->createOfflineTkMap(tkMapPSet, ibooker , igetter , map_type, ssq); 
      }
    }

  if (!usedWithEDMtoMEConverter_) {
    if (printFaultyModuleList_) {
      std::ostringstream str_val;
      actionExecutor_->printFaultyModuleList(ibooker , igetter , str_val);
      std::cout << str_val.str() << std::endl;
    }  
  }
  
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripOfflineDQM);
