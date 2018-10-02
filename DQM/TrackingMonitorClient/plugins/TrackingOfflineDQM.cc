// -*- C++ -*-
//
// Package:    TrackingMonitorClient
// Class:      TrackingOfflineDQM
// 
/**\class TrackingOfflineDQM TrackingOfflineDQM.cc DQM/TrackingMonitorCluster/src/TrackingOfflineDQM.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//         Created:  Wed Oct  5 16:42:34 CET 2006
// $Id: TrackingOfflineDQM.cc,v 1.42 2013/01/02 17:41:51 wmtan Exp $
//
//

#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/TrackingMonitorClient/interface/TrackingActionExecutor.h"

#include "DQM/TrackingMonitorClient/plugins/TrackingOfflineDQM.h"

//Run Info
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"
#include "CondFormats/RunInfo/interface/RunSummary.h"
#include "CondFormats/RunInfo/interface/RunInfo.h"
// Cabling
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include <iostream>
#include <iomanip>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>


/** 
* @brief 
*   Construct object
* 
* @param roPARAMETER_SET 
*   Regular Parameter Set that represent read configuration file
*/
TrackingOfflineDQM::TrackingOfflineDQM(edm::ParameterSet const& pSet) :
  configPar_(pSet)
{

  // Action Executor
  actionExecutor_ = new TrackingActionExecutor(pSet);

  usedWithEDMtoMEConverter_= configPar_.getUntrackedParameter<bool>("UsedWithEDMtoMEConverter",false); 
  inputFileName_           = configPar_.getUntrackedParameter<std::string>("InputFileName","");
  outputFileName_          = configPar_.getUntrackedParameter<std::string>("OutputFileName","");
  globalStatusFilling_     = configPar_.getUntrackedParameter<int>("GlobalStatusFilling", 1);

}
/** 
* @brief 
*   Destructor
* 
*/
TrackingOfflineDQM::~TrackingOfflineDQM() {

}
/** 
* @brief 
*   Executed at the begining of application
* 
* @param eSetup
*   Event Setup object
*/
void TrackingOfflineDQM::beginJob() {

  edm::LogInfo("BeginJobDone") << "TrackingOfflineDQM::beginJob done";
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
void TrackingOfflineDQM::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("BeginRun") <<"TrackingOfflineDQM:: Begining of Run";

  int nFEDs = 0;
  int nPixelFEDs = 0;

  if(auto runInfoRec = eSetup.tryToGet<RunInfoRcd>()) {

    edm::ESHandle<RunInfo> sumFED;
    runInfoRec->get(sumFED);

    if ( sumFED.isValid() ) {

      const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
      const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID;
      const int siPixelFedIdMin = FEDNumbering::MINSiPixelFEDID;
      const int siPixelFedIdMax = FEDNumbering::MAXSiPixelFEDID;
      
      std::vector<int> FedsInIds= sumFED->m_fed_in;   
      for ( auto fedID : FedsInIds ) {
	if ( fedID >= siPixelFedIdMin && fedID <= siPixelFedIdMax ) {
	  ++nPixelFEDs;
	  ++nFEDs;
	}
	if ( fedID >= siStripFedIdMin && fedID <= siStripFedIdMax )
	  ++nFEDs;
      }
    }
  }
  const int siPixelFedN = (FEDNumbering::MAXSiPixelFEDID-FEDNumbering::MINSiPixelFEDID+1);
  allpixelFEDsFound_ = (nPixelFEDs == siPixelFedN);
  trackerFEDsFound_  = (nFEDs > 0);
  
}

/** 
 * @brief 
 * 
 * End Lumi
 *
*/
void TrackingOfflineDQM::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_,edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& iSetup) {


  edm::LogInfo("TrackingOfflineDQM") << "TrackingOfflineDQM::dqmBeginLuminosityBlock";

  if (globalStatusFilling_ > 0) {
    actionExecutor_->createLSStatus(ibooker_,igetter_);

    if (trackerFEDsFound_) actionExecutor_->fillStatusAtLumi(ibooker_,igetter_);
    else actionExecutor_->fillDummyLSStatus();
  }
}
/** 
 * @brief 
 * 
 * End Run
 *
*/
void TrackingOfflineDQM::dqmEndJob(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_)
{

  edm::LogInfo("TrackingOfflineDQM") << "TrackingOfflineDQM::dqmEndJob";

  if (globalStatusFilling_ > 0) {
    actionExecutor_->createGlobalStatus(ibooker_,igetter_);

    if ( !trackerFEDsFound_ ) {
      actionExecutor_->fillDummyGlobalStatus();
      return;
    } else {
      actionExecutor_->fillGlobalStatus(ibooker_,igetter_);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingOfflineDQM);
