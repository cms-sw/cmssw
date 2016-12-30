

/*
 * \file SiStripAnalyser.cc
 * 
 * \author  S. Dutta INFN-Pisa
 *
 */


#include "DQM/SiStripMonitorClient/interface/SiStripAnalyser.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"


#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

#define BUF_SIZE 256

//
// -- Constructor
//
SiStripAnalyser::SiStripAnalyser(edm::ParameterSet const& ps) :
  verbose_(ps.getUntrackedParameter<bool>("verbose",false))
{
  
  // Get TkMap ParameterSet 
  tkMapPSet_ = ps.getParameter<edm::ParameterSet>("TkmapParameters");

  std::string localPath = std::string("DQM/SiStripMonitorClient/test/loader.html");
  std::ifstream fin(edm::FileInPath(localPath).fullPath().c_str(), std::ios::in);
  char buf[BUF_SIZE];
  
  if (!fin) {
    std::cerr << "Input File: loader.html"<< " could not be opened!" << std::endl;
    return;
  }

  while (fin.getline(buf, BUF_SIZE, '\n')) { // pops off the newline character 
    html_out_ << buf ;
  }
  fin.close();



  edm::LogInfo("SiStripAnalyser") << " SiStripAnalyser::Creating SiStripAnalyser ";
  summaryFrequency_      = ps.getUntrackedParameter<int>("SummaryCreationFrequency",1);
  tkMapFrequency_        = ps.getUntrackedParameter<int>("TkMapCreationFrequency",1); 
  staticUpdateFrequency_ = ps.getUntrackedParameter<int>("StaticUpdateFrequency",1);
  globalStatusFilling_   = ps.getUntrackedParameter<int>("GlobalStatusFilling", 1);
  shiftReportFrequency_  = ps.getUntrackedParameter<int>("ShiftReportFrequency", 1);   
  rawDataTag_            = ps.getUntrackedParameter<edm::InputTag>("RawDataTag"); 
  printFaultyModuleList_ = ps.getUntrackedParameter<bool>("PrintFaultyModuleList", true);
  nFEDinfoDir_           = ps.getUntrackedParameter<std::string>("nFEDinfoDir");
  nFEDinVsLSname_        = ps.getUntrackedParameter<std::string>("nFEDinVsLSname");

  rawDataToken_ = consumes<FEDRawDataCollection>(ps.getUntrackedParameter<edm::InputTag>("RawDataTag") );

  // instantiate web interface
  actionExecutor_ = new SiStripActionExecutor(ps);
  condDataMon_    = new SiStripClassToMonitorCondData(ps);
  trackerFEDsFound_ = false;
  endLumiAnalysisOn_ = false;

  // Read the summary configuration file
  if (!actionExecutor_->readConfiguration()) {
    edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Error to read configuration file!! Summary will not be produced!!!";
    summaryFrequency_ = -1;
  }
  nLumiSecs_ = 0;
}
//
// -- Destructor
//
SiStripAnalyser::~SiStripAnalyser(){

  edm::LogInfo("SiStripAnalyser") << "SiStripAnalyser::Deleting SiStripAnalyser ";
//  if (sistripWebInterface_) {
//     delete sistripWebInterface_;
//     sistripWebInterface_ = 0;
//  }
//  if (trackerMapCreator_) {
//    delete trackerMapCreator_;
//    trackerMapCreator_ = 0;
//  }

}
//
// -- Begin Run
//
void SiStripAnalyser::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser::beginRun: " 
				    << " Change in Cabling, recrated TrackerMap";     
    if (!actionExecutor_->readTkMapConfiguration(eSetup)) {
      edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Error to read configuration file!! TrackerMap will not be produced!!!";    
      tkMapFrequency_ = -1;

    }
    eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);
  } 
  if (condDataMon_) condDataMon_->beginRun(eSetup);
}
//
// -- Begin Luminosity Block
//
void SiStripAnalyser::dqmBeginLuminosityBlock(DQMStore::IBooker & ibooker , DQMStore::IGetter & igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: Begin of LS transition";
}
//
// -- End Luminosity Block
//
void SiStripAnalyser::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker , DQMStore::IGetter & igetter , edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of LS transition, performing the DQM client operation";

  nLumiSecs_++;

  if (globalStatusFilling_) actionExecutor_->createStatus( ibooker , igetter );


  //------ FROM ANALYZE - what to do?
  if (globalStatusFilling_ > 0) {
    checkTrackerFEDsInLS( igetter , lumiSeg.id().luminosityBlock() );
    if (!trackerFEDsFound_) {
      actionExecutor_->fillDummyStatus();
      actionExecutor_->createDummyShiftReport();
    } else {
      edm::ESHandle<TrackerTopology> tTopoHandle;
      eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
      const TrackerTopology* const tTopo = tTopoHandle.product();
      actionExecutor_->fillStatus(ibooker , igetter , detCabling_, tTopo);
      if (shiftReportFrequency_ != -1) actionExecutor_->createShiftReport(ibooker , igetter);
    }
  }
  //------ END FROM ANALYZE

  /* //not needed?
  if (!trackerFEDsFound_) {
    actionExecutor_->fillDummyStatus();
    return;
  } 
  */
  
  endLumiAnalysisOn_ = true;

  if (verbose_)
    {  
      std::cout << "====================================================== " << std::endl;
      std::cout << " ===> Iteration # " << nLumiSecs_ << " " 
		<< lumiSeg.luminosityBlock() << std::endl;
      std::cout << "====================================================== " << std::endl;
    }
  // Fill Global Status
  if (globalStatusFilling_ > 0) {
    edm::ESHandle<TrackerTopology> tTopoHandle;
    eSetup.get<IdealGeometryRecord>().get(tTopoHandle);
    const TrackerTopology* const tTopo = tTopoHandle.product();
    actionExecutor_->fillStatus(ibooker , igetter , detCabling_, tTopo);
  }
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%summaryFrequency_ == 0) {
    if (verbose_) std::cout << " Creating Summary " << std::endl;
    actionExecutor_->createSummary(ibooker , igetter);
  }
  endLumiAnalysisOn_ = false;
}

//
// -- End Run
//
void SiStripAnalyser::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of Run";
}
//
// -- End Job
//
void SiStripAnalyser::dqmEndJob(DQMStore::IBooker & ibooker, DQMStore::IGetter & igetter){
  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: endjob called!";
  if (printFaultyModuleList_) {
    std::ostringstream str_val;
    actionExecutor_->printFaultyModuleList(ibooker , igetter, str_val);
    if (verbose_) std::cout << str_val.str() << std::endl;
  }
}
//
// Check Tracker FEDs
//
void SiStripAnalyser::checkTrackerFEDsInLS(DQMStore::IGetter & igetter, double iLS) {

  double nFEDinLS = 0.;
  MonitorElement* tmpME = igetter.get(nFEDinfoDir_+"/"+nFEDinVsLSname_);
  if (tmpME) {
    TProfile* tmpP = tmpME->getTProfile();
    nFEDinLS = tmpME->getBinContent( tmpP->GetXaxis()->FindBin(iLS) );
  }

  trackerFEDsFound_ = (nFEDinLS>0);
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
