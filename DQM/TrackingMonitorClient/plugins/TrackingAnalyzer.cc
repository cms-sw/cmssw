
#include "DQM/TrackingMonitorClient/plugins/TrackingAnalyzer.h"


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
#include "DQM/TrackingMonitorClient/interface/TrackingActionExecutor.h"
#include "DQM/TrackingMonitorClient/interface/TrackingUtility.h"

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
TrackingAnalyser::TrackingAnalyser(edm::ParameterSet const& ps) {
  
  // Get TkMap ParameterSet 
  tkMapPSet_ = ps.getParameter<edm::ParameterSet>("TkmapParameters");

  std::string localPath = std::string("DQM/TrackingMonitorClient/test/loader.html");
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



  edm::LogInfo("TrackingAnalyser") << " TrackingAnalyser::Creating TrackingAnalyser ";
  staticUpdateFrequency_ = ps.getUntrackedParameter<int>("StaticUpdateFrequency",1);
  globalStatusFilling_   = ps.getUntrackedParameter<int>("GlobalStatusFilling", 1);
  shiftReportFrequency_  = ps.getUntrackedParameter<int>("ShiftReportFrequency", 1);   
  
  edm::InputTag rawDataTag = ps.getUntrackedParameter<edm::InputTag>("RawDataTag"); 
  rawDataToken_ = consumes<FEDRawDataCollection>(rawDataTag);

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();


  // instantiate web interface
  actionExecutor_      = new TrackingActionExecutor(ps);
  trackerFEDsFound_  = false;
  endLumiAnalysisOn_ = false;
}
//
// -- Destructor
//
TrackingAnalyser::~TrackingAnalyser(){

  edm::LogInfo("TrackingAnalyser") << "TrackingAnalyser::Deleting TrackingAnalyser ";

}
//
// -- Begin Job
//
void TrackingAnalyser::beginJob(){

  nLumiSecs_ = 0;
  nEvents_   = 0;
}
//
// -- Begin Run
//
void TrackingAnalyser::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  std::cout << "[TrackingAnalyser::beginRun] .. starting" << std::endl;
  edm::LogInfo ("TrackingAnalyser") <<"TrackingAnalyser:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser::beginRun: " 
				    << " Change in Cabling, recrated TrackerMap";     
    eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);
  } 
  if (globalStatusFilling_) actionExecutor_->createGlobalStatus(dqmStore_);
}
//
// -- Begin Luminosity Block
//
void TrackingAnalyser::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser:: Begin of LS transition";
}
//
//  -- Analyze 
//
void TrackingAnalyser::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  std::cout << "[TrackingAnalyser::analyze] .. starting" << std::endl;
  nEvents_++;  
  if (nEvents_ == 1 && globalStatusFilling_ > 0) {
    checkTrackerFEDs(e);
    if (!trackerFEDsFound_) {
      actionExecutor_->fillDummyGlobalStatus();
      actionExecutor_->createDummyShiftReport();
    } else {
      actionExecutor_->fillGlobalStatus(dqmStore_);
      if (shiftReportFrequency_ != -1) actionExecutor_->createShiftReport(dqmStore_);
    }
  }

}
//
// -- End Luminosity Block
//
void TrackingAnalyser::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo ("TrackingAnalyser") <<"TrackingAnalyser:: End of LS transition, performing the DQM client operation";

  nLumiSecs_++;

  if (!trackerFEDsFound_) {
    actionExecutor_->fillDummyLSStatus();
    return;
  }   
  endLumiAnalysisOn_ = true;

  std::cout << "====================================================== " << std::endl;
  std::cout << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() << std::endl;
  std::cout << "====================================================== " << std::endl;
  // Fill Global Status
  if (globalStatusFilling_ > 0) {
    actionExecutor_->fillStatusAtLumi(dqmStore_);
  }
  endLumiAnalysisOn_ = false;
}

//
// -- End Run
//
void TrackingAnalyser::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  edm::LogInfo ("TrackingAnalyser") <<"TrackingAnalyser:: End of Run";
}
//
// -- End Job
//
void TrackingAnalyser::endJob(){
  edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser:: endjob called!";
}
//
// Check Tracker FEDs
//
void TrackingAnalyser::checkTrackerFEDs(edm::Event const& e) {
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  e.getByToken( rawDataToken_, rawDataHandle );
  if ( !rawDataHandle.isValid() ) return;
  
  const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
  const int siStripFedIdMin = FEDNumbering::MINSiStripFEDID;
  const int siStripFedIdMax = FEDNumbering::MAXSiStripFEDID; 
  const int siPixelFedIdMin = FEDNumbering::MINSiPixelFEDID;
  const int siPixelFedIdMax = FEDNumbering::MAXSiPixelFEDID;    

  unsigned int nFed = 0;
  for (int i=siStripFedIdMin; i <= siStripFedIdMax; i++) {
    if (rawDataCollection.FEDData(i).size() &&
	rawDataCollection.FEDData(i).data()    )
      nFed++;
  }
  for (int i=siPixelFedIdMin; i <= siPixelFedIdMax; i++) {
    if (rawDataCollection.FEDData(i).size() &&
	rawDataCollection.FEDData(i).data()    )
      nFed++;
  }

  trackerFEDsFound_ = (nFed > 0);

}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingAnalyser);
