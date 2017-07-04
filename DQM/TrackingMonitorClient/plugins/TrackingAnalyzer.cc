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
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>

#define BUF_SIZE 256

//
// -- Constructor
//
TrackingAnalyser::TrackingAnalyser(edm::ParameterSet const& ps) :
  verbose_(ps.getUntrackedParameter<bool>("verbose",false))
{
  if (verbose_) std::cout << "[TrackingAnalyser::TrackingAnalyser]" << std::endl;
  // Get TkMap ParameterSet 
  //  tkMapPSet_ = ps.getParameter<edm::ParameterSet>("TkmapParameters");

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

  // instantiate web interface
  actionExecutor_      = new TrackingActionExecutor(ps);
  trackerFEDsFound_      = false;
  trackerFEDsWdataFound_ = false;

  nFEDinfoDir_         = ps.getParameter<std::string>("nFEDinfoDir");
  nFEDinVsLSname_      = ps.getParameter<std::string>("nFEDinVsLSname");
  nFEDinWdataVsLSname_ = ps.getParameter<std::string>("nFEDinWdataVsLSname");
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
}
//
// -- Begin Run
//
void TrackingAnalyser::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("TrackingAnalyser") <<"TrackingAnalyser:: Begining of Run";

  if (verbose_) std::cout << "[TrackingAnalyser::beginRun]" << std::endl;
  // Check latest Fed cabling and create TrackerMapCreator
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser::beginRun: " 
				    << " Change in Cabling, recrated TrackerMap";     
    eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
    eSetup.get<SiStripDetCablingRcd>().get(detCabling_);
  } 
}
//
// -- Begin Luminosity Block
//
void TrackingAnalyser::dqmBeginLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_,edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser:: Begin of LS transition";
  if (verbose_) std::cout << "[TrackingAnalyser::dqmBeginLuminosityBlock]" << std::endl;
}

//
// -- End Luminosity Block
//
void TrackingAnalyser::dqmEndLuminosityBlock(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_,edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo ("TrackingAnalyser") <<"TrackingAnalyser:: End of LS transition, performing the DQM client operation";
  if (verbose_) std::cout << "[TrackingAnalyser::endLuminosityBlock]" << std::endl;
  nLumiSecs_++;

  if (verbose_) std::cout << "[TrackingAnalyser::endLuminosityBlock] globalStatusFilling_ " << (globalStatusFilling_ ? "YES" : "NOPE") << std::endl;
  if (globalStatusFilling_) actionExecutor_->createGlobalStatus(ibooker_,igetter_);

  double iLS = lumiSeg.id().luminosityBlock();
  checkTrackerFEDsInLS(igetter_,iLS);
  checkTrackerFEDsWdataInLS(igetter_,iLS);
  if (verbose_) std::cout << "[TrackingAnalyser::endLuminosityBlock] trackerFEDsFound_ "      << (trackerFEDsFound_      ? "YES" : "NOPE") << std::endl;
  if (verbose_) std::cout << "[TrackingAnalyser::endLuminosityBlock] trackerFEDsWdataFound_ " << (trackerFEDsWdataFound_ ? "YES" : "NOPE") << std::endl;

  if (!trackerFEDsFound_) {
    actionExecutor_->fillDummyGlobalStatus();
    actionExecutor_->createDummyShiftReport();
  } else {
    if (trackerFEDsWdataFound_) {
      actionExecutor_->fillGlobalStatus(ibooker_,igetter_);
      if (shiftReportFrequency_ != -1) actionExecutor_->createShiftReport(ibooker_,igetter_);
    }
  }

  if (!trackerFEDsFound_) {
    actionExecutor_->fillDummyLSStatus();
    return;
  }   

  if (verbose_) std::cout << "====================================================== " << std::endl;
  if (verbose_) std::cout << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() << std::endl;
  if (verbose_) std::cout << "====================================================== " << std::endl;

}

//
// -- End Job
//
void TrackingAnalyser::dqmEndJob(DQMStore::IBooker & ibooker_, DQMStore::IGetter & igetter_) {
  edm::LogInfo("TrackingAnalyser") <<"TrackingAnalyser:: endjob called!";
  if (verbose_) std::cout << "[TrackingAnalyser::dqmEndJob]" << std::endl;

  if (globalStatusFilling_) actionExecutor_->createGlobalStatus(ibooker_,igetter_);
  // Fill Global Status
  if (globalStatusFilling_ > 0) {
    actionExecutor_->fillGlobalStatus(ibooker_,igetter_);
  }
  

}
//
// Check Tracker FEDs
//
void TrackingAnalyser::checkTrackerFEDsInLS(DQMStore::IGetter & igetter, double iLS)
{

  double nFEDinLS = 0.;
  MonitorElement* tmpME = igetter.get(nFEDinfoDir_+"/"+nFEDinVsLSname_);
  if (tmpME) {
    TProfile* tmpP = tmpME->getTProfile();
    size_t ibin = tmpP->GetXaxis()->FindBin(iLS);
    if (verbose_) std::cout << "iLS: " << iLS << " ibin: " << ibin;
    nFEDinLS = tmpME->getBinContent(ibin);
    if (verbose_) std::cout << " ---> nFEDinLS: " << nFEDinLS;
  }

  trackerFEDsFound_ = (nFEDinLS>0);
  if (verbose_) std::cout << " ---> trackerFEDsFound_: " << trackerFEDsFound_ << std::endl;
}

void TrackingAnalyser::checkTrackerFEDsWdataInLS(DQMStore::IGetter & igetter, double iLS)
{

  double nFEDinLS = 0.;
  MonitorElement* tmpME = igetter.get(nFEDinfoDir_+"/"+nFEDinWdataVsLSname_);
  if (verbose_) std::cout << "found " << nFEDinfoDir_ << "/" << nFEDinWdataVsLSname_ << " ? " << (tmpME ? "YES" : "NOPE") << std::endl;
  if (tmpME) {
    TProfile* tmpP = tmpME->getTProfile();
    size_t ibin = tmpP->GetXaxis()->FindBin(iLS);
    if (verbose_) std::cout << "iLS: " << iLS << " ibin: " << ibin;
    nFEDinLS = tmpME->getBinContent(ibin);
    if (verbose_) std::cout << " ---> nFEDinLS: " << nFEDinLS;
  }

  trackerFEDsWdataFound_ = (nFEDinLS>0);
  if (verbose_) std::cout << " ---> trackerFEDsWdataFound_: " << trackerFEDsWdataFound_ << std::endl;
}


#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TrackingAnalyser);
