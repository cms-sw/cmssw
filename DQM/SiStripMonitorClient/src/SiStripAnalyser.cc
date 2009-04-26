/*
 * \file SiStripAnalyser.cc
 * 
 * $Date: 2009/03/26 08:03:23 $
 * $Revision: 1.47 $
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
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/DataRecord/interface/SiStripCondDataRecords.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

#define BUF_SIZE 256

using namespace edm;
using namespace std;
//
// -- Constructor
//
SiStripAnalyser::SiStripAnalyser(edm::ParameterSet const& ps) :
  ModuleWeb("SiStripAnalyser") {
  
  // Get TkMap ParameterSet 
  tkMapPSet_ = ps.getParameter<edm::ParameterSet>("TkmapParameters");

  string localPath = string("DQM/SiStripMonitorClient/test/loader.html");
  ifstream fin(edm::FileInPath(localPath).fullPath().c_str(), ios::in);
  char buf[BUF_SIZE];
  
  if (!fin) {
    cerr << "Input File: loader.html"<< " could not be opened!" << endl;
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

  // get back-end interface
  dqmStore_ = Service<DQMStore>().operator->();


  // instantiate web interface
  sistripWebInterface_ = new SiStripWebInterface(dqmStore_);
  actionExecutor_ = new SiStripActionExecutor(ps);
  condDataMon_    = new SiStripClassToMonitorCondData(ps);
  trackerFEDsFound_ = false;
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
// -- Begin Job
//
void SiStripAnalyser::beginJob(edm::EventSetup const& eSetup){

  // Read the summary configuration file
  if (!actionExecutor_->readConfiguration()) {
     edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Error to read configuration file!! Summary will not be produced!!!";
     summaryFrequency_ = -1;
  }
  if (globalStatusFilling_) actionExecutor_->createStatus(dqmStore_);
  nLumiSecs_ = 0;
  nEvents_   = 0;
}
//
// -- Begin Run
//
void SiStripAnalyser::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Begining of Run";

  // Check latest Fed cabling and create TrackerMapCreator
  unsigned long long cacheID = eSetup.get<SiStripFedCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser::beginRun: " 
				    << " Change in Cabling, recrated TrackerMap";     
    if (!actionExecutor_->readTkMapConfiguration()) {
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
void SiStripAnalyser::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: Begin of LS transition";
}
//
//  -- Analyze 
//
void SiStripAnalyser::analyze(edm::Event const& e, edm::EventSetup const& eSetup){
  nEvents_++;  
  if (nEvents_ == 1 && globalStatusFilling_ > 0) {
    checkTrackerFEDs(e);
    if (!trackerFEDsFound_) {
      actionExecutor_->fillDummyStatus();
      actionExecutor_->createDummyShiftReport();
    } else {
      actionExecutor_->fillStatus(dqmStore_);
      if (shiftReportFrequency_ != -1) actionExecutor_->createShiftReport(dqmStore_);
    }
  }

  unsigned int nval = sistripWebInterface_->getNumberOfConDBPlotRequest();
  if (nval > 0) {
    for (unsigned int ival = 0; ival < nval; ival++) {
      uint32_t det_id;
      string   subdet_type;
      uint32_t subdet_side;
      uint32_t layer_number;
      sistripWebInterface_->getConDBPlotParameters(ival, det_id, subdet_type, subdet_side, layer_number);
      if (condDataMon_) {
        if (det_id == 999) condDataMon_->getLayerMEsOnDemand(eSetup,subdet_type, subdet_side,layer_number);
        else if (layer_number == 999 && subdet_side == 999) condDataMon_->getModMEsOnDemand(eSetup,det_id);
      }
    }
    sistripWebInterface_->clearConDBPlotRequests();
  }
  sistripWebInterface_->setActionFlag(SiStripWebInterface::CreatePlots);
  sistripWebInterface_->performAction();
}
//
// -- End Luminosity Block
//
void SiStripAnalyser::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {

  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of LS transition, performing the DQM client operation";

  nLumiSecs_++;

  //  sistripWebInterface_->setCabling(detCabling_);
 
  cout << "====================================================== " << endl;
  cout << " ===> Iteration # " << nLumiSecs_ << " " 
                               << lumiSeg.luminosityBlock() << endl;
  cout << "====================================================== " << endl;
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%summaryFrequency_ == 0) {
    cout << " Creating Summary " << endl;
    actionExecutor_->createSummary(dqmStore_);
  }
  // -- Create TrackerMap  according to the frequency
  if (tkMapFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%tkMapFrequency_ == 0) {
    cout << " Creating Tracker Map " << endl;
    string tkmap_type =  sistripWebInterface_->getTkMapType();
    actionExecutor_->createTkMap(tkMapPSet_, fedCabling_, dqmStore_, tkmap_type);
  }
  // Create predefined plots
  if (staticUpdateFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%staticUpdateFrequency_  == 0) {
    cout << " Creating predefined plots " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::PlotHistogramFromLayout);
    sistripWebInterface_->performAction();
  }
  // Fill Global Status
  if (globalStatusFilling_ > 0 && trackerFEDsFound_) {
    if (globalStatusFilling_ == 1) actionExecutor_->fillStatus(dqmStore_);
    if (globalStatusFilling_ == 2) actionExecutor_->fillStatus(dqmStore_);
  }
  // Create Shift Report
  if (shiftReportFrequency_ != -1 && trackerFEDsFound_ && nLumiSecs_%shiftReportFrequency_  == 0) {
    actionExecutor_->createShiftReport(dqmStore_);
  }
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
void SiStripAnalyser::endJob(){
  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: endjob called!";
  if (printFaultyModuleList_) {
    std::ostringstream str_val;
    actionExecutor_->printFaultyModuleList(dqmStore_, str_val);
    std::cout << str_val.str() << std::endl;
  }
}
//
// Check Tracker FEDs
//
void SiStripAnalyser::checkTrackerFEDs(edm::Event const& e) {
  edm::Handle<FEDRawDataCollection> rawDataHandle;
  e.getByLabel(rawDataTag_, rawDataHandle);
  if ( !rawDataHandle.isValid() ) return;
  
  const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
  const FEDNumbering numbering;
  const int siStripFedIdMin = numbering.getSiStripFEDIds().first;
  const int siStripFedIdMax = numbering.getSiStripFEDIds().second; 
    
  unsigned int nFed = 0;
  for (int i=siStripFedIdMin; i <= siStripFedIdMax; i++) {
    if (rawDataCollection.FEDData(i).size() && rawDataCollection.FEDData(i).data()) {
      nFed++;
    }
  }
  if (nFed > 0) trackerFEDsFound_ = true;
}
//
// -- Create default web page
//
void SiStripAnalyser::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
  bool isRequest = false;
  cgicc::Cgicc cgi(in);
  cgicc::CgiEnvironment cgie(in);
  //  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: defaultWebPage "
  //             << " query string : " << cgie.getQueryString();
  //  if ( xgi::Utils::hasFormElement(cgi,"ClientRequest") ) isRequest = true;
  string q_string = cgie.getQueryString();
  if (q_string.find("RequestID") != string::npos) isRequest = true;
  if (!isRequest) {    
    *out << html_out_.str() << std::endl;
  }  else {
    // Handles all HTTP requests of the form
    int iter = nEvents_/10;
    sistripWebInterface_->handleAnalyserRequest(in, out, detCabling_, iter);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
