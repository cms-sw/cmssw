/*
 * \file SiStripAnalyser.cc
 * 
 * $Date: 2008/03/24 23:56:38 $
 * $Revision: 1.28 $
 * \author  S. Dutta INFN-Pisa
 *
 */


#include "DQM/SiStripMonitorClient/interface/SiStripAnalyser.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripActionExecutor.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include <SealBase/Callback.h>

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
  summaryFrequency_      = ps.getUntrackedParameter<int>("SummaryCreationFrequency",20);
  tkMapFrequency_        = ps.getUntrackedParameter<int>("TkMapCreationFrequency",50); 
  staticUpdateFrequency_ = ps.getUntrackedParameter<int>("StaticUpdateFrequency",10);


  // get back-end interface
  dqmStore_ = Service<DQMStore>().operator->();


  // instantiate web interface
  sistripWebInterface_ = new SiStripWebInterface(dqmStore_);
  actionExecutor_ = new SiStripActionExecutor();
  
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
    actionExecutor_->createTkMap(tkMapPSet_, fedCabling_, dqmStore_);
  }
  // Create predefined plots
  if (staticUpdateFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%staticUpdateFrequency_  == 0) {
    cout << " Creating predefined plots " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::PlotHistogramFromLayout);
    sistripWebInterface_->performAction();
  }
  fillGlobalStatus();
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
    int iter = nEvents_/100;
    sistripWebInterface_->handleAnalyserRequest(in, out, detCabling_, iter);
  }
}
//
// -- Get Global Status of Tracker
//
void SiStripAnalyser::fillGlobalStatus() {
  float gStatus = 0.0;
  // get connected detectors
  std::vector<uint32_t> SelectedDetIds;
  detCabling_->addActiveDetectorsRawIds(SelectedDetIds);
  int nDetsWithErr = 0;
  int nDetsTotal = 0;
  int nSubDetsWithErr[6] = {0, 0, 0, 0, 0, 0};
  int nSubDetsTotal[6]   = {0, 0, 0, 0, 0, 0};

  SiStripFolderOrganizer folder_organizer;
  for (std::vector<uint32_t>::const_iterator idetid=SelectedDetIds.begin(), iEnd=SelectedDetIds.end();idetid!=iEnd;++idetid){    
    uint32_t detId = *idetid;
    if (detId == 0 || detId == 0xFFFFFFFF){
      edm::LogError("SiStripAnalyser") 
                          << "SiStripAnalyser::fillGlobalStatus : " 
                          << "Wrong DetId !!!!!! " <<  detId << " Neglecting !!!!!! ";
      continue;
    }
    StripSubdetector subdet(*idetid);
    string dir_path;
    folder_organizer.getFolderName(detId, dir_path);     
    vector<MonitorElement*> detector_mes = dqmStore_->getContents(dir_path);
    int error_me = 0;
    for (vector<MonitorElement *>::const_iterator it = detector_mes.begin();
	 it!= detector_mes.end(); it++) {
      MonitorElement * me = (*it);     
      if (!me) continue;
      if (me->getQReports().size() == 0) continue;
      int istat =  SiStripUtility::getMEStatus((*it)); 
      if (istat == dqm::qstatus::ERROR)  error_me++;
    }
    nDetsTotal++;
        
    if (error_me > 0) {
     nDetsWithErr++;
    }
    switch (subdet.subdetId()) 
      {
      case StripSubdetector::TIB:
	{
	  nSubDetsTotal[0]++;
	  if (error_me > 0) nSubDetsWithErr[0]++;
	  break;       
	}
      case StripSubdetector::TID:
	{
	  TIDDetId tidId(detId);
	  if (tidId.isZPlusSide()) {
	    nSubDetsTotal[1]++;
	    if (error_me > 0) nSubDetsWithErr[1]++;
	  }  else if (tidId.isZMinusSide()) {
	    nSubDetsTotal[2]++;
	    if (error_me > 0) nSubDetsWithErr[2]++;
	  }
	  break;       
	}
      case StripSubdetector::TOB:
	{
	  nSubDetsTotal[3]++;
	  if (error_me > 0) nSubDetsWithErr[3]++;
	  break;       
	}
      case StripSubdetector::TEC:
	{
	  TECDetId tecId(detId);
	  if (tecId.isZPlusSide()) {
	    nSubDetsTotal[4]++;
	    if (error_me > 0) nSubDetsWithErr[4]++;
	  }  else if (tecId.isZMinusSide()) {
	    nSubDetsTotal[5]++;
	    if (error_me > 0) nSubDetsWithErr[5]++;
	  }
	  break;       
	}
      }
  }
  gStatus = (1 - nDetsWithErr*1.0/nDetsTotal) *100.0;
  dqmStore_->cd();
  MonitorElement* err_summ_me = dqmStore_->get("SiStrip/EventInfo/errorSummary");
  if(err_summ_me) err_summ_me->Fill(gStatus);
  for (unsigned int i = 0; i < 6; i++) {
    ostringstream hname;
    hname << setiosflags(ios::fixed);
    hname.fill('0');
    hname << "SiStrip/EventInfo/errorSummarySegments/Segment" << setw(2) << i;
    
    MonitorElement* seg_me = dqmStore_->get(hname.str());
    if (seg_me) {
      float eff = -1.0;
      if (nSubDetsTotal[i] > 0) {
	eff = (1 - nSubDetsWithErr[i] * 1.0 / nSubDetsTotal[i]) *100.0 ;
      }
      seg_me->Fill(eff);
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
