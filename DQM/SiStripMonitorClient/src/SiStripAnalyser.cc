/*
 * \file SiStripAnalyser.cc
 * 
 * $Date: 2007/12/10 20:54:15 $
 * $Revision: 1.22 $
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

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/SiStripTrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include <SealBase/Callback.h>

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <iostream>
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
  
  // Get TkMap ParameterSet and Fed Cabling
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
  bei_ = Service<DaqMonitorBEInterface>().operator->();


  // instantiate web interface
  sistripWebInterface_ = new SiStripWebInterface(bei_);
  trackerMapCreator_ = 0;
  
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
  if (!sistripWebInterface_->readConfiguration()) {
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
				    << " Reading  new Cabling ";     
    if (trackerMapCreator_) delete trackerMapCreator_;
    trackerMapCreator_ = new SiStripTrackerMapCreator();
    if (!trackerMapCreator_->readConfiguration()) {
      edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Error to read configuration file!! TrackerMap will not be produced!!!";    
      tkMapFrequency_ = -1;
    }
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
}
//
// -- End Luminosity Block
//
void SiStripAnalyser::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {

  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of LS transition, performing the DQM client operation";

  nLumiSecs_++;

  eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
 
  cout << "====================================================== " << endl;
  cout << " ===> Iteration # " << nLumiSecs_ << " " 
                               << lumiSeg.luminosityBlock() << endl;
  cout << "====================================================== " << endl;
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%summaryFrequency_ == 0) {
    cout << " Creating Summary " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::Summary);
    sistripWebInterface_->performAction();
  }
  // -- Create TrackerMap  according to the frequency
  if (tkMapFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%tkMapFrequency_ == 0) {
    cout << " Creating Tracker Map " << endl;
    trackerMapCreator_->create(tkMapPSet_, fedCabling_, bei_);
  }
  // Create predefined plots
  if (staticUpdateFrequency_ != -1 && nLumiSecs_ > 0 && nLumiSecs_%staticUpdateFrequency_  == 0) {
    cout << " Creating predefined plots " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::PlotHistogramFromLayout);
    sistripWebInterface_->performAction();
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
    sistripWebInterface_->handleAnalyserRequest(in, out, iter);
  }
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
