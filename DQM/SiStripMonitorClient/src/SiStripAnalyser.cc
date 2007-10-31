/*
 * \file SiStripAnalyser.cc
 * 
 * $Date: 2007/10/24 17:13:25 $
 * $Revision: 1.14 $
 * \author  S. Dutta INFN-Pisa
 *
 */


#include "DQM/SiStripMonitorClient/interface/SiStripAnalyser.h"


#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"


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

using namespace edm;
using namespace std;
//
// -- Constructor
//
SiStripAnalyser::SiStripAnalyser(const edm::ParameterSet& ps) :
  DQMAnalyzer(ps),
  ModuleWeb("SiStripAnalyser"){
  
  edm::LogInfo("SiStripAnalyser") << " SiStripAnalyser::Creating SiStripAnalyser ";

  tkMapFrequency_   = -1;
  summaryFrequency_ = -1;
  fileSaveFrequency_ = parameters_.getUntrackedParameter<int>("FileSaveFrequency",50); 

  // instantiate Monitor UI without connecting to any monitoring server
  // (i.e. "standalone mode")
  mui_ = new MonitorUIRoot();

  // instantiate web interface
  sistripWebInterface_ = new SiStripWebInterface("dummy", "dummy", &mui_);
  defaultPageCreated_ = false;
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
void SiStripAnalyser::beginJob(const edm::EventSetup& eSetup){

  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginJob(eSetup);

  sistripWebInterface_->readConfiguration(summaryFrequency_);
  edm::LogInfo("SiStripAnalyser") << " Configuration files read out correctly" 
                                  << "\n" ;
  cout  << " Update Frequencies are " << tkMapFrequency_ << " " 
                                      << summaryFrequency_ << endl ;

          collationFlag_ = parameters_.getUntrackedParameter<int>("CollationtionFlag",0);
         outputFilePath_ = parameters_.getUntrackedParameter<string>("OutputFilePath",".");
  staticUpdateFrequency_ = parameters_.getUntrackedParameter<int>("StaticUpdateFrequency",10);
  // Get Fed cabling
  eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
  trackerMapCreator_ = new SiStripTrackerMapCreator();
  if (trackerMapCreator_->readConfiguration()) {
    tkMapFrequency_ = trackerMapCreator_->getFrequency();
  }
}
//
// -- Begin Run
//
void SiStripAnalyser::beginRun(const Run& run, const edm::EventSetup& eSetup) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginRun(run, eSetup);

  // then do your thing
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: Begining of Run";
}
//
// -- Begin Luminosity Block
//
void SiStripAnalyser::beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& eSetup) {
  // call DQMAnalyzer in the beginning 
  DQMAnalyzer::beginLuminosityBlock(lumiSeg,eSetup);

  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: Begin of LS transition";
}
//
//  -- Analyze 
//
void SiStripAnalyser::analyze(const edm::Event& e, const edm::EventSetup& eSetup){
  // call DQMAnalyzer some place
  DQMAnalyzer::analyze(e,eSetup);
}
//
// -- End Luminosity Block
//
void SiStripAnalyser::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {

  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of LS transition, performing the DQM client operation";

  int nLumiSecs = DQMAnalyzer::getNumLumiSecs();

  if (nLumiSecs%prescaleLS_ != 0 ) return;

  eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
 
  cout << "====================================================== " << endl;
  cout << " ===> Iteration # " << nLumiSecs << " " 
                               << lumiSeg.luminosityBlock() << endl;
  cout << "====================================================== " << endl;
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nLumiSecs > 1 && nLumiSecs%summaryFrequency_ == 1) {
    cout << " Creating Summary " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::Summary);
    sistripWebInterface_->performAction();
  }
  // -- Create TrackerMap  according to the frequency
  if (tkMapFrequency_ != -1 && nLumiSecs > 1 && nLumiSecs%tkMapFrequency_ == 1) {
    cout << " Creating Tracker Map " << endl;
    //    trackerMapCreator_->create(dbe_);
    trackerMapCreator_->create(fedCabling_, dbe_);
    sistripWebInterface_->setTkMapFlag(true);
  }
  // Create predefined plots
  if (nLumiSecs > 1 && nLumiSecs%staticUpdateFrequency_  == 1) {
    cout << " Creating predefined plots " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::PlotHistogramFromLayout);
    sistripWebInterface_->performAction();
  }

  // Save MEs in a file
  if ((nLumiSecs % fileSaveFrequency_) == 0) DQMAnalyzer::save();
  

  // call endLuminosityBlock at the end 
  DQMAnalyzer::endLuminosityBlock(lumiSeg,eSetup);
}

//
// -- End Run
//
void SiStripAnalyser::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  edm::LogInfo ("SiStripAnalyser") <<"SiStripAnalyser:: End of Run";
  DQMAnalyzer::endRun(run, eSetup);
}
//
// -- End Job
//
void SiStripAnalyser::endJob(){
  // do your thing here
  edm::LogInfo("SiStripAnalyser") <<"SiStripAnalyser:: endjob called!";

  // call DQMAnalyzer in the end
   DQMAnalyzer::endJob();
}
//
// -- Create default web page
//
void SiStripAnalyser::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
      
  if (!defaultPageCreated_) {
    static const int BUF_SIZE = 256;
    ifstream fin("loader.html", ios::in);
    if (!fin) {
      cerr << "Input File: loader.html"<< " could not be opened!" << endl;
      return;
    }
    char buf[BUF_SIZE];
    ostringstream html_dump;
    while (fin.getline(buf, BUF_SIZE, '\n')) { // pops off the newline character 
      html_dump << buf << std::endl;
    }
    fin.close();
    
    *out << html_dump.str() << std::endl;
    defaultPageCreated_ = true;
  }
  
  // Handles all HTTP requests of the form
  sistripWebInterface_->handleAnalyserRequest(in, out,DQMAnalyzer::getNumLumiSecs());

}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripAnalyser);
