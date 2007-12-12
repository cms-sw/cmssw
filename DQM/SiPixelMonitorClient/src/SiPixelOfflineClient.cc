// -*- C++ -*-
//
// Package:    SiPixelMonitorClient
// Class:      SiPixelOfflineClient
// 
/**\class SiPixelOfflineClient SiPixelOfflineClient.cc DQM/SiPixelMonitorClient/src/SiPixelOfflineClient.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Samvel Khalatyan (ksamdev at gmail dot com)
//	   Created:  Wed Oct  5 16:42:34 CET 2006
// $Id: SiPixelOfflineClient.cc,v 1.3 2007/10/19 14:37:13 merkelp Exp $
//
//

// Root UI that is used by original Client's SiPixelActionExecuter
#include "DQM/SiPixelMonitorClient/interface/SiPixelOfflineClient.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "DQMServices/Core/interface/MonitorElementBaseT.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/WebPage.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelWebInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelTrackerMapCreator.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

#include <SealBase/Callback.h>

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <TF1.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
using edm::LogInfo;

/** 
* @brief 
*   Construct object
* 
* @param roPARAMETER_SET 
*   Regular Parameter Set that represent read configuration file
*/
SiPixelOfflineClient::SiPixelOfflineClient(const edm::ParameterSet& ps_):
  ModuleWeb("SiPixelOfflineClient")
  //: 
  //verbose_(ps_.getUntrackedParameter<bool>("verbose")),
  //save_(ps_.getUntrackedParameter<bool>("save")),
  //outFileName_(ps_.getUntrackedParameter<std::string>("outFileName")),
  //mui_(new MonitorUIRoot()),
  //ae_(){
  {
  parameters_ = ps_;
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  
  //verbose_ = parameters_.getUntrackedParameter<bool>("verbose");
  //verbose_ = true;
  //dbe_->setVerbose(verbose_);
  
  tkMapFreq_ = -1;
  barrelSumFreq_ = -1;
  endcapSumFreq_ = -1;
  barrelGrandSumFreq_ = -1;
  endcapGrandSumFreq_ = -1;
  saveFreq_ = parameters_.getUntrackedParameter<int>("FileSaveFrequency",500);
  
  //Instantiate Monitor UI in standalone mode:
  mui_ = new MonitorUIRoot();
 
  //Instantiate Web Interface:
  sipixelWebInterface_ = new SiPixelWebInterface("edanaClient","edanaClient",&mui_);
  defPageCreated_ = false;
   
  // Create MessageSender
  LogInfo("SiPixelOfflineClient")<<"Creating SiPixelOfflineClient!"<<"\n";
}

SiPixelOfflineClient::~SiPixelOfflineClient() {
  LogInfo("SiPixelOfflineClient")<<"Deleting SiPixelOfflineClient!"<<"\n";
  if(sipixelWebInterface_) delete sipixelWebInterface_;
  if(tkMapCreator_) delete tkMapCreator_;
  
//  delete mui_;
}

void SiPixelOfflineClient::beginJob( const edm::EventSetup& es_) {
  
  nevents_ = 0;
  run_ = 0;
  
  sipixelWebInterface_->readConfiguration(tkMapFreq_, 
				    barrelSumFreq_,
				    endcapSumFreq_,
				    barrelGrandSumFreq_,
				    endcapGrandSumFreq_,
				    messageLimit_,
				    sourceType_);
  cout<<"barrelSumFreq_="<<barrelSumFreq_<<" , endcapSumFreq_="<<
        endcapSumFreq_<<" , tkMapFreq_="<<tkMapFreq_<<endl;
  
  //collFlag_ = parameters_.getUntrackedParameter<int>("CollationFlag",0);
  
  //tkMapCreator_ = new SiPixelTrackerMapCreator();
  //if(tkMapCreator_->readConfiguration()) tkMapFreq_ = tkMapCreator_->getFrequency();
  
  //Setup quality tests:
  sipixelWebInterface_->setupQTests();
  
  //Do collation:
  //if(collFlag_==0){
  //  sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Collate);
  //  sipixelWebInterface_->performAction();
  //}
  
  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[beginJob] done";
}

void SiPixelOfflineClient::analyze(const edm::Event& evt_, 
				   const edm::EventSetup& es_){
  nevents_++;
  run_ = evt_.id().run();
  
  //if(nevents_<=3) return;
  
  cout<<" Analyze event number "<<nevents_<<endl;
  
  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[analyze] done";
}


void SiPixelOfflineClient::endLuminosityBlock(const edm::LuminosityBlock& lb_, 
				              const edm::EventSetup& es_){
  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[endLuminosityBlock] start";
  
  //Create summary ME's:
  //if(barrelSumFreq_!=-1 && nevents_%barrelSumFreq_==1){
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    sipixelWebInterface_->performAction();
  //}
  
  //Create TrackerMap:
  //if(tkMapFreq_!=-1 && nevents_%tkMapFreq_==1){
    //tkMapCreator_->create(dbe_);
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::CreateTkMap);
    sipixelWebInterface_->performAction();
  //}
  //Update Quality Test Results:
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    sipixelWebInterface_->performAction();
  
  //mui_->doMonitoring();
  //mui_->runQTests();

 /* LogInfo("SiPixelOfflineClient")
    << "Summary";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummary(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryLite";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryLite(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryXML";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryXML(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryXMLLite";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryXMLLite(mui_);*/


  //if(save_){
  //  ae_.saveMEs(mui_,outFileName_);
  //}
  // Save ME's into a file: 
  //if (saveFreq_!=-1 && nevents_%saveFreq_==1){
    //ostringstream fname;
    //fname<<"SiPixelOfflineClient_"<<run_<<".root";
    //cout<<"Saving ME's in "<<fname.str()<<endl;
    //sipixelWebInterface_->setOutputFileName(fname.str());
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::SaveData);
    //sipixelWebInterface_->performAction();
  //}  

  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[endLuminosityBlock] done";
}


void SiPixelOfflineClient::endJob(){
  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[endJob] start";

 /* ae_.createSummary(mui_);
  mui_->doMonitoring();
  mui_->runQTests();

  LogInfo("SiPixelOfflineClient")
    << "Summary";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummary(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryLite";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryLite(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryXML";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryXML(mui_);

  LogInfo("SiPixelOfflineClient")
    << "SummaryXMLLite";
  LogInfo("SiPixelOfflineClient")
    << ae_.getQTestSummaryXMLLite(mui_);


  if(save_) {
    ae_.saveMEs(mui_,outFileName_);
  }
*/
  //if(verbose_) LogInfo("SiPixelOfflineClient") << "[endJob] done";
}


void SiPixelOfflineClient::defaultWebPage(xgi::Input* in, xgi::Output* out){
  if(!defPageCreated_){
    static const int BUF_SIZE = 256;
    ifstream fin("loader.html", ios::in);
    if(!fin){
      cerr<<"Input file: loader.html could not be opened!"<<endl;
      return;
    }
    char buf[BUF_SIZE];
    ostringstream html_dump;
    while (fin.getline(buf,BUF_SIZE,'\n')){
      html_dump<<buf<<std::endl;
    }
    fin.close();
    
    *out<<html_dump.str()<<std::endl;
    defPageCreated_ = true;
  }
  
  sipixelWebInterface_->handleEDARequest(in,out,nevents_);
}
