#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
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
SiPixelEDAClient::SiPixelEDAClient(const edm::ParameterSet& ps) :
  ModuleWeb("SiPixelEDAClient"){
 //cout<<"Entering  SiPixelEDAClient::SiPixelEDAClient: "<<endl;
 
  edm::LogInfo("SiPixelEDAClient") <<  " Creating SiPixelEDAClient " << "\n" ;
  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  tkMapFrequency_   = -1;
  summaryFrequency_ = -1;

  // instantiate Monitor UI without connecting to any monitoring server
  // (i.e. "standalone mode")
  mui_ = new MonitorUIRoot();

  // instantiate web interface
  sipixelWebInterface_ = new SiPixelWebInterface("dummy", "dummy", &mui_);
  defaultPageCreated_ = false;
  
 //cout<<"...leaving  SiPixelEDAClient::SiPixelEDAClient. "<<endl;
}
//
// -- Destructor
//
SiPixelEDAClient::~SiPixelEDAClient(){
//  cout<<"Entering SiPixelEDAClient::~SiPixelEDAClient: "<<endl;
  
//  edm::LogInfo("SiPixelEDAClient") <<  " Deleting SiPixelEDAClient " << "\n" ;
//  if (sipixelWebInterface_) {
//     delete sipixelWebInterface_;
//     sipixelWebInterface_ = 0;
//  }
//  if (trackerMapCreator_) {
//    delete trackerMapCreator_;
//    trackerMapCreator_ = 0;
//  }

//  cout<<"...leaving SiPixelEDAClient::~SiPixelEDAClient. "<<endl;
}
//
// -- End Job
//
void SiPixelEDAClient::endJob(){
//  cout<<"In SiPixelEDAClient::endJob "<<endl;

}
//
// -- Begin Job
//
void SiPixelEDAClient::beginJob(const edm::EventSetup& eSetup){
  //cout<<"Entering SiPixelEDAClient::beginJob: "<<endl;

  nLumiBlock = 0;

  sipixelWebInterface_->readConfiguration(tkMapFrequency_,summaryFrequency_);
  edm::LogInfo("SiPixelEDAClient") << " Configuration files read out correctly" << "\n" ;
  //cout  << " Update Frequencies are " << tkMapFrequency_ << " " 
  //                                    << summaryFrequency_ << endl ;

          //collationFlag_ = parameters.getUntrackedParameter<int>("CollationtionFlag",0);
         //outputFilePath_ = parameters.getUntrackedParameter<string>("OutputFilePath",".");
  staticUpdateFrequency_ = parameters.getUntrackedParameter<int>("StaticUpdateFrequency",10);
 // trackerMapCreator_ = new SiPixelTrackerMapCreator();
//  if (trackerMapCreator_->readConfiguration()) {
//    tkMapFrequency_ = trackerMapCreator_->getFrequency();
 // }
  //cout<<"...leaving SiPixelEDAClient::beginJob. "<<endl;
}
//
//  -- Analyze 
//
void SiPixelEDAClient::analyze(const edm::Event& e, const edm::EventSetup& eSetup){
//  cout<<"In SiPixelEDAClient::analyze "<<endl;

}
//
// -- Begin  Luminosity Block
//
void SiPixelEDAClient::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& context) {
//  cout<<"Entering SiPixelEDAClient::beginLuminosityBlock: "<<endl;
  
  edm::LogVerbatim ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Begin of LS transition";

//  cout<<"...leaving SiPixelEDAClient::beginLuminosityBlock. "<<endl;
}
//
// -- End Luminosity Block
//
void SiPixelEDAClient::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  //cout<<"Entering SiPixelEDAClient::endLuminosityBlock: "<<endl;

  edm::LogVerbatim ("SiPixelEDAClient") <<"[SiPixelEDAClient]: End of LS transition, performing the DQM client operation";

  nLumiBlock++;

  //cout << "====================================================== " << endl;
  //cout << " ===> Iteration # " << nLumiBlock << " " 
  //                             << lumiSeg.luminosityBlock() << endl;
  //cout << "====================================================== " << endl;

//  if (nLumiBlock==2) {
//    cout << " Creating Collation " << endl;
//    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Collate);
//    sipixelWebInterface_->performAction();
//  }
  // -- Create summary monitor elements according to the frequency
//  if (summaryFrequency_ != -1 && nLumiBlock%summaryFrequency_ == 1) {
    //cout << " Creating Summary " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    sipixelWebInterface_->performAction();
//  }
  if (nLumiBlock==1) {
    //cout << " Setting up QTests " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::setupQTest);
    sipixelWebInterface_->performAction();
  }
    //cout << " Checking QTest results " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    sipixelWebInterface_->performAction();
  
  // -- Create TrackerMap  according to the frequency
//  if (tkMapFrequency_ != -1 && nLumiBlock%tkMapFrequency_ == 1) {
//    cout << " Creating Tracker Map " << endl;
//    trackerMapCreator_->create(dbe);
//    //sipixelWebInterface_->setTkMapFlag(true);
//
//  }
  // Create predefined plots
//  if (nLumiBlock%staticUpdateFrequency_  == 1) {
//    cout << " Creating predefined plots " << endl;
//    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::PlotHistogramFromLayout);
//    sipixelWebInterface_->performAction();
//  }

//  if ((nLumiBlock % fileSaveFrequency_) == 0) {
   // int iRun = lumiSeg.run();
   // int iLumi  = lumiSeg.luminosityBlock();
   // cout << " Saving histos " << endl;
   // saveAll(iRun, iLumi);
//  }
  //cout<<"...leaving SiPixelEDAClient::endLuminosityBlock. "<<endl;
}
//
// -- End Run
//
void SiPixelEDAClient::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
  //cout<<"Entering SiPixelEDAClient::endRun: "<<endl;

  //edm::LogVerbatim ("SiPixelEDAClient") <<"[SiPixelEDAClient]: End of Run, saving  DQM output ";
  //int iRun = run.run();
  
  //cout << " Updating Summary " << endl;
  sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
  sipixelWebInterface_->performAction();
  //cout << " Checking QTest results " << endl;
  sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
  sipixelWebInterface_->performAction();
  
  
  //saveAll(iRun, -1);

  //cout<<"...leaving SiPixelEDAClient::endRun. "<<endl;
}
//
// -- Save file
//
/*void SiPixelEDAClient::saveAll(int irun, int ilumi) {
//  cout<<"Entering SiPixelEDAClient::saveAll: "<<endl;

  ostringstream fname;
  if (ilumi != -1) {
    fname << outputFilePath_ << "/" << "SiPixel." << irun << "."<< ilumi << ".root";
  } else {
    fname << outputFilePath_ << "/" << "SiPixel." << irun << ".root";
  }
  //cout<<"Output filename = "<<fname.str()<<endl;
  sipixelWebInterface_->setOutputFileName(fname.str());
  sipixelWebInterface_->setActionFlag(SiPixelWebInterface::SaveData);
  sipixelWebInterface_->performAction();

//  cout<<"...leaving SiPixelEDAClient::saveAll. "<<endl;
}*/
//
// -- Create default web page
//
void SiPixelEDAClient::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
//  cout<<"Entering SiPixelEDAClient::defaultWebPage: "<<endl;
      
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
  sipixelWebInterface_->handleEDARequest(in, out, nLumiBlock);

//  cout<<"...leaving SiPixelEDAClient::defaultWebPage. "<<endl;
}
