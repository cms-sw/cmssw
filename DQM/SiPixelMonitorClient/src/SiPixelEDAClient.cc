#include "DQM/SiPixelMonitorClient/interface/SiPixelEDAClient.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "CondFormats/SiPixelObjects/interface/DetectorIndex.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelFrameConverter.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DQM/SiPixelMonitorClient/interface/SiPixelWebInterface.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDataQuality.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

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
SiPixelEDAClient::SiPixelEDAClient(const edm::ParameterSet& ps) :
  ModuleWeb("SiPixelEDAClient"){
// cout<<"Entering  SiPixelEDAClient::SiPixelEDAClient: "<<endl;
 
  edm::LogInfo("SiPixelEDAClient") <<  " Creating SiPixelEDAClient " << "\n" ;
  
  bei_ = Service<DQMStore>().operator->();

  summaryFrequency_      = ps.getUntrackedParameter<int>("SummaryCreationFrequency",20);
  tkMapFrequency_        = ps.getUntrackedParameter<int>("TkMapCreationFrequency",50); 
  staticUpdateFrequency_ = ps.getUntrackedParameter<int>("StaticUpdateFrequency",10);
  actionOnLumiSec_       = ps.getUntrackedParameter<bool>("ActionOnLumiSection",false); //client
  actionOnRunEnd_        = ps.getUntrackedParameter<bool>("ActionOnRunEnd",true); //client
  evtOffsetForInit_      = ps.getUntrackedParameter<int>("EventOffsetForInit",10); //client
  offlineXMLfile_        = ps.getUntrackedParameter<bool>("UseOfflineXMLFile",false); //client
  hiRes_                 = ps.getUntrackedParameter<bool>("HighResolutionOccupancy",false); //client
  noiseRate_             = ps.getUntrackedParameter<double>("NoiseRateCutValue",0.001); //client
  noiseRateDenominator_  = ps.getUntrackedParameter<int>("NEventsForNoiseCalculation",100000); //client
  Tier0Flag_             = ps.getUntrackedParameter<bool>("Tier0Flag",false); //client
  
  if(!Tier0Flag_){
    string localPath = string("DQM/SiPixelMonitorClient/test/loader.html");
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

  }
  
  // instantiate web interface
  sipixelWebInterface_ = new SiPixelWebInterface(bei_,offlineXMLfile_,Tier0Flag_);
  //instantiate the three work horses of the client:
  sipixelInformationExtractor_ = new SiPixelInformationExtractor(offlineXMLfile_);
  sipixelActionExecutor_ = new SiPixelActionExecutor(offlineXMLfile_, Tier0Flag_);
  sipixelDataQuality_ = new SiPixelDataQuality(offlineXMLfile_);
  
// cout<<"...leaving  SiPixelEDAClient::SiPixelEDAClient. "<<endl;
}
//
// -- Destructor
//
SiPixelEDAClient::~SiPixelEDAClient(){
//  cout<<"Entering SiPixelEDAClient::~SiPixelEDAClient: "<<endl;
  
  edm::LogInfo("SiPixelEDAClient") <<  " Deleting SiPixelEDAClient " << "\n" ;
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
// -- Begin Job
//
void SiPixelEDAClient::beginJob(const edm::EventSetup& eSetup){
//  cout<<"Entering SiPixelEDAClient::beginJob: "<<endl;

  // Read the summary configuration file
  if (!sipixelWebInterface_->readConfiguration(tkMapFrequency_,summaryFrequency_)) {
     edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Error to read configuration file!! Summary will not be produced!!!";
     summaryFrequency_ = -1;
     tkMapFrequency_ = -1;
     actionOnLumiSec_ = false;
     actionOnRunEnd_ = true;
     evtOffsetForInit_ = -1;
  }
  nLumiSecs_ = 0;
  nEvents_   = 0;
  if(Tier0Flag_) nFEDs_ = 40;
  
  bei_->setCurrentFolder("Pixel/");
  // Setting up QTests:
  sipixelActionExecutor_->setupQTests(bei_);
  // Creating Summary Histos:
  sipixelActionExecutor_->createSummary(bei_);
  // Creating occupancy plots:
  sipixelActionExecutor_->bookOccupancyPlots(bei_, hiRes_);
  // Booking noisy pixel ME's:
  sipixelInformationExtractor_->bookNoisyPixels(bei_, noiseRate_, Tier0Flag_);
  // Booking summary report ME's:
  sipixelDataQuality_->bookGlobalQualityFlag(bei_, Tier0Flag_, nFEDs_);

//  cout<<"...leaving SiPixelEDAClient::beginJob. "<<endl;
}
//
// -- Begin Run
//
void SiPixelEDAClient::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Begining of Run";

}
//
// -- Begin  Luminosity Block
//
void SiPixelEDAClient::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) {
//  cout<<"Entering SiPixelEDAClient::beginLuminosityBlock: "<<endl;
  
  edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Begin of LS transition";

//  cout<<"...leaving SiPixelEDAClient::beginLuminosityBlock. "<<endl;
}
//
//  -- Analyze 
//
void SiPixelEDAClient::analyze(const edm::Event& e, const edm::EventSetup& eSetup){
//  cout<<"[SiPixelEDAClient::analyze()] "<<endl;
  nEvents_++;  
  if(!Tier0Flag_){
   
    if(nEvents_==1){
      // check if any Pixel FED is in readout:
      edm::Handle<FEDRawDataCollection> rawDataHandle;
      e.getByLabel("source", rawDataHandle);
      const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
      nFEDs_ = 0;
      for(int i = 0; i != 40; i++){
        if(rawDataCollection.FEDData(i).size() && rawDataCollection.FEDData(i).data()) nFEDs_++;
      }
    }
    
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::CreatePlots);
    sipixelWebInterface_->performAction();
  }
  
}
//
// -- End Luminosity Block
//
void SiPixelEDAClient::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  //cout<<"Entering SiPixelEDAClient::endLuminosityBlock: "<<endl;

  edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: End of LS transition, performing the DQM client operation";

  nLumiSecs_++;
  //cout << "nLumiSecs_: "<< nLumiSecs_ << endl;
  
  edm::LogInfo("SiPixelEDAClient") << "====================================================== " << endl << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() << endl  << "====================================================== " << endl;

  if(actionOnLumiSec_ && nLumiSecs_ % 4 == 0 ){
    //cout << " Updating Summary " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    sipixelWebInterface_->performAction();
    //cout << " Checking QTest results " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    sipixelWebInterface_->performAction();
     //cout << " Updating occupancy plots" << endl;
    sipixelActionExecutor_->bookOccupancyPlots(bei_, hiRes_);
    sipixelActionExecutor_->createOccupancy(bei_);
    //cout  << " Checking Pixel quality flags " << endl;;
    bei_->cd();
    bool init=true;
    sipixelDataQuality_->computeGlobalQualityFlag(bei_,init,nFEDs_,Tier0Flag_);
    init=true;
    sipixelDataQuality_->fillGlobalQualityPlot(bei_,init,eSetup,nFEDs_,Tier0Flag_);
    //cout << " Checking for new noisy pixels " << endl;
    init=true;
    if(noiseRate_>=0.) sipixelInformationExtractor_->findNoisyPixels(bei_, init, noiseRate_, noiseRateDenominator_, eSetup);
  }   
         
  // -- Create TrackerMap  according to the frequency
//  if (tkMapFrequency_ != -1 && nLumiBlock%tkMapFrequency_ == 1) {
//    cout << " Creating Tracker Map " << endl;
//    trackerMapCreator_->create(bei_);
//    //sipixelWebInterface_->setTkMapFlag(true);
//
//  }
  // Create predefined plots
//  if (nLumiBlock%staticUpdateFrequency_  == 1) {
//    cout << " Creating predefined plots " << endl;
//    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::PlotHistogramFromLayout);
//    sipixelWebInterface_->performAction();
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
  
  if(actionOnRunEnd_){
    //cout << " Updating Summary " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    sipixelWebInterface_->performAction();
    //cout << " Checking QTest results " << endl;
    sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    sipixelWebInterface_->performAction();
    //cout << " Updating occupancy plots" << endl;
    sipixelActionExecutor_->bookOccupancyPlots(bei_, hiRes_);
    sipixelActionExecutor_->createOccupancy(bei_);
    //cout  << " Checking Pixel quality flags " << endl;;
    bei_->cd();
    bool init=true;
    sipixelDataQuality_->computeGlobalQualityFlag(bei_,init,nFEDs_,Tier0Flag_);
    init=true;
    sipixelDataQuality_->fillGlobalQualityPlot(bei_,init,eSetup,nFEDs_,Tier0Flag_);
    //cout << " Checking for new noisy pixels " << endl;
    init=true;
    if(noiseRate_>=0.) sipixelInformationExtractor_->findNoisyPixels(bei_, init, noiseRate_, noiseRateDenominator_, eSetup);

    // On demand, dump module ID's and stuff on the screen:
    //sipixelActionExecutor_->dumpModIds(bei_,eSetup);
  
  }
  
  //cout<<"...leaving SiPixelEDAClient::endRun. "<<endl;
}

//
// -- End Job
//
void SiPixelEDAClient::endJob(){
//  cout<<"In SiPixelEDAClient::endJob "<<endl;
  edm::LogInfo("SiPixelEDAClient") <<"[SiPixelEDAClient]: endjob called!";

}
//
// -- Create default web page
//
void SiPixelEDAClient::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
//  cout<<"Entering SiPixelEDAClient::defaultWebPage: "<<endl;
      
  bool isRequest = false;
  cgicc::Cgicc cgi(in);
  cgicc::CgiEnvironment cgie(in);
  //  edm::LogInfo("SiPixelEDAClient") <<"[SiPixelEDAClient]: defaultWebPage "
  //             << " query string : " << cgie.getQueryString();
  //  if ( xgi::Utils::hasFormElement(cgi,"ClientRequest") ) isRequest = true;
  string q_string = cgie.getQueryString();
  if (q_string.find("RequestID") != string::npos) isRequest = true;
  if (!isRequest) {    
    *out << html_out_.str() << std::endl;
  }  else {
    // Handles all HTTP requests of the form
    int iter = nEvents_/100;
    sipixelWebInterface_->handleEDARequest(in, out, iter);
  }

//  cout<<"...leaving SiPixelEDAClient::defaultWebPage. "<<endl;
}

