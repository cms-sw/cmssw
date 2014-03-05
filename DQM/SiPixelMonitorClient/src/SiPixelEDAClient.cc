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
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
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

#include "DQM/SiPixelMonitorClient/interface/SiPixelActionExecutor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelInformationExtractor.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelDataQuality.h"
#include "DQM/SiPixelMonitorClient/interface/SiPixelUtility.h"

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
SiPixelEDAClient::SiPixelEDAClient(const edm::ParameterSet& ps) {
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
  doHitEfficiency_       = ps.getUntrackedParameter<bool>("DoHitEfficiency",true); //client
  inputSource_           = ps.getUntrackedParameter<string>("inputSource",  "source");
  isUpgrade_             = ps.getUntrackedParameter<bool>("isUpgrade",false); //client
  
  if(!Tier0Flag_){
    string localPath = string("DQM/SiPixelMonitorClient/test/loader.html");
    std::ifstream fin(edm::FileInPath(localPath).fullPath().c_str(), ios::in);
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
  //sipixelWebInterface_ = new SiPixelWebInterface(bei_,offlineXMLfile_,Tier0Flag_);
  //instantiate the three work horses of the client:
  sipixelInformationExtractor_ = new SiPixelInformationExtractor(offlineXMLfile_);
  sipixelActionExecutor_ = new SiPixelActionExecutor(offlineXMLfile_, Tier0Flag_);
  sipixelDataQuality_ = new SiPixelDataQuality(offlineXMLfile_);

  //set Token(-s)
  inputSourceToken_ = consumes<FEDRawDataCollection>(ps.getUntrackedParameter<string>("inputSource", "source")); 
// cout<<"...leaving  SiPixelEDAClient::SiPixelEDAClient. "<<endl;
}
//
// -- Destructor
//
SiPixelEDAClient::~SiPixelEDAClient(){
//  cout<<"Entering SiPixelEDAClient::~SiPixelEDAClient: "<<endl;
  
  edm::LogInfo("SiPixelEDAClient") <<  " Deleting SiPixelEDAClient " << "\n" ;
  /* Removing xdaq deps
  if (sipixelWebInterface_) {
     delete sipixelWebInterface_;
     sipixelWebInterface_ = 0;
  }
  */
  if (sipixelInformationExtractor_) {
     delete sipixelInformationExtractor_;
     sipixelInformationExtractor_ = 0;
  }
  if (sipixelActionExecutor_) {
     delete sipixelActionExecutor_;
     sipixelActionExecutor_ = 0;
  }
  if (sipixelDataQuality_) {
     delete sipixelDataQuality_;
     sipixelDataQuality_ = 0;
  }

//  cout<<"...leaving SiPixelEDAClient::~SiPixelEDAClient. "<<endl;
}
//
// -- Begin Job
//
void SiPixelEDAClient::beginJob(){
  firstRun = true;
}
//
// -- Begin Run
//
void SiPixelEDAClient::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Begining of Run";
//  cout<<"Entering SiPixelEDAClient::beginRun: "<<endl;

  if(firstRun){
  
  // Read the summary configuration file
    //if (!sipixelWebInterface_->readConfiguration(tkMapFrequency_,summaryFrequency_)) {
    // edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Error to read configuration file!! Summary will not be produced!!!";
    //}
    summaryFrequency_ = -1;
    tkMapFrequency_ = -1;
    actionOnLumiSec_ = false;
    actionOnRunEnd_ = true;
    evtOffsetForInit_ = -1;

  nLumiSecs_ = 0;
  nEvents_   = 0;
  if(Tier0Flag_) nFEDs_ = 40;
  else nFEDs_ = 0;
  
  bei_->setCurrentFolder("Pixel/");
  // Setting up QTests:
//  sipixelActionExecutor_->setupQTests(bei_);
  // Creating Summary Histos:
  sipixelActionExecutor_->createSummary(bei_, isUpgrade_);
  // Booking Deviation Histos:
  if(!Tier0Flag_) sipixelActionExecutor_->bookDeviations(bei_, isUpgrade_);
  // Booking Efficiency Histos:
  if(doHitEfficiency_) sipixelActionExecutor_->bookEfficiency(bei_, isUpgrade_);
  // Creating occupancy plots:
  sipixelActionExecutor_->bookOccupancyPlots(bei_, hiRes_);
  // Booking noisy pixel ME's:
  if(noiseRate_>0.) sipixelInformationExtractor_->bookNoisyPixels(bei_, noiseRate_, Tier0Flag_);
  // Booking summary report ME's:
  sipixelDataQuality_->bookGlobalQualityFlag(bei_, Tier0Flag_, nFEDs_);
  // Booking Static Tracker Maps:
//  sipixelActionExecutor_->bookTrackerMaps(bei_, "adc");
//  sipixelActionExecutor_->bookTrackerMaps(bei_, "charge");
//  sipixelActionExecutor_->bookTrackerMaps(bei_, "ndigis");
//  sipixelActionExecutor_->bookTrackerMaps(bei_, "NErrors");
  
  firstRun = false;
  }

//  cout<<"...leaving SiPixelEDAClient::beginRun. "<<endl;

}
//
// -- Begin  Luminosity Block
//
void SiPixelEDAClient::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) {
//  cout<<"Entering SiPixelEDAClient::beginLuminosityBlock: "<<endl;
  
  edm::LogInfo ("SiPixelEDAClient") <<"[SiPixelEDAClient]: Begin of LS transition";

  nEvents_lastLS_=0; nErrorsBarrel_lastLS_=0; nErrorsEndcap_lastLS_=0;
  MonitorElement * me = bei_->get("Pixel/AdditionalPixelErrors/byLumiErrors");
  if(me){
    nEvents_lastLS_ = int(me->getBinContent(0));
    nErrorsBarrel_lastLS_ = int(me->getBinContent(1));
    nErrorsEndcap_lastLS_ = int(me->getBinContent(2));
    //std::cout<<"Nevts in lastLS in EDAClient: "<<nEvents_lastLS_<<" "<<nErrorsBarrel_lastLS_<<" "<<nErrorsEndcap_lastLS_<<std::endl;
    me->Reset();
  }
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
      e.getByToken(inputSourceToken_, rawDataHandle);
      if(!rawDataHandle.isValid()){
        edm::LogInfo("SiPixelEDAClient") << inputSource_ << " is empty";
	return;
      } 
      const FEDRawDataCollection& rawDataCollection = *rawDataHandle;
      nFEDs_ = 0;
      for(int i = 0; i != 40; i++){
        if(rawDataCollection.FEDData(i).size() && rawDataCollection.FEDData(i).data()) nFEDs_++;
      }
    }
    
    // This is needed for plotting with the Pixel Expert GUI (interactive client):
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::CreatePlots);
    //sipixelWebInterface_->performAction();
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
  
  //if(actionOnLumiSec_ && !Tier0Flag_ && nLumiSecs_ % 1 == 0 ){
  if(actionOnLumiSec_ && nLumiSecs_ % 1 == 0 ){
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    //sipixelWebInterface_->performAction();
     //cout << " Updating efficiency plots" << endl;
    if(doHitEfficiency_) sipixelActionExecutor_->createEfficiency(bei_, isUpgrade_);
    //cout << " Checking QTest results " << endl;
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    //sipixelWebInterface_->performAction();
     //cout << " Updating occupancy plots" << endl;
    sipixelActionExecutor_->createOccupancy(bei_);
    //cout  << " Checking Pixel quality flags " << endl;;
    bei_->cd();
    bool init=true;
    //sipixelDataQuality_->computeGlobalQualityFlag(bei_,init,nFEDs_,Tier0Flag_);
    sipixelDataQuality_->computeGlobalQualityFlagByLumi(bei_,init,nFEDs_,Tier0Flag_,nEvents_lastLS_,nErrorsBarrel_lastLS_,nErrorsEndcap_lastLS_);
    init=true;
    bei_->cd();
    sipixelDataQuality_->fillGlobalQualityPlot(bei_,init,eSetup,nFEDs_,Tier0Flag_,nLumiSecs_);
    //cout << " Checking for new noisy pixels " << endl;
    init=true;
    if(noiseRate_>=0.) sipixelInformationExtractor_->findNoisyPixels(bei_, init, noiseRate_, noiseRateDenominator_, eSetup);
    // cout << "*** Creating Tracker Map Histos for End Run ***" << endl;
//    sipixelActionExecutor_->createMaps(bei_, "adc_siPixelDigis", "adc", Mean);
//    sipixelActionExecutor_->createMaps(bei_, "charge_siPixelClusters", "charge", Mean);
//    sipixelActionExecutor_->createMaps(bei_, "ndigis_siPixelDigis", "ndigis", WeightedSum);
//    sipixelActionExecutor_->createMaps(bei_, "NErrors_siPixelDigis", "NErrors", WeightedSum);
    // cout << "*** Done with Tracker Map Histos for End Run ***" << endl;
  }   
         
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
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::Summary);
    //sipixelWebInterface_->performAction();
    sipixelActionExecutor_->createSummary(bei_, isUpgrade_);
     //cout << " Updating efficiency plots" << endl;
    if(doHitEfficiency_) sipixelActionExecutor_->createEfficiency(bei_, isUpgrade_);
    //cout << " Checking QTest results " << endl;
    //sipixelWebInterface_->setActionFlag(SiPixelWebInterface::QTestResult);
    //sipixelWebInterface_->performAction();
    //cout << " Updating occupancy plots" << endl;
    sipixelActionExecutor_->createOccupancy(bei_);
    //cout  << " Checking Pixel quality flags " << endl;;
    bei_->cd();
    bool init=true;
    sipixelDataQuality_->computeGlobalQualityFlag(bei_,init,nFEDs_,Tier0Flag_);
    init=true;
    bei_->cd();
    //cout  << " Making run end reportSummaryMap: " <<nFEDs_<< endl;;
    sipixelDataQuality_->fillGlobalQualityPlot(bei_,init,eSetup,nFEDs_,Tier0Flag_,nLumiSecs_);
    //cout << " Checking for new noisy pixels " << endl;
    init=true;
    if(noiseRate_>=0.) sipixelInformationExtractor_->findNoisyPixels(bei_, init, noiseRate_, noiseRateDenominator_, eSetup);
    // cout << "*** Creating Tracker Map Histos for End Run ***" << endl;
//    sipixelActionExecutor_->createMaps(bei_, "adc_siPixelDigis", "adc", Mean);
//    sipixelActionExecutor_->createMaps(bei_, "charge_siPixelClusters", "charge", Mean);
//    sipixelActionExecutor_->createMaps(bei_, "ndigis_siPixelDigis", "ndigis", WeightedSum);
//    sipixelActionExecutor_->createMaps(bei_, "NErrors_siPixelDigis", "NErrors", WeightedSum);
    // cout << "*** Done with Tracker Map Histos for End Run ***" << endl;

    // On demand, dump module ID's and stuff on the screen:
    //sipixelActionExecutor_->dumpModIds(bei_,eSetup);
    // On demand, dump summary histo values for reference on the screen:
    //sipixelActionExecutor_->dumpRefValues(bei_,eSetup);
  }
  
//  cout<<"...leaving SiPixelEDAClient::endRun. "<<endl;
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
/* removing xdaq deps
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
*/
