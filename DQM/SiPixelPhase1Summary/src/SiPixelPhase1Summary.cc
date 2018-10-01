// -*- C++ -*-
//
// Package:    SiPixelPhase1Summary
// Class:      SiPixelPhase1Summary
// 
/**\class 

 Description: Create the Phsae 1 pixel summary map

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Duncan Leggat
//         Created:  5th December 2016
//
//
#include "DQM/SiPixelPhase1Summary/interface/SiPixelPhase1Summary.h"
// Framework
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// DQM Framework
#include "DQM/SiPixelCommon/interface/SiPixelFolderOrganizer.h"
#include "DQMServices/Core/interface/DQMStore.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
// DataFormats
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelName.h"
#include "DataFormats/SiPixelDetId/interface/PixelBarrelNameUpgrade.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapName.h"
#include "DataFormats/SiPixelDetId/interface/PixelEndcapNameUpgrade.h"
//
#include <string>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace edm;

SiPixelPhase1Summary::SiPixelPhase1Summary(const edm::ParameterSet& iConfig) :
  conf_(iConfig),
  firstLumi(true)
{

   LogInfo ("PixelDQM") << "SiPixelPhase1Summary::SiPixelPhase1Summary: Got DQM BackEnd interface"<<endl;
   topFolderName_ = conf_.getParameter<std::string>("TopFolderName"); 
   runOnEndLumi_ = conf_.getParameter<bool>("RunOnEndLumi"); 
   runOnEndJob_ = conf_.getParameter<bool>("RunOnEndJob");

   std::vector<edm::ParameterSet> mapPSets = conf_.getParameter<std::vector<edm::ParameterSet> >("SummaryMaps");

   //Go through the configuration file and add in 
   for (auto const mapPSet : mapPSets){
     summaryPlotName_[mapPSet.getParameter<std::string>("MapName")] = mapPSet.getParameter<std::string>("MapHist");
   }
   deadRocThresholds_ = conf_.getParameter<std::vector<double> >("DeadROCErrorThreshold");
}

SiPixelPhase1Summary::~SiPixelPhase1Summary()
{
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  LogInfo ("PixelDQM") << "SiPixelPhase1Summary::~SiPixelPhase1Summary: Destructor"<<endl;
}

void SiPixelPhase1Summary::beginRun(edm::Run const& run, edm::EventSetup const& eSetup){
}

void SiPixelPhase1Summary::dqmEndLuminosityBlock(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, const edm::LuminosityBlock & lumiSeg, edm::EventSetup const& c){
  if (firstLumi){
    bookSummaries(iBooker);
    bookTrendPlots(iBooker);
    firstLumi = false;
  }

  if (runOnEndLumi_){
    int lumiSec = lumiSeg.id().luminosityBlock();
    fillTrendPlots(iBooker,iGetter,lumiSec);
    fillSummaries(iBooker,iGetter);
  }

  //  iBooker.cd();

}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelPhase1Summary::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter)
{
  if (firstLumi){ //Book the plots in the (maybe possible?) case that they aren't booked in the dqmEndLuminosityBlock method
    bookSummaries(iBooker);
    bookTrendPlots(iBooker);
    firstLumi = false;
  }
  if (runOnEndJob_){
    if (!runOnEndLumi_) fillTrendPlots(iBooker,iGetter); //If we're filling these plots at the end lumi step, it doesn't really make sense to also do them at the end job
    fillSummaries(iBooker,iGetter);

  }

}

//------------------------------------------------------------------
// Used to book the summary plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::bookSummaries(DQMStore::IBooker & iBooker){

  iBooker.cd();

  std::vector<std::string> xAxisLabels_ = {"BMO","BMI","BPO ","BPI","HCMO_1","HCMO_2","HCMI_1","HCMI_2","HCPO_1","HCPO_2","HCPI_1","HCPI_2"}; // why not having a global variable !?!?!?! 
  std::vector<std::string> yAxisLabels_ = {"1","2","3","4"}; // why not having a global variable ?!?!?!!? - I originally did, but was told not to by David Lange!
  
  iBooker.setCurrentFolder("PixelPhase1/Summary");
  //Book the summary plots for the variables as described in the config file  
  for (auto mapInfo: summaryPlotName_){
    auto name = mapInfo.first;
    summaryMap_[name] = iBooker.book2D("pixel"+name+"Summary","Pixel "+name+" Summary",12,0,12,4,0,4);
  }
  //Make the new 6 bin ROC summary
  deadROCSummary = iBooker.book2D("deadROCSummary","Percentage of dead ROCs per layer/ring",2,0,2,4,0,4);
  std::vector<std::string> xAxisLabelsReduced_ = {"Barrel","Forward"};
  deadROCSummary->setAxisTitle("Subdetector",1);
  for (unsigned int i = 0; i < xAxisLabelsReduced_.size(); i++){
    deadROCSummary->setBinLabel(i+1,xAxisLabelsReduced_[i]);
  }

  //Book the summary plot
  iBooker.setCurrentFolder("PixelPhase1/EventInfo");

  if (runOnEndLumi_){
  //New less granular summary plot - this is currently only done online
    summaryMap_["Grand"] = iBooker.book2D("reportSummaryMap","Pixel Summary Map",2,0,2,4,0,4);
    summaryMap_["Grand"]->setAxisTitle("Subdetector",1);
    for (unsigned int i = 0; i < xAxisLabelsReduced_.size(); i++){
      summaryMap_["Grand"]->setBinLabel(i+1,xAxisLabelsReduced_[i]);
      for (unsigned int j = 0; j < 4; j++){ summaryMap_["Grand"]->setBinContent(i,j,-1);}
    }
  }
  else{
    //Book the original summary plot, for now juts doing this one offline.
    summaryMap_["Grand"] = iBooker.book2D("reportSummaryMap","Pixel Summary Map",12,0,12,4,0,4);
  }
  
  reportSummary = iBooker.bookFloat("reportSummary");
  

  //Now set up axis and bin labels
  for (auto summaryMapEntry: summaryMap_){
    if (summaryMapEntry.first == "Grand") continue;
    auto summaryMap = summaryMapEntry.second;
    for (unsigned int i = 0; i < xAxisLabels_.size(); i++){
      summaryMap->setBinLabel(i+1, xAxisLabels_[i],1);
    }
    for (unsigned int i = 0; i < yAxisLabels_.size(); i++){
      summaryMap->setBinLabel(i+1,yAxisLabels_[i],2);
    }
    summaryMap->setAxisTitle("Subdetector",1);
    summaryMap->setAxisTitle("Layer/disk",2);
    for (int i = 0; i < summaryMap->getTH1()->GetXaxis()->GetNbins(); i++){ // !??!?!? xAxisLabels_.size() ?!?!
      for (int j = 0; j < summaryMap->getTH1()->GetYaxis()->GetNbins(); j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
	summaryMap->Fill(i,j,-1.);
      }
    }
  }
  reportSummary->Fill(-1.);

  //Reset the iBooker
  iBooker.setCurrentFolder("PixelPhase1/");
}

//------------------------------------------------------------------
// Used to book the trend plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::bookTrendPlots(DQMStore::IBooker & iBooker){
  //We need different plots depending on if we're online (runOnEndLumi) or offline (!runOnEndLumi)
  iBooker.setCurrentFolder("PixelPhase1/");
  std::vector<string> binAxisLabels = {"Layer 1", "Layer 2", "Layer 3", "Layer 4", "Ring 1", "Ring 2"};
  if (runOnEndLumi_){
    std::vector<trendPlots> histoOrder = {layer1,layer2,layer3,layer4,ring1,ring2};
    std::vector<string> varName ={"Layer_1","Layer_2","Layer_3","Layer_4","Ring_1","Ring_2"};
    std::vector<int> yMax = {1536,3584,5632,8192,4224,6528};
    for (unsigned int i = 0; i < histoOrder.size(); i++){
      string varNameStr = "deadRocTrend"+varName[i];
      string varTitle = binAxisLabels[i]+" dead ROC trend";
      deadROCTrends_[histoOrder[i]] = iBooker.bookProfile(varNameStr,varTitle,500,0.,5000,0.,yMax[i],"");  
      varNameStr = "ineffRocTrend"+varName[i];
      varTitle = binAxisLabels[i]+" inefficient ROC trend";
      ineffROCTrends_[histoOrder[i]] = iBooker.bookProfile(varNameStr,varTitle,500,0.,5000,0.,yMax[i],"");
      deadROCTrends_[histoOrder[i]]->setAxisTitle("Lumisection",1);
      ineffROCTrends_[histoOrder[i]]->setAxisTitle("Lumisection",1);
    }
  }
  else {
    deadROCTrends_[offline] = iBooker.bookProfile("deadRocTotal","N dead ROCs",6,0,6,0.,8192,"");
    ineffROCTrends_[offline] = iBooker.bookProfile("ineffRocTotal","N inefficient ROCs",6,0,6,0.,8192,""); 
    deadROCTrends_[offline]->setAxisTitle("Subdetector",1);
    ineffROCTrends_[offline]->setAxisTitle("Subdetector",1);
    for (unsigned int i = 1; i <= binAxisLabels.size(); i++){
      deadROCTrends_[offline]->setBinLabel(i,binAxisLabels[i-1]);
      ineffROCTrends_[offline]->setBinLabel(i,binAxisLabels[i-1]);
    }
  }
  
}

//------------------------------------------------------------------
// Fill the summary histograms
//------------------------------------------------------------------
void SiPixelPhase1Summary::fillSummaries(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter){
  //Firstly, we will fill the regular summary maps.
  for (auto mapInfo: summaryPlotName_){
    auto name = mapInfo.first;
    std::ostringstream histNameStream;
    std::string histName;

    for (int i = 0; i < 12; i++){ // !??!?!? xAxisLabels_.size() ?!?!
      for (int j = 0; j < 4; j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
	if (i > 3 && j == 3) continue;
	bool minus = i < 2  || (i > 3 && i < 8); // bleah !
	int iOver2 = floor(i/2.);
	bool outer = (i > 3)?iOver2%2==0:i%2==0;
	//Complicated expression that creates the name of the histogram we are interested in.
	histNameStream.str("");
	histNameStream << topFolderName_.c_str() << "PX" << ((i > 3)?"Forward":"Barrel") << "/" << ((i > 3)?"HalfCylinder":"Shell") << "_" << (minus?"m":"p") << ((outer)?"O":"I") << "/" << ((i > 3)?((i%2 == 0)?"PXRing_1/":"PXRing_2/"):"") << summaryPlotName_[name].c_str() << "_PX" << ((i > 3)?"Disk":"Layer") << "_" << ((i>3)?((minus)?"-":"+"):"") << (j+1);
	histName = histNameStream.str();
	MonitorElement * me = iGetter.get(histName);

	if (!me) {
	  edm::LogWarning("SiPixelPhase1Summary") << "ME " << histName << " is not available !!";
	  continue; // Ignore non-existing MEs, as this can cause the whole thing to crash
	}

	if (summaryMap_[name]==nullptr){
	  edm::LogWarning("SiPixelPhase1Summary") << "Summary map " << name << " is not available !!";
	  continue; // Based on reported errors it seems possible that we're trying to access a non-existant summary map, so if the map doesn't exist but we're trying to access it here we'll skip it instead.
	}
	if (!(me->getQReports()).empty()) summaryMap_[name]->setBinContent(i+1,j+1,(me->getQReports())[0]->getQTresult());
	else summaryMap_[name]->setBinContent(i+1,j+1,-1);
      }  
    }    
  }

  //Fill the dead ROC summary
  std::vector<trendPlots> trendOrder = {layer1,layer2,layer3,layer4,ring1,ring2};
  std::vector<int> nRocsPerTrend = {1536,3584,5632,8192,4224,6528};
  for (unsigned int i = 0; i < trendOrder.size(); i++){
    int xBin = i < 4 ? 1 : 2;
    int yBin = i%4 + 1;
    float nROCs = 0.;
    if (runOnEndLumi_){ //Online case
      TH1 * tempProfile = deadROCTrends_[trendOrder[i]]->getTH1();
      nROCs = tempProfile->GetBinContent(tempProfile->FindLastBinAbove());
    }
    else { //Offline case
      TH1* tempProfile = deadROCTrends_[offline]->getTH1();
      nROCs = tempProfile->GetBinContent(i+1);
    }
    deadROCSummary->setBinContent(xBin,yBin,nROCs/nRocsPerTrend[i]);
  }

  //Sum of non-negative bins for the reportSummary
  float sumOfNonNegBins = 0.;
  //Now we will use the other summary maps to create the overall map.
  //For now we only want to do this offline
  if (!runOnEndLumi_){
    for (int i = 0; i < 12; i++){ // !??!?!? xAxisLabels_.size() ?!?!
      if (summaryMap_["Grand"]==nullptr){
	edm::LogWarning("SiPixelPhase1Summary") << "Grand summary does not exist!";
	break;
      }
      for (int j = 0; j < 4; j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
	summaryMap_["Grand"]->setBinContent(i+1,j+1,1); // This resets the map to be good. We only then set it to 0 if there has been a problem in one of the other summaries.
	for (auto const mapInfo: summaryPlotName_){ //Check summary maps
	  auto name = mapInfo.first;
	  if (summaryMap_[name]==nullptr){
	    edm::LogWarning("SiPixelPhase1Summary") << "Summary " << name << " does not exist!";
	    continue;
	    }
	  if (summaryMap_["Grand"]->getBinContent(i+1,j+1) > summaryMap_[name]->getBinContent(i+1,j+1)) summaryMap_["Grand"]->setBinContent(i+1,j+1,summaryMap_[name]->getBinContent(i+1,j+1));
	}
	if (summaryMap_["Grand"]->getBinContent(i+1,j+1) > -0.1) sumOfNonNegBins += summaryMap_["Grand"]->getBinContent(i+1,j+1);
      }
    }
    reportSummary->Fill(sumOfNonNegBins/40.); // The average of the 40 useful bins in the summary map.
  }

  //Fill the new overall map
  //  if (!runOnEndLumi_) return;
  else{ //Do this for online only
    for (int i = 0; i < 2; i++){
      if (summaryMap_["Grand"]==nullptr){
	edm::LogWarning("SiPixelPhase1Summary") << "Grand summary does not exist!";
	break;
      }
      for (int j = 0; j < 4; j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
	//Ignore the bins without detectors in them
	if (i == 1 && j > 1) continue;
	summaryMap_["Grand"]->setBinContent(i+1,j+1,1); // This resets the map to be good. We only then set it to 0 if there has been a problem in one of the other summaries.
	if (deadROCSummary->getBinContent(i+1,j+1) > deadRocThresholds_[i*4+j]) summaryMap_["Grand"]->setBinContent(i+1,j+1,0);
	sumOfNonNegBins += summaryMap_["Grand"]->getBinContent(i+1,j+1);
      }
    }
  }

}

//------------------------------------------------------------------
// Fill the trend plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::fillTrendPlots(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, int lumiSec){

  // If we're running in online mode and the lumi section is not modulo 10, return. Offline running always uses lumiSec=0, so it will pass this test.
  if (lumiSec%10 != 0) return;

  if (runOnEndLumi_) {
    MonitorElement * nClustersAll = iGetter.get("PixelPhase1/Phase1_MechanicalView/num_clusters_per_Lumisection_PXAll");
    if (nClustersAll==nullptr){
      edm::LogWarning("SiPixelPhase1Summary") << "All pixel cluster trend plot not available!!";
      return;
    }
    if (nClustersAll->getTH1()->GetBinContent(lumiSec) < 100) return;
  }
  
  std::string histName;
  
  //Find the total number of filled bins and hi efficiency bins
  std::vector<trendPlots> trendOrder = {layer1,layer2,layer3,layer4,ring1,ring2};
  std::vector<int> nFilledROCs(trendOrder.size(),0);
  std::vector<int> hiEffROCs(trendOrder.size(),0);
  std::vector<int> nRocsPerTrend = {1536,3584,5632,8192,4224,6528};
  std::vector<string> trendNames = {};

  for (auto it : {1,2,3,4}) {
    histName = "PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_" + std::to_string(it);
    trendNames.push_back(histName);
  }
  for (auto it : {1,2}) {
    histName = "PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_" + std::to_string(it);
    trendNames.push_back(histName);
  }
  //Loop over layers. This will also do the rings, but we'll skip the ring calculation for 
  for (unsigned int trendIt = 0; trendIt < trendOrder.size(); trendIt++){
    iGetter.cd();
    histName = "PixelPhase1/Phase1_MechanicalView/" + trendNames[trendIt];
    MonitorElement * tempLayerME = iGetter.get(histName);
    if (tempLayerME==nullptr) continue;
    float lowEffValue = 0.25 * (tempLayerME->getTH1()->Integral() / nRocsPerTrend[trendIt]);
    for (int i=1; i<=tempLayerME->getTH1()->GetXaxis()->GetNbins(); i++){
      for (int j=1; j<=tempLayerME->getTH1()->GetYaxis()->GetNbins(); j++){
	if (tempLayerME->getBinContent(i,j) > 0.) nFilledROCs[trendIt]++;
	if (tempLayerME->getBinContent(i,j) > lowEffValue) hiEffROCs[trendIt]++;
      }
    }
    if (runOnEndLumi_) {
      tempLayerME->Reset(); //If we're doing online monitoring, reset the digi maps.
    }
  } // Close layers/ring loop
  
  if (!runOnEndLumi_) { //offline
    for (unsigned int i = 0; i < trendOrder.size(); i++){
      deadROCTrends_[offline]->Fill(i,nRocsPerTrend[i]-nFilledROCs[i]);
      ineffROCTrends_[offline]->Fill(i,nFilledROCs[i]-hiEffROCs[i]);
    }
  }
  else { //online
    for (unsigned int i = 0; i < trendOrder.size(); i++){
      deadROCTrends_[trendOrder[i]]->Fill(lumiSec-1,nRocsPerTrend[i]-nFilledROCs[i]);
      ineffROCTrends_[trendOrder[i]]->Fill(lumiSec-1,nFilledROCs[i]-hiEffROCs[i]);
    }
  }

  if (!runOnEndLumi_) return; // The following only occurs in the online
  //Reset some MEs every 10LS here
  for (auto it : {1,2,3,4}) { //PXBarrel
    histName = "PixelPhase1/Phase1_MechanicalView/PXBarrel/clusterposition_zphi_PXLayer_" +std::to_string(it);
    MonitorElement * toReset = iGetter.get(histName);
    if (toReset!=nullptr) {
      toReset->Reset();
    }
  }
  for (auto it : {"-3","-2","-1","+1","+2","+3"}){ //PXForward
    histName = "PixelPhase1/Phase1_MechanicalView/PXForward/clusterposition_xy_PXDisk_" + std::string(it);
    MonitorElement * toReset = iGetter.get(histName);
    if (toReset!=nullptr) {
      toReset->Reset();
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1Summary);
