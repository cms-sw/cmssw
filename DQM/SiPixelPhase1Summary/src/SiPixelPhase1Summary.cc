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
#include <stdlib.h>
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
    fillSummaries(iBooker,iGetter);
    int lumiSec = lumiSeg.id().luminosityBlock();
    fillTrendPlots(iBooker,iGetter,lumiSec);
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
    fillSummaries(iBooker,iGetter);
    if (!runOnEndLumi_) fillTrendPlots(iBooker,iGetter); //If we're filling these plots at the end lumi step, it doesn't really make sense to also do them at the end job
  }

}

//------------------------------------------------------------------
// Used to book the summary plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::bookSummaries(DQMStore::IBooker & iBooker){

  iBooker.cd();

  std::vector<std::string> xAxisLabels_ = {"BMO","BMI","BPO ","BPI","HCMO_1","HCMO_2","HCMI_1","HCMI_2","HCPO_1","HCPO_2","HCPI_1","HCPI_2"}; // why not having a global variable !?!?!?! 
  std::vector<std::string> yAxisLabels_ = {"1","2","3","4"}; // why not having a global variable ?!?!?!!? - I originally did, but was told not to by David Lange!
    
  for (auto mapInfo: summaryPlotName_){
    auto name = mapInfo.first;
    if (name == "Grand") iBooker.setCurrentFolder("PixelPhase1/EventInfo");
    else iBooker.setCurrentFolder("PixelPhase1/Summary");
    summaryMap_[name] = iBooker.book2D("pixel"+name+"Summary","Pixel "+name+" Summary",12,0,12,4,0,4);
    for (unsigned int i = 0; i < xAxisLabels_.size(); i++){
      summaryMap_[name]->setBinLabel(i+1, xAxisLabels_[i],1);
    }
    for (unsigned int i = 0; i < yAxisLabels_.size(); i++){
      summaryMap_[name]->setBinLabel(i+1,yAxisLabels_[i],2);
    }
    summaryMap_[name]->setAxisTitle("Subdetector",1);
    summaryMap_[name]->setAxisTitle("Layer/disk",2);
    for (int i = 0; i < 12; i++){ // !??!?!? xAxisLabels_.size() ?!?!
      for (int j = 0; j < 4; j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
	summaryMap_[name]->Fill(i,j,-1.);
      }
    }
  }
}

//------------------------------------------------------------------
// Used to book the trend plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::bookTrendPlots(DQMStore::IBooker & iBooker){
  //We need different plots depending on if we're online (runOnEndLumi) or offline (!runOnEndLumi)
  if (runOnEndLumi_){
    deadROCTrends_[bpix] = iBooker.book1D("deadRocTrendBPix","BPIX dead ROC trend",500,0.,5000);
    deadROCTrends_[bpix]->setAxisTitle("Lumisection",1);
    deadROCTrends_[fpix] = iBooker.book1D("deadRocTrendFPix","FPIX dead ROC trend",500,0.,5000);
    deadROCTrends_[fpix]->setAxisTitle("Lumisection",1);
    ineffROCTrends_[bpix] = iBooker.book1D("ineffRocTrendBPix","BPIX inefficient ROC trend",500,0.,5000);
    ineffROCTrends_[bpix]->setAxisTitle("Lumisection",1);
    ineffROCTrends_[fpix] = iBooker.book1D("ineffRocTrendFPix","FPIX inefficient ROC trend",500,0.,5000);
    ineffROCTrends_[fpix]->setAxisTitle("Lumisection",1);
  }
  else {
    deadROCTrends_[offline] = iBooker.book1D("deadRocTotal","N dead ROCs",2,0,2);
    deadROCTrends_[offline]->setBinLabel(1,"Barrel");
    deadROCTrends_[offline]->setBinLabel(2,"Endcap");
    deadROCTrends_[offline]->setAxisTitle("Subdetector",1);
    ineffROCTrends_[offline] = iBooker.book1D("ineffRocTotal","N inefficient ROCs",2,0,2); 
    ineffROCTrends_[offline]->setBinLabel(1,"Barrel");
    ineffROCTrends_[offline]->setBinLabel(2,"Endcap");
    ineffROCTrends_[offline]->setAxisTitle("Subdetector",1);

  }
  
}

//------------------------------------------------------------------
// Fill the summary histograms
//------------------------------------------------------------------
void SiPixelPhase1Summary::fillSummaries(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter){
  //Firstly, we will fill the regular summary maps.
  for (auto mapInfo: summaryPlotName_){
    auto name = mapInfo.first;
    if (name == "Grand") continue;
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

	if (!summaryMap_[name]){
	  edm::LogWarning("SiPixelPhase1Summary") << "Summary map " << name << " is not available !!";
	  continue; // Based on reported errors it seems possible that we're trying to access a non-existant summary map, so if the map doesn't exist but we're trying to access it here we'll skip it instead.
	}
	if ((me->getQReports()).size()!=0) summaryMap_[name]->setBinContent(i+1,j+1,(me->getQReports())[0]->getQTresult());
	else summaryMap_[name]->setBinContent(i+1,j+1,-1);
      }  
    }    
  }
  //Now we will use the other summary maps to create the overall map.
  for (int i = 0; i < 12; i++){ // !??!?!? xAxisLabels_.size() ?!?!
    if (!summaryMap_["Grand"]){
      edm::LogWarning("SiPixelPhase1Summary") << "Grand summary does not exist!";
      break;
    }
    for (int j = 0; j < 4; j++){ // !??!?!? yAxisLabels_.size() ?!?!?!
      summaryMap_["Grand"]->setBinContent(i+1,j+1,1); // This resets the map to be good. We only then set it to 0 if there has been a problem in one of the other summaries.
      for (auto const mapInfo: summaryPlotName_){ //Check summary maps
	auto name = mapInfo.first;
	if (name == "Grand") continue;
	if (!summaryMap_[name]){
	  edm::LogWarning("SiPixelPhase1Summary") << "Summary " << name << " does not exist!";
	  continue;
	}
	if (summaryMap_["Grand"]->getBinContent(i+1,j+1) > summaryMap_[name]->getBinContent(i+1,j+1)) summaryMap_["Grand"]->setBinContent(i+1,j+1,summaryMap_[name]->getBinContent(i+1,j+1));
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

  std::ostringstream histNameStream;
  std::string histName;
  

  //Find the total number of filled bins and hi efficiency bins
  int nFilledROCsFPix = 0, nFilledROCsBPix = 0;
  int hiEffROCsFPix = 0, hiEffROCsBPix = 0;
  std::vector<int> nRocsPerLayer = {1536,3584,5632,8192};
  std::vector<int> nRocsPerRing = {4224,6528};
  //Loop over layers. This will also do the rings, but we'll skip the ring calculation for 
  for (auto it : {1,2,3,4}){

    iGetter.cd();
    histNameStream.str("");
    histNameStream << "PixelPhase1/Phase1_MechanicalView/PXBarrel/digi_occupancy_per_SignedModuleCoord_per_SignedLadderCoord_PXLayer_" << it;
    histName = histNameStream.str();
    MonitorElement * tempLayerME = iGetter.get(histName);
    if (!tempLayerME) continue;
    float lowEffValue = 0.25 * tempLayerME->getTH1()->Integral() / nRocsPerLayer[it-1];
    for (int i=1; i<=tempLayerME->getTH1()->GetXaxis()->GetNbins(); i++){
      for (int j=1; j<=tempLayerME->getTH1()->GetYaxis()->GetNbins(); j++){
	if (tempLayerME->getBinContent(i,j) > 0.) nFilledROCsBPix++;
	if (tempLayerME->getBinContent(i,j) > lowEffValue) hiEffROCsBPix++;
      }
    }
    if (runOnEndLumi_) tempLayerME->Reset(); //If we're doing online monitoring, reset the digi maps.
    if (it > 2) continue;
    //And now do the fpix if we're in the first 2 layers
    histNameStream.str("");
    histNameStream << "PixelPhase1/Phase1_MechanicalView/PXForward/digi_occupancy_per_SignedDiskCoord_per_SignedBladePanelCoord_PXRing_" << it;
    histName = histNameStream.str();
    MonitorElement * tempDiskME = iGetter.get(histName);
    lowEffValue = 0.25 * tempDiskME->getTH1()->Integral()/ nRocsPerRing[it-1];
    for (int i=1; i<=tempDiskME->getTH1()->GetXaxis()->GetNbins(); i++){
      for (int j=1; j<=tempDiskME->getTH1()->GetYaxis()->GetNbins(); j++){
	if (tempDiskME->getBinContent(i,j) > 0.) nFilledROCsFPix++;
	if (tempDiskME->getBinContent(i,j) > lowEffValue) hiEffROCsFPix++;
      }
    }
    if (runOnEndLumi_) tempLayerME->Reset();

      
  } // Close layers/ring loop
  
  if (!runOnEndLumi_) { //offline
    deadROCTrends_[offline]->setBinContent(1,18944-nFilledROCsBPix);
    deadROCTrends_[offline]->setBinContent(2,10752-nFilledROCsFPix);
    ineffROCTrends_[offline]->setBinContent(1,nFilledROCsBPix-hiEffROCsBPix);
    ineffROCTrends_[offline]->setBinContent(2,nFilledROCsFPix-hiEffROCsFPix);
  }
  else { //online
    deadROCTrends_[fpix]->setBinContent(lumiSec/10,10752-nFilledROCsFPix);
    deadROCTrends_[bpix]->setBinContent(lumiSec/10,18944-nFilledROCsBPix);
    ineffROCTrends_[fpix]->setBinContent(lumiSec/10,nFilledROCsFPix-hiEffROCsFPix);
    ineffROCTrends_[bpix]->setBinContent(lumiSec/10,nFilledROCsBPix-hiEffROCsBPix);
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1Summary);
