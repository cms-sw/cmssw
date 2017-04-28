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

void SiPixelPhase1Summary::dqmEndLuminosityBlock(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter, edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& c){
  if (firstLumi){
    bookSummaries(iBooker);
    firstLumi = false;
  }

  if (runOnEndLumi_) fillSummaries(iBooker,iGetter);

  //  iBooker.cd();

}

//------------------------------------------------------------------
// Method called for every event
//------------------------------------------------------------------
void SiPixelPhase1Summary::dqmEndJob(DQMStore::IBooker & iBooker, DQMStore::IGetter & iGetter)
{

  if (runOnEndJob_) fillSummaries(iBooker,iGetter);

}

//------------------------------------------------------------------
// Used to book the summary plots
//------------------------------------------------------------------
void SiPixelPhase1Summary::bookSummaries(DQMStore::IBooker & iBooker){
  iBooker.setCurrentFolder("PixelPhase1/Summary");

  std::vector<std::string> xAxisLabels_ = {"BMO","BMI","BPO ","BPI","HCMO_1","HCMO_2","HCMI_1","HCMI_2","HCPO_1","HCPO_2","HCPI_1","HCPI_2"};
  std::vector<std::string> yAxisLabels_ = {"1","2","3","4"};
    
  for (auto mapInfo: summaryPlotName_){
    auto name = mapInfo.first;
    summaryMap_[name] = iBooker.book2D("pixel"+name+"Summary","Pixel "+name+" Summary",12,0,12,4,0,4);
    for (unsigned int i = 0; i < xAxisLabels_.size(); i++){
      summaryMap_[name]->setBinLabel(i+1, xAxisLabels_[i],1);
    }
    for (unsigned int i = 0; i < yAxisLabels_.size(); i++){
      summaryMap_[name]->setBinLabel(i+1,yAxisLabels_[i],2);
    }
    summaryMap_[name]->setAxisTitle("Subdetector",1);
    summaryMap_[name]->setAxisTitle("Layer/disk",2);
    for (int i = 0; i < 12; i++){
      for (int j = 0; j < 4; j++){
	summaryMap_[name]->Fill(i,j,-1.);
      }
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
    if (name == "Grand") continue;
    std::ostringstream histNameStream;
    std::string histName;

    for (int i = 0; i < 12; i++){
      for (int j = 0; j < 4; j++){
	if (i > 3 && j == 3) continue;
	bool minus = i < 2  || (i > 3 && i < 8);
	int iOver2 = floor(i/2.);
	bool outer = (i > 3)?iOver2%2==0:i%2==0;
	//Complicated expression that creates the name of the histogram we are interested in.
	histNameStream.str("");
	histNameStream << topFolderName_.c_str() << "PX" << ((i > 3)?"Forward":"Barrel") << "/" << ((i > 3)?"HalfCylinder":"Shell") << "_" << (minus?"m":"p") << ((outer)?"O":"I") << "/" << ((i > 3)?((i%2 == 0)?"PXRing_1/":"PXRing_2/"):"") << summaryPlotName_[name].c_str() << "_PX" << ((i > 3)?"Disk":"Layer") << "_" << ((i>3)?((minus)?"-":"+"):"") << (j+1);
	histName = histNameStream.str();
	//	std::cout << histName << std::endl;
	MonitorElement * me = iGetter.get(histName);
	
	if (!me) continue; //Ignore non-existant MEs
	      
	if (me->hasError()) {
	  //If there is an error, fill with 0
	  summaryMap_[name]->setBinContent(i+1,j+1,0);
	} //Do we want to include warnings here?
	else if (me->hasWarning()){
	  summaryMap_[name]->setBinContent(i+1,j+1,0.5);
	}
	else summaryMap_[name]->setBinContent(i+1,j+1,1);
      }
    }  
  }    
  //Now we will use the other summary maps to create the overall map.
  for (int i = 0; i < 12; i++){
    for (int j = 0; j < 4; j++){
      summaryMap_["Grand"]->setBinContent(i+1,j+1,1); // This resets the map to be good. We only then set it to 0 if there has been a problem in one of the other summaries.
      for (auto const mapInfo: summaryPlotName_){ //Check summary maps
	auto name = mapInfo.first;
	if (name == "Grand") continue;
	if (summaryMap_[name]->getBinContent(i+1,j+1) < 0.9 && summaryMap_["Grand"]->getBinContent(i+1,j+1) > summaryMap_[name]->getBinContent(i+1,j+1)) summaryMap_["Grand"]->setBinContent(i+1,j+1,summaryMap_[name]->getBinContent(i+1,j+1)); // This could be changed to include warnings if we want?
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(SiPixelPhase1Summary);
