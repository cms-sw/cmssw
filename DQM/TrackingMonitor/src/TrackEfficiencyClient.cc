/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/10/16 10:07:41 $
 *  $Revision: 1.3 $
 *  \author Anne-Catherine Le Bihan
 */

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/TrackingMonitor/interface/TrackEfficiencyClient.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

//-----------------------------------------------------------------------------------
TrackEfficiencyClient::TrackEfficiencyClient(edm::ParameterSet const& iConfig) 
//-----------------------------------------------------------------------------------
{
  edm::LogInfo( "TrackEfficiencyClient") << "TrackEfficiencyClient::Deleting TrackEfficiencyClient ";

  // get back-end interface
  dqmStore_ = edm::Service<DQMStore>().operator->();
  
  FolderName_      = iConfig.getParameter<std::string>("FolderName");
  algoName_        = iConfig.getParameter<std::string>("AlgoName");
  trackEfficiency_ = iConfig.getParameter<bool>("trackEfficiency");
  
  conf_ = iConfig;
}


//-----------------------------------------------------------------------------------
TrackEfficiencyClient::~TrackEfficiencyClient() 
//-----------------------------------------------------------------------------------
{
  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::Deleting TrackEfficiencyClient ";
}


//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::beginJob(void) 
//-----------------------------------------------------------------------------------
{  
 
  dqmStore_->setCurrentFolder(FolderName_);
  
  //
  int    effXBin = conf_.getParameter<int>   ("effXBin");
  double effXMin = conf_.getParameter<double>("effXMin");
  double effXMax = conf_.getParameter<double>("effXMax");
 
  histName = "effX_";
  effX = dqmStore_->book1D(histName+algoName_, histName+algoName_, effXBin, effXMin, effXMax);
  if (effX->getTH1F()) effX->getTH1F()->Sumw2();
  effX->setAxisTitle("");
  

  //
  int    effYBin = conf_.getParameter<int>   ("effYBin");
  double effYMin = conf_.getParameter<double>("effYMin");
  double effYMax = conf_.getParameter<double>("effYMax");
 
  histName = "effY_";
  effY = dqmStore_->book1D(histName+algoName_, histName+algoName_, effYBin, effYMin, effYMax);
  if (effY->getTH1F()) effY->getTH1F()->Sumw2();
  effY->setAxisTitle("");
  
  //
  int    effZBin = conf_.getParameter<int>   ("effZBin");
  double effZMin = conf_.getParameter<double>("effZMin");
  double effZMax = conf_.getParameter<double>("effZMax");
 
  histName = "effZ_";
  effZ = dqmStore_->book1D(histName+algoName_, histName+algoName_, effZBin, effZMin, effZMax);
  if (effZ->getTH1F()) effZ->getTH1F()->Sumw2();
  effZ->setAxisTitle("");

  //
  int    effEtaBin = conf_.getParameter<int>   ("effEtaBin");
  double effEtaMin = conf_.getParameter<double>("effEtaMin");
  double effEtaMax = conf_.getParameter<double>("effEtaMax");
 
  histName = "effEta_";
  effEta = dqmStore_->book1D(histName+algoName_, histName+algoName_, effEtaBin, effEtaMin, effEtaMax);
  if (effEta->getTH1F()) effEta->getTH1F()->Sumw2();
  effEta->setAxisTitle("");
  
  //
  int    effPhiBin = conf_.getParameter<int>   ("effPhiBin");
  double effPhiMin = conf_.getParameter<double>("effPhiMin");
  double effPhiMax = conf_.getParameter<double>("effPhiMax");
 
  histName = "effPhi_";
  effPhi = dqmStore_->book1D(histName+algoName_, histName+algoName_, effPhiBin, effPhiMin, effPhiMax);
  if (effPhi->getTH1F()) effPhi->getTH1F()->Sumw2();
  effPhi->setAxisTitle("");
  
  //
  int    effD0Bin = conf_.getParameter<int>   ("effD0Bin");
  double effD0Min = conf_.getParameter<double>("effD0Min");
  double effD0Max = conf_.getParameter<double>("effD0Max");
 
  histName = "effD0_";
  effD0 = dqmStore_->book1D(histName+algoName_, histName+algoName_, effD0Bin, effD0Min, effD0Max);
  if (effD0->getTH1F()) effD0->getTH1F()->Sumw2();
  effD0->setAxisTitle("");
  
 
  //
  int    effCompatibleLayersBin = conf_.getParameter<int>   ("effCompatibleLayersBin");
  double effCompatibleLayersMin = conf_.getParameter<double>("effCompatibleLayersMin");
  double effCompatibleLayersMax = conf_.getParameter<double>("effCompatibleLayersMax");
 
  histName = "effCompatibleLayers_";
  effCompatibleLayers = dqmStore_->book1D(histName+algoName_, histName+algoName_, effCompatibleLayersBin, effCompatibleLayersMin, effCompatibleLayersMax);
  if (effCompatibleLayers->getTH1F()) effCompatibleLayers->getTH1F()->Sumw2();
  effCompatibleLayers->setAxisTitle("");

  edm::LogInfo("TrackEfficiencyClient") << "TrackEfficiencyClient::beginJob done";
}


//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) 
//-----------------------------------------------------------------------------------
{
  edm::LogInfo ("TrackEfficiencyClient") <<"TrackEfficiencyClient:: Begining of Run";
}


//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::analyze(edm::Event const& e, edm::EventSetup const& eSetup)
//-----------------------------------------------------------------------------------
{
  edm::LogInfo( "TrackEfficiencyClient") << "TrackEfficiencyClient::analyze";
}

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::endJob() 
//-----------------------------------------------------------------------------------
{
} 

//-----------------------------------------------------------------------------------
void TrackEfficiencyClient::endRun() 
//-----------------------------------------------------------------------------------
{
}


//----------------------------------------------------------------------------------- 
void TrackEfficiencyClient::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup)
//-----------------------------------------------------------------------------------
{
 edm::LogInfo( "TrackEfficiencyClient") << "TrackEfficiencyClient::endLuminosityBlock"; 

 histName = "/trackX_";
 MonitorElement* trackX = dqmStore_->get(FolderName_+histName+algoName_);
 
 histName = "/muonX_";
 MonitorElement* muonX  = dqmStore_->get(FolderName_+histName+algoName_);
 
 histName = "/trackY_";
 MonitorElement* trackY = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonY_";
 MonitorElement* muonY  = dqmStore_->get(FolderName_+histName+algoName_);
 
 histName = "/trackZ_";
 MonitorElement* trackZ = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonZ_";
 MonitorElement* muonZ  = dqmStore_->get(FolderName_+histName+algoName_);
  
 histName = "/trackEta_";
 MonitorElement* trackEta = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonEta_";
 MonitorElement* muonEta  = dqmStore_->get(FolderName_+histName+algoName_);
 
 histName = "/trackPhi_";
 MonitorElement* trackPhi = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonPhi_";
 MonitorElement* muonPhi  = dqmStore_->get(FolderName_+histName+algoName_);
 
 histName = "/trackD0_";
 MonitorElement* trackD0 = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonD0_";
 MonitorElement* muonD0  = dqmStore_->get(FolderName_+histName+algoName_);

 histName = "/trackCompatibleLayers_";
 MonitorElement* trackCompatibleLayers = dqmStore_->get(FolderName_+histName+algoName_);
 histName = "/muonCompatibleLayers_";
 MonitorElement* muonCompatibleLayers  = dqmStore_->get(FolderName_+histName+algoName_);

 if(trackX && muonX && trackY && muonY && trackZ && muonZ && trackEta && muonEta && trackPhi && muonPhi && trackD0 && muonD0 && trackCompatibleLayers && muonCompatibleLayers){
   if (trackEfficiency_)
     { 
       if (effX  ->getTH1F() && trackX  ->getTH1F() && muonX  ->getTH1F()) { effX	-> getTH1F()->Divide(trackX->getTH1F(),muonX->getTH1F(),1.,1.,"");}
       if (effY  ->getTH1F() && trackY  ->getTH1F() && muonY  ->getTH1F()) { effY	-> getTH1F()->Divide(trackY->getTH1F(),muonY->getTH1F(),1.,1.,"");}
       if (effZ  ->getTH1F() && trackZ  ->getTH1F() && muonZ  ->getTH1F()) { effZ	-> getTH1F()->Divide(trackZ->getTH1F(),muonZ->getTH1F(),1.,1.,"");}
       if (effEta->getTH1F() && trackEta->getTH1F() && muonEta->getTH1F()) { effEta -> getTH1F()->Divide(trackEta->getTH1F(),muonEta->getTH1F(),1.,1.,"");}
       if (effPhi->getTH1F() && trackPhi->getTH1F() && muonPhi->getTH1F()) { effPhi -> getTH1F()->Divide(trackPhi->getTH1F(),muonPhi->getTH1F(),1.,1.,"");}
       if (effD0 ->getTH1F() && trackD0 ->getTH1F() && muonD0 ->getTH1F()) { effD0  -> getTH1F()->Divide(trackD0->getTH1F(),muonD0->getTH1F(),1.,1.,"");}
       if (effCompatibleLayers->getTH1F() && trackCompatibleLayers->getTH1F() && muonCompatibleLayers->getTH1F()) { effCompatibleLayers -> getTH1F()->Divide(trackCompatibleLayers->getTH1F(),muonCompatibleLayers->getTH1F(),1.,1.,"");}
 }
   else {
     if (effX  ->getTH1F() && trackX  ->getTH1F() && muonX  ->getTH1F()) { effX	-> getTH1F()->Divide(muonX->getTH1F(),trackX->getTH1F(),1.,1.,"");}
     if (effY  ->getTH1F() && trackY  ->getTH1F() && muonY  ->getTH1F()) { effY	-> getTH1F()->Divide(muonY->getTH1F(),trackY->getTH1F(),1.,1.,"");}
     if (effZ  ->getTH1F() && trackZ  ->getTH1F() && muonZ  ->getTH1F()) { effZ	-> getTH1F()->Divide(muonZ->getTH1F(),trackZ->getTH1F(),1.,1.,"");}
     if (effEta->getTH1F() && trackEta->getTH1F() && muonEta->getTH1F()) { effEta -> getTH1F()->Divide(muonEta->getTH1F(),trackEta->getTH1F(),1.,1.,"");}
     if (effPhi->getTH1F() && trackPhi->getTH1F() && muonPhi->getTH1F()) { effPhi -> getTH1F()->Divide(muonPhi->getTH1F(),trackPhi->getTH1F(),1.,1.,"");}
     if (effD0 ->getTH1F() && trackD0 ->getTH1F() && muonD0 ->getTH1F()) { effD0  -> getTH1F()->Divide(muonD0->getTH1F(),trackD0->getTH1F(),1.,1.,"");}
   }
 }
 
}

DEFINE_FWK_MODULE(TrackEfficiencyClient);
