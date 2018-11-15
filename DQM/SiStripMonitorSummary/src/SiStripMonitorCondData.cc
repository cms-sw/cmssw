// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripMonitorCondData
// 
// Original Author:  Evelyne Delmeire
//


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "DQM/SiStripMonitorSummary/interface/SiStripMonitorCondData.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripThresholdDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripBackPlaneCorrectionDQM.h"
#include "DQM/SiStripMonitorSummary/interface/SiStripCablingDQM.h"

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"

// std
#include <cstdlib>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>


//
// ----- Constructor
//
SiStripMonitorCondData::SiStripMonitorCondData(edm::ParameterSet const& iConfig):conf_(iConfig){
  
  monitorPedestals_      = iConfig.getParameter<bool>("MonitorSiStripPedestal");
  monitorNoises_         = iConfig.getParameter<bool>("MonitorSiStripNoise");
  monitorLowThreshold_   = iConfig.getParameter<bool>("MonitorSiStripLowThreshold");
  monitorHighThreshold_  = iConfig.getParameter<bool>("MonitorSiStripHighThreshold");
  monitorQuality_        = iConfig.getParameter<bool>("MonitorSiStripQuality");
  monitorApvGains_       = iConfig.getParameter<bool>("MonitorSiStripApvGain");
  monitorLorentzAngle_   = iConfig.getParameter<bool>("MonitorSiStripLorentzAngle");
  monitorBackPlaneCorrection_   = iConfig.getParameter<bool>("MonitorSiStripBackPlaneCorrection");
  monitorCabling_        = iConfig.getParameter<bool>("MonitorSiStripCabling");
  
}
// -----



//
// ----- Destructor
// 
SiStripMonitorCondData::~SiStripMonitorCondData(){
}
// -----



//
// ----- beginRun
//    
void SiStripMonitorCondData::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {

  if(monitorPedestals_){
    pedestalsDQM_ = std::make_unique<SiStripPedestalsDQM>(eSetup,
							  run.run(),
							  conf_.getParameter<edm::ParameterSet>("SiStripPedestalsDQM_PSet"),
							  conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  
  
  if(monitorNoises_){
    noisesDQM_ = std::make_unique<SiStripNoisesDQM>(eSetup,
						    run.run(),
						    conf_.getParameter<edm::ParameterSet>("SiStripNoisesDQM_PSet"),
						    conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  
  
  if(monitorLowThreshold_){
    lowthresholdDQM_ = std::make_unique<SiStripThresholdDQM>(eSetup,
							     run.run(),
							     conf_.getParameter<edm::ParameterSet>("SiStripLowThresholdDQM_PSet"),
							     conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }

  if(monitorHighThreshold_){
    highthresholdDQM_ = std::make_unique<SiStripThresholdDQM>(eSetup,
							      run.run(),
							      conf_.getParameter<edm::ParameterSet>("SiStripHighThresholdDQM_PSet"),
							      conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }

  if(monitorQuality_){
    qualityDQM_ = std::make_unique<SiStripQualityDQM>(eSetup,
						      run.run(),
						      conf_.getParameter<edm::ParameterSet>("SiStripQualityDQM_PSet"),
						      conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  } 
  
 
  if(monitorApvGains_){
    apvgainsDQM_ = std::make_unique<SiStripApvGainsDQM>(eSetup,
							run.run(),
							conf_.getParameter<edm::ParameterSet>("SiStripApvGainsDQM_PSet"),
							conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  
  
  if(monitorLorentzAngle_){
    lorentzangleDQM_ = std::make_unique<SiStripLorentzAngleDQM>(eSetup,
								run.run(),
								conf_.getParameter<edm::ParameterSet>("SiStripLorentzAngleDQM_PSet"),
								conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }

  if(monitorBackPlaneCorrection_){
    bpcorrectionDQM_ = std::make_unique<SiStripBackPlaneCorrectionDQM>(eSetup,
								       run.run(),
								       conf_.getParameter<edm::ParameterSet>("SiStripBackPlaneCorrectionDQM_PSet"),
								       conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  if(monitorCabling_){
    cablingDQM_ = std::make_unique<SiStripCablingDQM>(eSetup,
						      run.run(),
						      conf_.getParameter<edm::ParameterSet>("SiStripCablingDQM_PSet"),
						      conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
} // beginRun
// -----



//
// ----- beginJob
//
void SiStripMonitorCondData::beginJob(void){} //beginJob
// -----



//
// ----- Analyze
//
void SiStripMonitorCondData::analyze(edm::Event const& iEvent, edm::EventSetup const& eSetup){
 
  if(monitorPedestals_)      { pedestalsDQM_     ->analysis(eSetup);}
  if(monitorNoises_)         { noisesDQM_        ->analysis(eSetup);}    
  if(monitorLowThreshold_)   { lowthresholdDQM_  ->analysis(eSetup);}    
  if(monitorHighThreshold_)  { highthresholdDQM_ ->analysis(eSetup);}    
  if(monitorApvGains_)       { apvgainsDQM_      ->analysis(eSetup);}    
  if(monitorLorentzAngle_)   { lorentzangleDQM_  ->analysis(eSetup);}
  if(monitorBackPlaneCorrection_)   { bpcorrectionDQM_  ->analysis(eSetup);}
  if(monitorQuality_)        { qualityDQM_->analysis(eSetup);qualityDQM_->fillGrandSummaryMEs(eSetup);}//fillGrand. for SiStripquality
  if(monitorCabling_)        { cablingDQM_       ->analysis(eSetup);}
} // analyze
// -----



//
// ----- endRun
//    
void SiStripMonitorCondData::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
  if(monitorPedestals_)      { pedestalsDQM_     ->end();}
  if(monitorNoises_)         { noisesDQM_        ->end();}    
  if(monitorLowThreshold_)   { lowthresholdDQM_  ->end();}    
  if(monitorHighThreshold_)  { highthresholdDQM_ ->end();}    
  if(monitorApvGains_)       { apvgainsDQM_      ->end();}    
  if(monitorLorentzAngle_)   { lorentzangleDQM_  ->end();}
  if(monitorBackPlaneCorrection_)   { bpcorrectionDQM_  ->end();}
  if(monitorQuality_)        { qualityDQM_       ->end();}
  if(monitorCabling_)        { cablingDQM_       ->end();}
 
  bool outputMEsInRootFile     = conf_.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName  = conf_.getParameter<std::string>("OutputFileName");

  DQMStore* dqmStore_=edm::Service<DQMStore>().operator->();
  
  if (outputMEsInRootFile) { 
    dqmStore_->showDirStructure();
    dqmStore_->save(outputFileName);
  } 

  
} // endRun
// -----


//
// ----- endJob
//
void SiStripMonitorCondData::endJob(void){} 
//endJob
// -----


#include "FWCore/Framework/interface/MakerMacros.h"
  DEFINE_FWK_MODULE(SiStripMonitorCondData);

  
