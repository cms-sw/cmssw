// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripClassToMonitorCondData
// 
// Original Author:  Evelyne Delmeire
//


#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"

#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"
#include "DQM/SiStripCommon/interface/ExtractTObject.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


#include "DQM/SiStripMonitorSummary/interface/SiStripClassToMonitorCondData.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripLorentzAngleDQM.h" 

#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"

#include "CondFormats/DataRecord/interface/SiStripPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/SiStripApvGainRcd.h"
#include "CondFormats/DataRecord/interface/SiStripNoisesRcd.h"

#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"

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
SiStripClassToMonitorCondData::SiStripClassToMonitorCondData(edm::ParameterSet const& iConfig):conf_(iConfig){
  
  monitorPedestals_      = iConfig.getParameter<bool>("MonitorSiStripPedestal");
  monitorNoises_         = iConfig.getParameter<bool>("MonitorSiStripNoise");
  monitorQuality_        = iConfig.getParameter<bool>("MonitorSiStripQuality");
  monitorApvGains_       = iConfig.getParameter<bool>("MonitorSiStripApvGain");
  monitorLorentzAngle_   = iConfig.getParameter<bool>("MonitorSiStripLorentzAngle");
  
}
// -----



//
// ----- Destructor
// 
SiStripClassToMonitorCondData::~SiStripClassToMonitorCondData(){
  
  if(monitorPedestals_)  { delete pedestalsDQM_;}
  if(monitorNoises_)     { delete noisesDQM_;   }
  if(monitorQuality_)    { delete qualityDQM_;  }
  if(monitorApvGains_)   { delete apvgainsDQM_; }
  if(monitorLorentzAngle_){ delete lorentzangleDQM_; }

}
// -----




//
// ----- beginRun
//    
void SiStripClassToMonitorCondData::beginRun(edm::EventSetup const& eSetup) {
  
  if(monitorPedestals_){
    pedestalsDQM_ = new SiStripPedestalsDQM(eSetup,
                                            conf_.getParameter<edm::ParameterSet>("SiStripPedestalsDQM_PSet"),
                                            conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  
  
  if(monitorNoises_){
    noisesDQM_ = new SiStripNoisesDQM(eSetup,
                                      conf_.getParameter<edm::ParameterSet>("SiStripNoisesDQM_PSet"),
                                      conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
 
  
  if(monitorQuality_){
    qualityDQM_ = new SiStripQualityDQM(eSetup,
                                        conf_.getParameter<edm::ParameterSet>("SiStripQualityDQM_PSet"),
                                        conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  } 
  
 
  if(monitorApvGains_){
    apvgainsDQM_ = new SiStripApvGainsDQM(eSetup,
                                          conf_.getParameter<edm::ParameterSet>("SiStripApvGainsDQM_PSet"),
                                          conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  
  if(monitorLorentzAngle_){
    lorentzangleDQM_ = new SiStripLorentzAngleDQM(eSetup,
                                          conf_.getParameter<edm::ParameterSet>("SiStripLorentzAngleDQM_PSet"),
                                          conf_.getParameter<edm::ParameterSet>("FillConditions_PSet"));
  }
  

} // beginRun
// -----



//
// ----- beginJob
//
void SiStripClassToMonitorCondData::beginJob(void){} //beginJob


//
// ----- getModuleMEsOnDemand
//
void SiStripClassToMonitorCondData::getModMEsOnDemand(edm::EventSetup const& eSetup, uint32_t requestedDetId){

  if(monitorPedestals_)      { pedestalsDQM_     ->analysisOnDemand(eSetup,requestedDetId);}
  if(monitorNoises_)         { noisesDQM_        ->analysisOnDemand(eSetup,requestedDetId);}    
  if(monitorQuality_)        { qualityDQM_       ->analysisOnDemand(eSetup,requestedDetId);}
  if(monitorApvGains_)       { apvgainsDQM_      ->analysisOnDemand(eSetup,requestedDetId);} 
  if(monitorLorentzAngle_)   { lorentzangleDQM_  ->analysisOnDemand(eSetup,requestedDetId);} 
    
}
// -----

//
// ----- getlayerMEsOnDemand
//
void SiStripClassToMonitorCondData::getLayerMEsOnDemand(edm::EventSetup const& eSetup, std::string requestedSubDetector, 
                                                        uint32_t requestedSide, 
							uint32_t requestedLayer){
 
  if(monitorPedestals_)      { pedestalsDQM_     ->analysisOnDemand(eSetup,requestedSubDetector, requestedSide,requestedLayer);}
  if(monitorNoises_)         { noisesDQM_        ->analysisOnDemand(eSetup,requestedSubDetector, requestedSide,requestedLayer);}    
  if(monitorQuality_)        { qualityDQM_       ->analysisOnDemand(eSetup,requestedSubDetector, requestedSide,requestedLayer);}
  if(monitorApvGains_)       { apvgainsDQM_      ->analysisOnDemand(eSetup,requestedSubDetector, requestedSide,requestedLayer);} 
  if(monitorLorentzAngle_)   { lorentzangleDQM_  ->analysisOnDemand(eSetup,requestedSubDetector, requestedSide,requestedLayer);} 
  
}

//
// ----- Analyze
//
void SiStripClassToMonitorCondData::analyseCondData(edm::EventSetup const& eSetup){

  if(monitorPedestals_)      { pedestalsDQM_     ->analysis(eSetup);}
  if(monitorNoises_)         { noisesDQM_        ->analysis(eSetup);}    
  if(monitorQuality_)        { qualityDQM_       ->analysis(eSetup);}
  if(monitorApvGains_)       { apvgainsDQM_      ->analysis(eSetup);} 
  if(monitorLorentzAngle_)   { lorentzangleDQM_  ->analysis(eSetup);} 
    
} // analyze
// -----



//
// ----- endRun
//    
void SiStripClassToMonitorCondData::endRun(edm::EventSetup const& eSetup) {
  
  bool outputMEsInRootFile    = conf_.getParameter<bool>("OutputMEsInRootFile");
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
void SiStripClassToMonitorCondData::endJob(void){} //endJob


  
