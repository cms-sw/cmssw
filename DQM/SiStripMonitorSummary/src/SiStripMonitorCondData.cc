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


#include "DQM/SiStripMonitorSummary/interface/SiStripMonitorCondData.h"

#include "DQM/SiStripMonitorSummary/interface/SiStripPedestalsDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripNoisesDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripQualityDQM.h" 
#include "DQM/SiStripMonitorSummary/interface/SiStripApvGainsDQM.h" 


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
  monitorQuality_        = iConfig.getParameter<bool>("MonitorSiStripQuality");
  monitorApvGains_       = iConfig.getParameter<bool>("MonitorSiStripApvGain");
 
}
// -----



//
// ----- Destructor
// 
SiStripMonitorCondData::~SiStripMonitorCondData(){

  if(monitorPedestals_)  { delete pedestalsDQM_;}
  if(monitorNoises_)     { delete noisesDQM_;   }
  if(monitorQuality_)    { delete qualityDQM_;  }
  if(monitorApvGains_)   { delete apvgainsDQM_; }

}
// -----




//
// ----- beginRun
//    
void SiStripMonitorCondData::beginRun(edm::Run const& run, edm::EventSetup const& eSetup) {

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
  
} // beginRun
// -----



//
// ----- beginJob
//
void SiStripMonitorCondData::beginJob(void){

  edm::LogInfo("SiStripMonitorCondData") << "[SiStripMonitorCondData::beginJob] Starting";        

} //beginJob



//
// ----- Analyze
//
void SiStripMonitorCondData::analyze(edm::Event const& iEvent, edm::EventSetup const& eSetup){
 
  if(monitorPedestals_)      { pedestalsDQM_     ->analysis(eSetup);}
  if(monitorNoises_)         { noisesDQM_        ->analysis(eSetup);}    
  if(monitorQuality_)        { qualityDQM_       ->analysis(eSetup);}
  if(monitorApvGains_)       { apvgainsDQM_      ->analysis(eSetup);}    
 
} // analyze
// -----



//
// ----- endRun
//    
void SiStripMonitorCondData::endRun(edm::Run const& run, edm::EventSetup const& eSetup) {
 
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
void SiStripMonitorCondData::endJob(void){

    edm::LogInfo("SiStripMonitorCondData") << "[SiStripMonitorCondData::EndJob] Finished";        

} //endJob


#include "FWCore/Framework/interface/MakerMacros.h"
  DEFINE_FWK_MODULE(SiStripMonitorCondData);

  
