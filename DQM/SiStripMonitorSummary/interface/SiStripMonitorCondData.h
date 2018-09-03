#ifndef SiStripMonitorSummary_SiStripMonitorSummary_h
#define SiStripMonitorSummary_SiStripMonitorSummary_h

// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripMonitorCondData
// 
// Original Author:  Evelyne Delmeire
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "boost/cstdint.hpp"
#include <iostream>
#include <string>
#include <vector>

class MonitorElement;

class SiStripPedestalsDQM;
class SiStripNoisesDQM;
class SiStripThresholdDQM;
class SiStripQualityDQM;
class SiStripApvGainsDQM;
class SiStripLorentzAngleDQM;
class SiStripBackPlaneCorrectionDQM;
class SiStripCablingDQM;

class SiStripMonitorCondData : public edm::EDAnalyzer {
 
 public:
 
   explicit SiStripMonitorCondData(const edm::ParameterSet&);
 
   ~SiStripMonitorCondData() override;
   
   void beginJob() override ;  
   void beginRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
   void analyze(const edm::Event&, const edm::EventSetup&) override;
   void endRun(edm::Run const& run, edm::EventSetup const& eSetup) override;
   void endJob() override ;
  
  
 private:  
  
   edm::ParameterSet conf_;
  
   bool monitorPedestals_   ;
   bool monitorNoises_      ;
   bool monitorLowThreshold_  ;
   bool monitorHighThreshold_ ;
   bool monitorQuality_     ;
   bool monitorApvGains_    ;
   bool monitorLorentzAngle_;
   bool monitorBackPlaneCorrection_;
   bool monitorCabling_;
     
   std::string outPutFileName;

   std::unique_ptr<SiStripPedestalsDQM>                  pedestalsDQM_;
   std::unique_ptr<SiStripNoisesDQM>                        noisesDQM_; 
   std::unique_ptr<SiStripThresholdDQM>               lowthresholdDQM_; 
   std::unique_ptr<SiStripThresholdDQM>              highthresholdDQM_; 
   std::unique_ptr<SiStripQualityDQM>                      qualityDQM_; 
   std::unique_ptr<SiStripApvGainsDQM>                    apvgainsDQM_;  
   std::unique_ptr<SiStripLorentzAngleDQM>            lorentzangleDQM_; 
   std::unique_ptr<SiStripBackPlaneCorrectionDQM>     bpcorrectionDQM_; 
   std::unique_ptr<SiStripCablingDQM>                      cablingDQM_;  
  
};

#endif
