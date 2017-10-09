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
 
   ~SiStripMonitorCondData();
   
   virtual void beginJob() ;  
   virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
   virtual void endJob() ;
  
  
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

   SiStripPedestalsDQM*           pedestalsDQM_;
   SiStripNoisesDQM*                 noisesDQM_; 
   SiStripThresholdDQM*        lowthresholdDQM_; 
   SiStripThresholdDQM*       highthresholdDQM_; 
   SiStripQualityDQM*               qualityDQM_; 
   SiStripApvGainsDQM*             apvgainsDQM_;  
   SiStripLorentzAngleDQM*     lorentzangleDQM_; 
   SiStripBackPlaneCorrectionDQM*     bpcorrectionDQM_; 
   SiStripCablingDQM*               cablingDQM_;  
  
};

#endif
