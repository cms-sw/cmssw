#ifndef SiStripMonitorSummary_SiStripClassToMonitorCondData_h
#define SiStripMonitorSummary_SiStripClassToMonitorCondData_h
// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripClassToMonitorCondData
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
class SiStripQualityDQM;
class SiStripApvGainsDQM;
class SiStripLorentzAngleDQM; 
class SiStripBackPlaneCorrectionDQM; 
class SiStripCablingDQM;
class SiStripThresholdDQM;

class SiStripClassToMonitorCondData{
 
 public:
 
   explicit SiStripClassToMonitorCondData(edm::ParameterSet const& iConfig);
 
   ~SiStripClassToMonitorCondData();
   
   void beginJob() ;  
   void beginRun(edm::EventSetup const& eSetup);
   void analyseCondData(const edm::EventSetup&);
   void endRun(edm::EventSetup const& eSetup);
   void endJob() ;
  
   void getModMEsOnDemand(edm::EventSetup const& eSetup, uint32_t requestedDetId);
   void getLayerMEsOnDemand(edm::EventSetup const& eSetup, std::string requestedSubDetector, uint32_t requestedSide, uint32_t requestedLayer);
  
 private:  
  
   
   edm::ParameterSet conf_;
  
   bool monitorPedestals_     ;
   bool monitorNoises_        ;
   bool monitorQuality_       ;
   bool monitorApvGains_      ;
   bool monitorLorentzAngle_  ;
   bool monitorBackPlaneCorrection_  ;
   bool monitorLowThreshold_  ;
   bool monitorHighThreshold_ ;
   bool monitorCabling_       ;

   bool gainRenormalisation_;
     
   std::string outPutFileName;

   SiStripPedestalsDQM*           pedestalsDQM_;
   SiStripNoisesDQM*                 noisesDQM_; 
   SiStripQualityDQM*               qualityDQM_; 
   SiStripApvGainsDQM*             apvgainsDQM_; 
   SiStripLorentzAngleDQM*     lorentzangleDQM_;    
   SiStripBackPlaneCorrectionDQM*     bpcorrectionDQM_;    
   SiStripCablingDQM*               cablingDQM_;
   SiStripThresholdDQM*        lowthresholdDQM_;
   SiStripThresholdDQM*       highthresholdDQM_;

  
};

#endif
