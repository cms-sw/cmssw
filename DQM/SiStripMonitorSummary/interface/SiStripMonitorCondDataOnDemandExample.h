#ifndef SiStripMonitorSummary_SiStripMonitorCondDataOnDemandExample_h
#define SiStripMonitorSummary_SiStripMonitorCondDataOnDemandExample_h
// -*- C++ -*-
//
// Package:     SiStripMonitorSummary
// Class  :     SiStripMonitorCondDataOnDemandExample
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

class SiStripClassToMonitorCondData;

class SiStripMonitorCondDataOnDemandExample : public edm::EDAnalyzer {
 
 public:
 
   explicit SiStripMonitorCondDataOnDemandExample(const edm::ParameterSet&);
 
   ~SiStripMonitorCondDataOnDemandExample();
   
   virtual void beginJob() ;  
   virtual void beginRun(edm::Run const& run, edm::EventSetup const& eSetup);
   virtual void analyze(const edm::Event&, const edm::EventSetup&);
   virtual void endRun(edm::Run const& run, edm::EventSetup const& eSetup);
   virtual void endJob() ;
  
   
  
 private: 
  int eventCounter_; 
  edm::ParameterSet conf_;
  SiStripClassToMonitorCondData*    condDataMonitoring_ ;

};

#endif
