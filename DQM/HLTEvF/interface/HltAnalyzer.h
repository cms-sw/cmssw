// -*- C++ -*-
//
// Package:    HltAnalyzer
// Class:      HltAnalyzer
// 
/**\class HltAnalyzer HltAnalyzer.cc DQM/HLTEvF/interface/HltAnalyzer.h

   Description: Correlate timings and pass/fail for paths and modules 
                on paths.

   Implementation:
     Produces a HltPerformanceInfo object
*/
//
// Original Author:  Peter Wittich
//         Created:  Thu Nov  9 07:51:28 CST 2006
// $Id$
//
//

#ifndef HLTANALYZER_H
#define HLTANALYZER_H

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/Timing.h"

#include "DataFormats/Common/interface/ModuleDescription.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/TriggerNamesService.h"


#include "DataFormats/HLTReco/interface/HLTPerformanceInfo.h"


//
// class declaration
//


class HltAnalyzer : public edm::EDFilter {
public:
  explicit HltAnalyzer(const edm::ParameterSet&);
  ~HltAnalyzer();
  void newTimingMeasurement(const edm::ModuleDescription& iMod, 
			    double diffTime) ;
    
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  
  // ----------member data ---------------------------
  HLTPerformanceInfo perfInfo_;
  std::string myName_;
  bool verbose_;
  bool verbose() { return verbose_; }
  edm::InputTag trigResLabel_;
      
};

#endif // HLTANALYZER_H
