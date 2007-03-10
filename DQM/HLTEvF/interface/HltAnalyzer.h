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
// $Id: HltAnalyzer.h,v 1.3 2006/12/20 17:32:35 wittich Exp $
//
//

#ifndef HLTANALYZER_H
#define HLTANALYZER_H

// system include files
#include <memory>
#include <vector>
#include <string>
#include <map>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Services/interface/Timing.h"

#include "DataFormats/Provenance/interface/ModuleDescription.h"

#include "DataFormats/Common/interface/TriggerResults.h"

#include "FWCore/Framework/interface/TriggerNamesService.h"



//
// class declaration
//

class TFile;
class TH1D;


class HltAnalyzer : public edm::EDFilter {
public:
  explicit HltAnalyzer(const edm::ParameterSet&);
  ~HltAnalyzer();
    
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  
  // ----------member data ---------------------------
  std::string myName_;
  std::string name() { return myName_; }
  bool verbose_;
  bool verbose() { return verbose_; }


  edm::InputTag hltPerfLabel_;
  // --------------------------------------
  // count the occurance of a string
  typedef std::map<std::string, int> ModuleCount_t;
  typedef std::map<std::string, ModuleCount_t> PathModuleCount_t;
  ModuleCount_t slowestModule_;
  PathModuleCount_t rejectionModule_;

  TFile *f_;
  TH1D *s1_;
  TH1D *s2_;

};

#endif // HLTANALYZER_H
