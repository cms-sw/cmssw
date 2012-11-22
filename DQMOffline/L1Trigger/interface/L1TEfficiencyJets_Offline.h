#ifndef DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYJETS_OFFLINE_H
#define DQMOFFLINE_L1TRIGGER_L1TEFFICIENCYJETS_OFFLINE_H

/*
 * \file L1TEfficiencyJets.h
 *
 * $Date: 2012/11/15 17:50:03 $
 * $Revision: 1.1 $
 * \author J. Pela
 *
 */

// system include files
#include <memory>
#include <unistd.h>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <TString.h>

#include <iostream>
#include <fstream>
#include <vector>

//
// class declaration
//

class L1TEfficiencyJets_Offline : public edm::EDAnalyzer {
  
public:
  
  enum Errors{
    UNKNOWN                = 1,
    WARNING_PY_MISSING_FIT = 2
  };
  
public:
  
  L1TEfficiencyJets_Offline(const edm::ParameterSet& ps);   // Constructor
  virtual ~L1TEfficiencyJets_Offline();                     // Destructor
  
protected:
  
  // Event
  void analyze (const edm::Event& e, const edm::EventSetup& c); 
  
  // Job
  void beginJob();  
  void endJob  ();
  
  // Run
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup);
  void endRun  (const edm::Run& run, const edm::EventSetup& iSetup);
  
  // Luminosity Block
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  virtual void endLuminosityBlock  (edm::LuminosityBlock const& lumiBlock, edm::EventSetup const& c);
  
private:
  
  // bool
  bool  m_verbose;
  
  DQMStore* dbe;  // The DQM Service Handle
  
};

#endif