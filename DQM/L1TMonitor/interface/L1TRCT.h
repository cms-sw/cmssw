#ifndef L1TRCT_H
#define L1TRCT_H

// system include files
#include <memory>
#include <unistd.h>


#include <iostream>
#include <fstream>
#include <vector>


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"


// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

//
// class declaration
//

class L1TRCT : public edm::EDAnalyzer {

public:

// Constructor
  L1TRCT(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TRCT();

protected:
// Analyze
 void analyze(const edm::Event& e, const edm::EventSetup& c);

// BeginRun
  void beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup);

// BeginJob
 void beginJob(void);

// EndJob
void endJob(void);

private:
  // ----------member data ---------------------------
  DQMStore * dbe;

  // trigger type information
  MonitorElement *triggerType_;

  // region global coordinates
  MonitorElement* rctRegionsEtEtaPhi_;
  MonitorElement* rctRegionsOccEtaPhi_;

  // region local coordinates
  MonitorElement* rctRegionsLocalEtEtaPhi_;
  MonitorElement* rctRegionsLocalOccEtaPhi_;
  MonitorElement* rctTauVetoLocalEtaPhi_;

  // Region rank
  MonitorElement* rctRegionRank_;


  MonitorElement* rctOverFlowEtaPhi_;
  MonitorElement* rctTauVetoEtaPhi_;
  MonitorElement* rctMipEtaPhi_;
  MonitorElement* rctQuietEtaPhi_;
  MonitorElement* rctHfPlusTauEtaPhi_;

  // Bx
  MonitorElement *rctRegionBx_;
  MonitorElement *rctEmBx_;

  // em
  // HW coordinates
  MonitorElement *rctEmCardRegion_;


  MonitorElement* rctIsoEmEtEtaPhi_;
  MonitorElement* rctIsoEmOccEtaPhi_;
  MonitorElement* rctNonIsoEmEtEtaPhi_;
  MonitorElement* rctNonIsoEmOccEtaPhi_;
  MonitorElement* rctIsoEmRank_;
  MonitorElement* rctNonIsoEmRank_;


  int nev_; // Number of events processed
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  
  edm::EDGetTokenT<L1CaloRegionCollection> rctSource_L1CRCollection_;
  edm::EDGetTokenT<L1CaloEmCollection> rctSource_L1CEMCollection_;
  
  /// filter TriggerType
  int filterTriggerType_;

};

#endif
