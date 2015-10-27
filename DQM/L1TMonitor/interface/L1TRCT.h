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

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


// GCT and RCT data formats
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"

//
// class declaration
//

class L1TRCT : public DQMEDAnalyzer {

public:

// Constructor
  L1TRCT(const edm::ParameterSet& ps);

// Destructor
 virtual ~L1TRCT();

protected:
// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  virtual void dqmBeginRun(const edm::Run&, const edm::EventSetup&);
  virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &ibooker, edm::Run const&, edm::EventSetup const&) override ;
 
private:
  // ----------member data ---------------------------

  // trigger type information
  MonitorElement *triggerType_;

  // RCT
  // regions
  MonitorElement* rctRegionsEtEtaPhi_;
  MonitorElement* rctRegionsOccEtaPhi_;
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
  MonitorElement* rctIsoEmEtEtaPhi_;
  MonitorElement* rctIsoEmOccEtaPhi_;
  MonitorElement* rctNonIsoEmEtEtaPhi_;
  MonitorElement* rctNonIsoEmOccEtaPhi_;
  MonitorElement* rctIsoEmRank_;
  MonitorElement* rctNonIsoEmRank_;

  MonitorElement* rctNotCentralRegionsEtEtaPhi_;
  MonitorElement* rctNotCentralRegionsOccEtaPhi_;
  MonitorElement* rctNotCentralIsoEmEtEtaPhi_;
  MonitorElement* rctNotCentralIsoEmOccEtaPhi_;
  MonitorElement* rctNotCentralNonIsoEmEtEtaPhi_;
  MonitorElement* rctNotCentralNonIsoEmOccEtaPhi_;


  // Layer2
  // regions
  MonitorElement* layer2RegionsEtEtaPhi_;
  MonitorElement* layer2RegionsOccEtaPhi_;
  MonitorElement* layer2RegionRank_;
  MonitorElement* layer2OverFlowEtaPhi_;
  MonitorElement* layer2TauVetoEtaPhi_;
  MonitorElement* layer2MipEtaPhi_;
  MonitorElement* layer2QuietEtaPhi_;
  MonitorElement* layer2HfPlusTauEtaPhi_;

  // Bx
  MonitorElement *layer2RegionBx_;
  MonitorElement *layer2EmBx_;

  // em
  MonitorElement* layer2IsoEmEtEtaPhi_;
  MonitorElement* layer2IsoEmOccEtaPhi_;
  MonitorElement* layer2NonIsoEmEtEtaPhi_;
  MonitorElement* layer2NonIsoEmOccEtaPhi_;
  MonitorElement* layer2IsoEmRank_;
  MonitorElement* layer2NonIsoEmRank_;

  // run/lumi
  MonitorElement* runId_;
  MonitorElement* lumisecId_;


  int nev_; // Number of events processed
  std::string histFolder_;
  std::string outputFile_; //file name for ROOT ouput
  bool verbose_;
  bool monitorDaemon_;
  std::ofstream logFile_;
  
  edm::EDGetTokenT<L1CaloRegionCollection> rctSource_L1CRCollection_;
  edm::EDGetTokenT<L1CaloEmCollection> rctSource_L1CEMCollection_;
  edm::EDGetTokenT<L1CaloRegionCollection> rctSource_GCT_L1CRCollection_;
  edm::EDGetTokenT<L1CaloEmCollection> rctSource_GCT_L1CEMCollection_;
  
  /// filter TriggerType
  int filterTriggerType_;
  int selectBX_;
};

#endif
