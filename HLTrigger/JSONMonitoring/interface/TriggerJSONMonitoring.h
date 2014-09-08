#ifndef HLTcore_TriggerJSONMonitoring_h
#define HLTcore_TriggerJSONMonitoring_h

/** \class TriggerJSONMonitoring
 *
 *  
 *  Description: This class prints JSON files with trigger info. 
 *
 *  Created:  Wed, 09 Jul 2014 
 *
 *  \author Aram Avetisyan
 *  \author Daniel Salerno
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//
// class declaration
//
class TriggerJSONMonitoring : public edm::EDAnalyzer {
  
 public:
  explicit TriggerJSONMonitoring(const edm::ParameterSet&);
  ~TriggerJSONMonitoring();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  void reset(bool changed = false);       // reset all counters

 private:
  boost::shared_ptr<jsoncollector::FastMonitor> jsonMonitor_;
  DataPointDefinition outJson_;
  IntJ processed_; 

  std::string baseRunDir;

  std::vector<std::string>  hltNames_;  // name of each HLT algorithm
  std::vector<IntJ> hltPaths_;          // stores name and # of event accepted by HLT paths

  std::string jsonDefinitionFile;       // JSON definition file name

  edm::InputTag triggerResults_;                               // Input tag for TriggerResults
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults
      
  HLTConfigProvider hltConfig_;         // to get configuration for HLT

};
#endif
