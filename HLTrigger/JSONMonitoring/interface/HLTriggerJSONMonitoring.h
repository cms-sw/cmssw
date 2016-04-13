#ifndef JSONMonitoring_HLTriggerJSONMonitoring_h
#define JSONMonitoring_HLTriggerJSONMonitoring_h

/** \class HLTriggerJSONMonitoring         
 *     
 *  
 *  Description: This class prints JSON files with HLT info.
 *          
 *  Created:  Fri, 11 Mar 2016     
 *       
 *  \author Aram Avetisyan   
 * 
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "FWCore/Framework/interface/ESHandle.h"          

#include <atomic>

namespace hltJson {
  //Struct for storing variables that must be written and reset every lumi section 
  struct lumiVars {
    jsoncollector::HistoJ<unsigned int> *processed; // # of events processed
    jsoncollector::HistoJ<unsigned int> *hltWasRun; // # of events where HLT[i] was run 
    jsoncollector::HistoJ<unsigned int> *hltL1s;    // # of events after L1 seed
    jsoncollector::HistoJ<unsigned int> *hltPre;    // # of events after HLT prescale 
    jsoncollector::HistoJ<unsigned int> *hltAccept; // # of events accepted by HLT[i]
    jsoncollector::HistoJ<unsigned int> *hltReject; // # of events rejected by HLT[i] 
    jsoncollector::HistoJ<unsigned int> *hltErrors; // # of events with error in HLT[i]
    jsoncollector::HistoJ<unsigned int> *hltDatasets; // # of events accepted by each dataset 

    std::string baseRunDir; //Base directory from EvFDaqDirector 
    std::string stHltJsd;   //Definition file name for JSON with rates  

    std::string streamHLTDestination;
  };
  //End lumi struct
  //Struct for storing variable written once per run
  struct runVars{
    mutable std::atomic<bool> wroteFiles;
    mutable std::string streamHLTDestination;
  };
}//End hltJson namespace   

// 
// class declaration
// 
class HLTriggerJSONMonitoring : public edm::stream::EDAnalyzer <edm::RunCache<hltJson::runVars>, edm::LuminosityBlockSummaryCache<hltJson::lumiVars>>
{
 public:
  explicit HLTriggerJSONMonitoring(const edm::ParameterSet&);
  ~HLTriggerJSONMonitoring();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void analyze(edm::Event const&,
               edm::EventSetup const&);

  void beginRun(edm::Run const&,
                edm::EventSetup const&);

  static std::shared_ptr<hltJson::runVars> globalBeginRun(edm::Run const&, edm::EventSetup const&, void const*){
    std::shared_ptr<hltJson::runVars> rv(new hltJson::runVars);
    if (edm::Service<evf::EvFDaqDirector>().isAvailable()) {
      rv->streamHLTDestination = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations("streamHLTRates");
    }
    rv->wroteFiles = false;
    return rv;
  }
  
  static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext){ } 

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  static std::shared_ptr<hltJson::lumiVars> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                              edm::EventSetup const&,
                                                                              LuminosityBlockContext const*);

  void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                 edm::EventSetup const&,
                                 hltJson::lumiVars*) const;


  static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const*,
                                              hltJson::lumiVars*);

  void resetRun(bool changed);   //Reset run-related info
  void resetLumi();              //Reset all counters 

  void writeHLTDefJson(std::string path);

  //Variables from cfg and associated tokens 
  edm::InputTag triggerResults_;                               // Input tag for TriggerResults 
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults

  //Variables that change at most once per run 
  HLTConfigProvider hltConfig_;         // to get configuration for HLT

  std::string baseRunDir_; //Base directory from EvFDaqDirector

  // hltIndex_[ds][p] stores the hltNames_ index of the p-th path of the ds-th dataset 
  std::vector<std::vector<unsigned int> > hltIndex_;
  std::vector<std::string> hltNames_;                      // list of HLT path names 
  std::vector<std::string> datasetNames_;                  // list of dataset names 
  std::vector<std::vector<std::string> > datasetContents_; // list of path names for each dataset 

  std::vector<int> posL1s_;               // pos # of last L1 seed
  std::vector<int> posPre_;               // pos # of last HLT prescale

  std::string stHltJsd_;                  //Definition file name for JSON with rates

  //Variables that need to be reset at lumi section boundaries 
  unsigned int              processed_;      // # of events processed 

  std::vector<unsigned int> hltWasRun_; // # of events where HLT[i] was run   
  std::vector<unsigned int> hltL1s_;    // # of events after L1 seed
  std::vector<unsigned int> hltPre_;    // # of events after HLT prescale 
  std::vector<unsigned int> hltAccept_; // # of events accepted by HLT[i]
  std::vector<unsigned int> hltReject_; // # of events rejected by HLT[i]
  std::vector<unsigned int> hltErrors_; // # of events with error in HLT[i]

  std::vector<unsigned int> hltDatasets_; // # of events accepted by each dataset 

 private:

};
#endif
