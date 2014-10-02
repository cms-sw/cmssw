#ifndef JSONMonitoring_TriggerJSONMonitoring_h
#define JSONMonitoring_TriggerJSONMonitoring_h

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
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
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

namespace hltJson {
  //Struct for storing variables that must be written and reset every lumi section
  struct lumiVars {
    HistoJ<unsigned int> *processed; // # of events processed

    HistoJ<unsigned int> *hltWasRun; // # of events where HLT[i] was run
    HistoJ<unsigned int> *hltL1s;    // # of events after L1 seed
    HistoJ<unsigned int> *hltPre;    // # of events after HLT prescale
    HistoJ<unsigned int> *hltAccept; // # of events accepted by HLT[i]
    HistoJ<unsigned int> *hltReject; // # of events rejected by HLT[i]
    HistoJ<unsigned int> *hltErrors; // # of events with error in HLT[i]

    HistoJ<unsigned int> *hltDatasets; // # of events accepted by each dataset

    //Names and directories aren't changed at lumi section boundaries, 
    //but they need to be in this struct to be write JSON files at the end
    HistoJ<std::string> *hltNames;     // list of HLT path names
    HistoJ<std::string> *datasetNames; // list of dataset names

    std::string baseRunDir;        //Base directory from EvFDaqDirector
    std::string jsonRateDefFile;   //Definition file name for JSON with rates
    std::string jsonLegendDefFile; //Definition file name for JSON with legend of names
  };
}//End hltJson namespace

//
// class declaration
//
class TriggerJSONMonitoring : public edm::stream::EDAnalyzer <edm::LuminosityBlockSummaryCache<hltJson::lumiVars>>
{
 public:
  explicit TriggerJSONMonitoring(const edm::ParameterSet&);
  ~TriggerJSONMonitoring();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void analyze(edm::Event const&, 
	       edm::EventSetup const&);

  void beginRun(edm::Run const&, 
		edm::EventSetup const&);

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

  void writeDefJson(std::string path);
  void writeDefLegJson(std::string path);
  
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

  std::vector<int> posL1s_;             // pos # of last L1 seed
  std::vector<int> posPre_;             // pos # of last HLT prescale
  
  std::string jsonRateDefFile_;   //Definition file name for JSON with rates
  std::string jsonLegendDefFile_; //Definition file name for JSON with legend of names
  
  //Variables that need to be reset at lumi section boundaries
  unsigned int              processed_; // # of events processed

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
