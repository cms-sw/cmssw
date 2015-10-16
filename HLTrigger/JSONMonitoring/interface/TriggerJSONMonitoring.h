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
#include "FWCore/Utilities/interface/Adler32Calculator.h"

#include "EventFilter/Utilities/interface/JsonMonitorable.h"
#include "EventFilter/Utilities/interface/FastMonitor.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"
#include "EventFilter/Utilities/interface/FastMonitoringService.h"
#include "EventFilter/Utilities/interface/EvFDaqDirector.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleProvider.h"

#include "FWCore/Framework/interface/ESHandle.h"          
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"  
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"  
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"  
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"

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

    unsigned int prescaleIndex; // Prescale index for each lumi section

    std::string baseRunDir; //Base directory from EvFDaqDirector 
    std::string stHltJsd;   //Definition file name for JSON with rates  

    jsoncollector::HistoJ<unsigned int> *L1AlgoAccept;            // # of events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1TechAccept;            // # of events accepted by L1 Technical Triggers[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptPhysics;     // # of Physics events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1TechAcceptPhysics;     // # of Physics events accepted by L1 Technical Triggers[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptCalibration; // # of Calibration events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1TechAcceptCalibration; // # of Calibration events accepted by L1 Technical Triggers[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptRandom;      // # of Random events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1TechAcceptRandom;      // # of Random events accepted by L1 Technical Triggers[i]  
    jsoncollector::HistoJ<unsigned int> *L1Global;                // Global # of Phyics, Cailibration and Random L1 triggers 
    
    std::string stL1Jsd;                 //Definition file name for JSON with L1 rates            
    std::string streamL1Destination;
    std::string streamHLTDestination;
  };
  //End lumi struct
  //Struct for storing variable written once per run
  struct runVars{
    mutable std::atomic<bool> wroteFiles;
    mutable std::string streamL1Destination;
    mutable std::string streamHLTDestination;
  };
}//End hltJson namespace   

// 
// class declaration
// 
class TriggerJSONMonitoring : public edm::stream::EDAnalyzer <edm::RunCache<hltJson::runVars>, edm::LuminosityBlockSummaryCache<hltJson::lumiVars>>
{
 public:
  explicit TriggerJSONMonitoring(const edm::ParameterSet&);
  ~TriggerJSONMonitoring();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void analyze(edm::Event const&,
               edm::EventSetup const&);

  void beginRun(edm::Run const&,
                edm::EventSetup const&);

  static std::shared_ptr<hltJson::runVars> globalBeginRun(edm::Run const&, edm::EventSetup const&, void const*){
    std::shared_ptr<hltJson::runVars> rv(new hltJson::runVars);
    if (edm::Service<evf::EvFDaqDirector>().isAvailable()) {
      rv->streamHLTDestination = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations("streamHLTRates");
      rv->streamL1Destination = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations("streamL1Rates");
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

  void writeDefJson(std::string path);

  void writeL1DefJson(std::string path);       

  //Variables from cfg and associated tokens 
  edm::InputTag triggerResults_;                               // Input tag for TriggerResults 
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults

  edm::InputTag level1Results_;                                 // Input tag for L1 GT Readout Record   
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1t_results; // Token for L1 GT Readout Record

  //Variables that change at most once per run 
  HLTPrescaleProvider hltPrescaleProvider_; // To get at HLTConfigProvider
  HLTConfigProvider hltConfig_;         // to get configuration for HLT
  const L1GtTriggerMenu* m_l1GtMenu;    // L1 trigger menu   
  AlgorithmMap algorithmMap;            // L1 algorithm map  
  AlgorithmMap technicalMap;            // L1 technical triggeral map 

  const L1GtTriggerMask* m_l1tAlgoMask;
  const L1GtTriggerMask* m_l1tTechMask;

  std::string baseRunDir_; //Base directory from EvFDaqDirector

  // hltIndex_[ds][p] stores the hltNames_ index of the p-th path of the ds-th dataset 
  std::vector<std::vector<unsigned int> > hltIndex_;
  std::vector<std::string> hltNames_;                      // list of HLT path names 
  std::vector<std::string> datasetNames_;                  // list of dataset names 
  std::vector<std::vector<std::string> > datasetContents_; // list of path names for each dataset 

  std::vector<int> posL1s_;               // pos # of last L1 seed
  std::vector<int> posPre_;               // pos # of last HLT prescale

  std::string stHltJsd_;                  //Definition file name for JSON with rates

  std::vector<std::string> L1AlgoNames_;  // name of each L1 algorithm trigger      
  std::vector<int> L1AlgoBitNumber_;      // bit number of each L1 algo trigger     
  std::vector<std::string> L1TechNames_;  // name of each L1 technical trigger      
  std::vector<int> L1TechBitNumber_;      // bit number of each L1 tech trigger     
  std::vector<std::string> L1GlobalType_; // experimentType: Physics, Calibration, Random  

  std::string stL1Jsd_;                   //Definition file name for JSON with L1 rates           

  //Variables that need to be reset at lumi section boundaries 
  unsigned int              processed_;      // # of events processed 
  unsigned int              prescaleIndex_;  //Prescale index for each lumi section

  std::vector<unsigned int> hltWasRun_; // # of events where HLT[i] was run   
  std::vector<unsigned int> hltL1s_;    // # of events after L1 seed
  std::vector<unsigned int> hltPre_;    // # of events after HLT prescale 
  std::vector<unsigned int> hltAccept_; // # of events accepted by HLT[i]
  std::vector<unsigned int> hltReject_; // # of events rejected by HLT[i]
  std::vector<unsigned int> hltErrors_; // # of events with error in HLT[i]

  std::vector<unsigned int> hltDatasets_; // # of events accepted by each dataset 

  std::vector<unsigned int> L1AlgoAccept_;            // # of events accepted by L1T[i]  
  std::vector<unsigned int> L1TechAccept_;            // # of events accepted by L1 Technical Triggers[i]  
  std::vector<unsigned int> L1AlgoAcceptPhysics_;     // # of Physics events accepted by L1T[i]  
  std::vector<unsigned int> L1TechAcceptPhysics_;     // # of Physics events accepted by L1 Technical Triggers[i]  
  std::vector<unsigned int> L1AlgoAcceptCalibration_; // # of Calibration events accepted by L1T[i]  
  std::vector<unsigned int> L1TechAcceptCalibration_; // # of Calibration events accepted by L1 Technical Triggers[i]  
  std::vector<unsigned int> L1AlgoAcceptRandom_;      // # of Random events accepted by L1T[i]  
  std::vector<unsigned int> L1TechAcceptRandom_;      // # of Random events accepted by L1 Technical Triggers[i]  
  std::vector<unsigned int> L1Global_;                // Global # of Physics, Calibration and Random L1 triggers 

  //Variables for confirming that prescale index did not change
  unsigned int oldLumi;
  unsigned int oldPrescaleIndex;

 private:

};
#endif
