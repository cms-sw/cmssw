#ifndef JSONMonitoring_L1TriggerJSONMonitoring_h
#define JSONMonitoring_L1TriggerJSONMonitoring_h

/** \class L1TriggerJSONMonitoring         
 *     
 *  
 *  Description: This class prints JSON files with L1 trigger info.
 *          
 *  Created:  Fri, 11 Mar 2016     
 *       
 *  \author Aram Avetisyan   
 * 
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
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
#include "FWCore/Framework/interface/ESHandle.h"          

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"
#include "CondFormats/L1TObjects/interface/L1TUtmAlgorithm.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"

#include <atomic>

namespace l1Json {
  //Struct for storing variables that must be written and reset every lumi section 
  struct lumiVars {

    unsigned int prescaleIndex; // Prescale index for each lumi section

    std::string baseRunDir; //Base directory from EvFDaqDirector 

    jsoncollector::HistoJ<unsigned int> *processed;               // # of events processed
    jsoncollector::HistoJ<unsigned int> *L1AlgoAccept;            // # of events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptPhysics;     // # of Physics events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptCalibration; // # of Calibration events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1AlgoAcceptRandom;      // # of Random events accepted by L1T[i]  
    jsoncollector::HistoJ<unsigned int> *L1Global;                // Global # of Phyics, Cailibration and Random L1 triggers 
    
    std::string stL1Jsd;                 //Definition file name for JSON with L1 rates            
    std::string streamL1Destination;
    std::string streamL1MergeType;
  };
  //End lumi struct
  //Struct for storing variable written once per run
  struct runVars{
    mutable std::atomic<bool> wroteFiles;
    mutable std::string streamL1Destination;
    mutable std::string streamL1MergeType;
  };
}//End l1Json namespace   

// 
// class declaration
// 
class L1TriggerJSONMonitoring : public edm::stream::EDAnalyzer <edm::RunCache<l1Json::runVars>, edm::LuminosityBlockSummaryCache<l1Json::lumiVars>>
{
 public:
  explicit L1TriggerJSONMonitoring(const edm::ParameterSet&);
  ~L1TriggerJSONMonitoring();
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

  void analyze(edm::Event const&,
               edm::EventSetup const&);

  void beginRun(edm::Run const&,
                edm::EventSetup const&);

  static std::shared_ptr<l1Json::runVars> globalBeginRun(edm::Run const&, edm::EventSetup const&, void const*){
    std::shared_ptr<l1Json::runVars> rv(new l1Json::runVars);
    if (edm::Service<evf::EvFDaqDirector>().isAvailable()) {
      rv->streamL1Destination = edm::Service<evf::EvFDaqDirector>()->getStreamDestinations("streamL1Rates");
      rv->streamL1MergeType = edm::Service<evf::EvFDaqDirector>()->getStreamMergeType("streamL1Rates",evf::MergeTypeJSNDATA);
    }
    rv->wroteFiles = false;
    return rv;
  }
  
  static void globalEndRun(edm::Run const& iRun, edm::EventSetup const&, RunContext const* iContext){ } 

  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  static std::shared_ptr<l1Json::lumiVars> globalBeginLuminosityBlockSummary(edm::LuminosityBlock const&,
                                                                              edm::EventSetup const&,
                                                                              LuminosityBlockContext const*);

  void endLuminosityBlockSummary(edm::LuminosityBlock const&,
                                 edm::EventSetup const&,
                                 l1Json::lumiVars*) const;


  static void globalEndLuminosityBlockSummary(edm::LuminosityBlock const&,
                                              edm::EventSetup const&,
                                              LuminosityBlockContext const*,
                                              l1Json::lumiVars*);

  void resetLumi(); //Reset all counters 

  void writeL1DefJson(std::string path);       

  //Variables from cfg and associated tokens 
  edm::InputTag level1Results_;                                  // Input tag for L1 Global collection   
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> level1ResultsToken_; // Token for L1 Global collection 

  //Variables that change at most once per run 
  std::string baseRunDir_; //Base directory from EvFDaqDirector

  std::vector<std::string> L1AlgoNames_;  // name of each L1 algorithm trigger      
  std::vector<int> L1AlgoBitNumber_;      // bit number of each L1 algo trigger     
  std::vector<std::string> L1GlobalType_; // experimentType: Physics, Calibration, Random  

  std::string stL1Jsd_;                   //Definition file name for JSON with L1 rates           

  //Variables that need to be reset at lumi section boundaries 
  unsigned int              processed_;      // # of events processed 
  unsigned int              prescaleIndex_;  //Prescale index for each lumi section

  std::vector<unsigned int> L1AlgoAccept_;            // # of events accepted by L1T[i]  
  std::vector<unsigned int> L1AlgoAcceptPhysics_;     // # of Physics events accepted by L1T[i]  
  std::vector<unsigned int> L1AlgoAcceptCalibration_; // # of Calibration events accepted by L1T[i]  
  std::vector<unsigned int> L1AlgoAcceptRandom_;      // # of Random events accepted by L1T[i]  
  std::vector<unsigned int> L1Global_;                // Global # of Physics, Calibration and Random L1 triggers 

  //Variables for confirming that prescale index did not change
  unsigned int oldLumi;
  unsigned int oldPrescaleIndex;

 private:

};
#endif
