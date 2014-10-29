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

#include "FWCore/Framework/interface/ESHandle.h" //DS                                                                                  
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h" //DS                                                                     
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h" //DS                                                                  
//#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h" //DS                                                 

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h" //DS                                                 


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
    std::string stHltJsd;   //Definition file name for JSON with rates                                                          

    HistoJ<unsigned int> *L1Global;     // Global # of Phyics, Cailibration and Random L1 triggers //DS  
    HistoJ<unsigned int> *L1Accept;     // # of events accepted by L1T[i] //DS                                                         
    HistoJ<unsigned int> *L1TechAccept; // # of events accepted by L1 Technical Triggers[i] //DS                                       
    HistoJ<std::string> *L1Names;       // list of L1 algorithm names    //DS                                                          
    HistoJ<std::string> *L1TechNames;   // list of L1 technical trigger names    //DS                                                  
    HistoJ<std::string> *L1GlobalType;  // list of L1 technical trigger names    //DS                                                  
    
    std::string stL1Jsd;                 //Definition file name for JSON with L1 rates            //DS                                  

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

  void writeL1DefJson(std::string path);       //DS                                                                                    

  //Variables from cfg and associated tokens                                                                                           
  edm::InputTag triggerResults_;                               // Input tag for TriggerResults                                         
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;  // Token for TriggerResults                                             

  edm::InputTag m_l1GtObjectMapTag;                            // input tag for L1 GT Object Maps //DS                                 
  edm::EDGetTokenT<L1GlobalTriggerObjectMapRecord> m_l1GtObjectMapToken; //Token forL1 GT Object Maps //DS                             

  //Variables that change at most once per run                                                                                         
  HLTConfigProvider hltConfig_;         // to get configuration for HLT    

  const L1GtTriggerMenu* m_l1GtMenu;    // L1 trigger menu  //DS                                                                       
  AlgorithmMap algorithmMap;            // L1 algorithm map //DS                                                                       
  AlgorithmMap technicalMap;            // L1 technical triggeral map //DS                                                             

  std::string baseRunDir_; //Base directory from EvFDaqDirector                                                                        

  // hltIndex_[ds][p] stores the hltNames_ index of the p-th path of the ds-th dataset                                                 
  std::vector<std::vector<unsigned int> > hltIndex_;
  std::vector<std::string> hltNames_;                      // list of HLT path names                                                   
  std::vector<std::string> datasetNames_;                  // list of dataset names                                                    
  std::vector<std::vector<std::string> > datasetContents_; // list of path names for each dataset                                      

  std::vector<int> posL1s_;               // pos # of last L1 seed                                                                       
  std::vector<int> posPre_;               // pos # of last HLT prescale                                                                  

  std::string stHltJsd_;                  //Definition file name for JSON with rates                                                     

  std::vector<std::string> L1Names_;      // name of each L1 algorithm trigger      //DS                                                
  std::vector<std::string> L1TechNames_;  // name of each L1 technical trigger      //DS                                                
  std::vector<std::string> L1GlobalType_; // experimentType: Physics, Calibration, Random  //DS                                         

  std::string stL1Jsd_;                   //Definition file name for JSON with L1 rates           //DS                                   

  //Variables that need to be reset at lumi section boundaries                                                                         
  unsigned int              processed_; // # of events processed                                                                       

  std::vector<unsigned int> hltWasRun_; // # of events where HLT[i] was run   
  std::vector<unsigned int> hltL1s_;    // # of events after L1 seed                                                                   
  std::vector<unsigned int> hltPre_;    // # of events after HLT prescale                                                              
  std::vector<unsigned int> hltAccept_; // # of events accepted by HLT[i]                                                              
  std::vector<unsigned int> hltReject_; // # of events rejected by HLT[i]                                                              
  std::vector<unsigned int> hltErrors_; // # of events with error in HLT[i]                                                            

  std::vector<unsigned int> hltDatasets_; // # of events accepted by each dataset                                                      

  std::vector<unsigned int> L1Global_;     // Global # of Phyics, Cailibration and Random L1 triggers //DS                             
  std::vector<unsigned int> L1Accept_;     // # of events accepted by L1T[i]  //DS                                                     
  std::vector<unsigned int> L1TechAccept_; // # of events accepted by L1 Technical Triggers[i]  //DS                                   

 private:

};
#endif
