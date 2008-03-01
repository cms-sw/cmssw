#ifndef L1Scalers_H
#define L1Scalers_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class L1Scalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  L1Scalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~L1Scalers() {};
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

//   /// Endjob
   void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  
//   /// Begin LumiBlock
//   void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
//                             const edm::EventSetup& c) ;

  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
  edm::InputTag scalersSource_;
//  edm::InputTag triggerScalersSource_;
//  edm::InputTag triggerRatesSource_;
//  edm::InputTag lumiScalersSource_;
  
  std::string outputFile_;	//file name for ROOT ouput
  bool verbose_, monitorDaemon_;
  int nev_; // Number of events processed
//  const L1TriggerScalers *previousTrig;


  MonitorElement * orbitNum;
  MonitorElement * trigNum;
  MonitorElement * eventNum;
  MonitorElement * finalTrig;
  MonitorElement * randTrig;
  MonitorElement * numberResets;
  MonitorElement * deadTime;
  MonitorElement * lostFinalTriggers;

  MonitorElement * trigNumRate;
  MonitorElement * eventNumRate;
  MonitorElement * finalTrigRate;
  MonitorElement * randTrigRate;
  MonitorElement * orbitNumRate;
  MonitorElement * numberResetsRate;
  MonitorElement * deadTimePercent;
  MonitorElement * lostFinalTriggersPercent;
  

  MonitorElement *  instLumi;
  MonitorElement *  instLumiErr; 
  MonitorElement *  instLumiQlty; 
  MonitorElement *  instEtLumi; 
  MonitorElement *  instEtLumiErr; 
  MonitorElement *  instEtLumiQlty; 
  MonitorElement *  sectionNum; 
  MonitorElement *  startOrbit; 
  MonitorElement *  numOrbits; 

};

#endif // L1Scalers_H
