// -*-c++-*-
// 
//
// $Id: HLTScalers.h,v 1.2 2007/12/01 19:28:56 wittich Exp $
// Class to collect HLT scaler information 
// for Trigger Cross Section Monitor
// [wittich 11/07] 

// $Log: HLTScalers.h,v $
// Revision 1.2  2007/12/01 19:28:56  wittich
// - fix cfi file (debug -> verbose, HLT -> FU for TriggerResults  label)
// - handle multiple beginRun for same run (don't call reset on DQM )
// - remove PathTimerService from cfg file in test subdir
//
// Revision 1.1  2007/11/26 16:37:50  wittich
// Prototype HLT scaler information.
//

#ifndef HLTSCALERS_H
#define HLTSCALERS_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class HLTScalers: public edm::EDAnalyzer
{
public:
  /// Constructors
  HLTScalers(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~HLTScalers() {};
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

//   /// Endjob
//   void endJob(void);
  
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
  DaqMonitorBEInterface * dbe_;
  MonitorElement *scalers_;
  MonitorElement *detailedScalers_;
  MonitorElement *l1scalers_;
  MonitorElement *nProc_;
  std::vector<MonitorElement*> hltPathNames_;
  edm::InputTag trigResultsSource_;
  edm::InputTag l1GtDataSource_; // L1 Scalers
  bool resetMe_, verbose_, monitorDaemon_;
  int nev_; // Number of events processed
  int currentRun_;
};

#endif // HLTSCALERS_H
