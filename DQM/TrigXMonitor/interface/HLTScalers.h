// -*-c++-*-
// 
//
// $Id$
// Class to collect HLT scaler information 
// for Trigger Cross Section Monitor
// [wittich 11/07] 

// $Log$

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
  edm::InputTag trigResultsSource_;
  bool resetMe_, verbose_, monitorDaemon_;
  int nev_; // Number of events processed
};

#endif // HLTSCALERS_H
