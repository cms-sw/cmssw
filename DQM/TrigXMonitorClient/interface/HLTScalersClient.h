// -*-c++-*-
// 
// Client class for HLT Scalers module.
// 
// $Id$

// $Log$

#ifndef HLTSCALERSCLIENT_H
#define HLTSCALERSCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// HARD CODE THE NUMBER OF HISTOGRAMS TO 200
#define MAX_PATHS 200

class HLTScalersClient: public edm::EDAnalyzer
{
  /// Constructors
  HLTScalersClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~HLTScalersClient() {
  };
  
  /// BeginJob
  void beginJob(const edm::EventSetup& c);

//   /// Endjob
//   void endJob(void);
  
  /// BeginRun
  void beginRun(const edm::Run& run, const edm::EventSetup& c);

  /// EndRun
  void endRun(const edm::Run& run, const edm::EventSetup& c);

  
  /// End LumiBlock
  /// DQM Client Diagnostic should be performed here
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
                          const edm::EventSetup& c);

  // unused
  void analyze(const edm::Event& e, const edm::EventSetup& c) ;


private:
  DQMStore * dbe_;
//   edm::InputTag trigResultsSource_;
//   edm::InputTag l1GtDataSource_; // L1 Scalers

  int nev_; // Number of events processed
  int nLumi_; // number of lumi blocks
  int currentRun_;

  unsigned long int scalerCounters_[MAX_PATHS];
  MonitorElement *currentRate_;
  int currentLumiBlockNumber_;
  MonitorElement *rateHistories_[MAX_PATHS]; // HARD CODE FOR NOW
};


#endif // HLTSCALERSCLIENT_H
