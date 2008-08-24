// -*-c++-*-
// 
// Client class for HLT Scalers module.
// 
// $Id: HLTScalersClient.h,v 1.1 2008/08/22 20:56:55 wittich Exp $

// $Log: HLTScalersClient.h,v $
// Revision 1.1  2008/08/22 20:56:55  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
//

#ifndef HLTSCALERSCLIENT_H
#define HLTSCALERSCLIENT_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

// HARD CODE THE NUMBER OF HISTOGRAMS TO 200, LENGTH OF MONITOR to 2000 
// segments
#define MAX_PATHS 200
#define MAX_LUMI_SEG 2000

class HLTScalersClient: public edm::EDAnalyzer
{
public:
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
