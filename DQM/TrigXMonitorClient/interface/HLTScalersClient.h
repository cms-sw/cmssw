// -*-c++-*-
// 
// Client class for HLT Scalers module.
// 
// $Id: HLTScalersClient.h,v 1.10 2010/02/11 23:55:18 wittich Exp $

// $Log: HLTScalersClient.h,v $
// Revision 1.10  2010/02/11 23:55:18  wittich
// - adapt to shorter Lumi Section length
// - fix bug in how history of counts was filled
//
// Revision 1.9  2010/02/11 00:11:09  wmtan
// Adapt to moved framework header
//
// Revision 1.8  2010/02/02 11:44:20  wittich
// more diagnostics for online scalers
//
// Revision 1.7  2009/12/15 20:41:16  wittich
// better hlt scalers client
//
// Revision 1.6  2009/11/22 14:17:46  puigh
// fix compilation warning
//
// Revision 1.5  2009/11/22 13:32:38  puigh
// clean beginJob
//

#ifndef HLTSCALERSCLIENT_H
#define HLTSCALERSCLIENT_H
#include <fstream>
#include <vector>
#include <deque>
#include <utility> 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// HARD CODE THE NUMBER OF HISTOGRAMS TO 200, LENGTH OF MONITOR to 2400 
// segments
#define MAX_PATHS 200
#define MAX_LUMI_SEG_HLT 2400

class HLTScalersClient: public edm::EDAnalyzer
{
private:
  std::ofstream textfile_;

public:
  typedef std::pair<int,double> CountLS_t;
  typedef std::deque<CountLS_t> CountLSFifo_t;
private:
  std::vector<CountLSFifo_t> recentPathCountsPerLS_;
  CountLSFifo_t recentOverallCountsPerLS_;
  
public:
  /// Constructors
  HLTScalersClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~HLTScalersClient() {
    if ( debug_ ) {
      textfile_.close();
    }
  };
  
  /// BeginJob
  void beginJob(void);

  
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

  int nev_; // Number of events processed
  int nLumi_; // number of lumi blocks
  int currentRun_;

  MonitorElement *currentRate_;
  int currentLumiBlockNumber_;
  std::vector<MonitorElement*> rateHistories_; 
  std::vector<MonitorElement*> countHistories_; 

  std::vector<MonitorElement*> hltCurrentRate_;
  MonitorElement *hltRate_; // global rate - any accept
  MonitorElement *hltCount_; // globalCounts
  MonitorElement *updates_;
  MonitorElement *mergeCount_;
  
  bool first_, missingPathNames_;
  std::string folderName_;
  unsigned int kRateIntegWindow_;
  std::string processName_;
  HLTConfigProvider hltConfig_;
  std::deque<int> ignores_;
  std::pair<double,double> getSlope_(CountLSFifo_t points);
  bool debug_;
};


#endif // HLTSCALERSCLIENT_H
