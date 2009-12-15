// -*-c++-*-
// 
// Client class for HLT Scalers module.
// 
// $Id: HLTScalersClient.h,v 1.6 2009/11/22 14:17:46 puigh Exp $

// $Log: HLTScalersClient.h,v $
// Revision 1.6  2009/11/22 14:17:46  puigh
// fix compilation warning
//
// Revision 1.5  2009/11/22 13:32:38  puigh
// clean beginJob
//
// Revision 1.4  2009/11/04 03:45:18  lorenzo
// added folder param
//
// Revision 1.3  2008/08/27 13:48:57  wittich
// re-add Don's 20 entry histograms with full bin labels
//
// Revision 1.2  2008/08/24 16:34:56  wittich
// - rate calculation cleanups
// - fix error logging with LogDebug
// - report the actual lumi segment number that we think it might be
//
// Revision 1.1  2008/08/22 20:56:55  wittich
// - add client for HLT Scalers
// - Move rate calculation to HLTScalersClient and slim down the
//   filter-farm part of HLTScalers
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
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

// HARD CODE THE NUMBER OF HISTOGRAMS TO 200, LENGTH OF MONITOR to 2000 
// segments
#define MAX_PATHS 200
#define MAX_LUMI_SEG_HLT 150

class HLTScalersClient: public edm::EDAnalyzer
{
private:
  double counts_[MAX_PATHS][MAX_LUMI_SEG_HLT];
  std::ofstream textfile_;

  // this is double rather than int cuz that's what
  // the histogram we use has. Don't blame me.
public:
  typedef std::pair<int,double> CountLS_t;
  typedef std::deque<CountLS_t> CountLSFifo_t;
private:
  std::vector<CountLSFifo_t> recentPathCountsPerLS_;
  
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

  int nev_; // Number of events processed
  int nLumi_; // number of lumi blocks
  int currentRun_;

  MonitorElement *currentRate_;
  int currentLumiBlockNumber_;
  MonitorElement *rateHistories_[MAX_PATHS]; // HARD CODE FOR NOW
  MonitorElement *countHistories_[MAX_PATHS]; // HARD CODE FOR NOW

  MonitorElement *hltCurrentRate_[MAX_PATHS];
  MonitorElement *hltRate_; // global rate - any accept
  MonitorElement *updates_;
  
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
