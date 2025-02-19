#ifndef HcalMonitorClient_GUARD_H
#define HcalMonitorClient_GUARD_H

/*
 * \file HcalMonitorClient.h
 * 
 * $Date: 2011/04/12 18:25:42 $
 * $Revision: 1.49 $
 * \author J. Temple
 * 
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

class DQMStore;
class HcalChannelQuality;
class HcalSummaryClient;

class HcalMonitorClient: public edm::EDAnalyzer
{

public:

  // Constructor
  HcalMonitorClient(const edm::ParameterSet & ps);
  
  // Destructor
  virtual ~HcalMonitorClient();

 /// Analyze
  void analyze(int LS=-1);
  void analyze(const edm::Event & e, const edm::EventSetup & c);
  
  /// BeginJob
  void beginJob(void);
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun();
  void beginRun(const edm::Run & r, const edm::EventSetup & c);
  
  /// EndRun
  void endRun();
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  
  /// BeginLumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);
  
  /// EndLumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);
  
  /// Reset
  void reset(void);
  
  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
  /// SoftReset
  void softReset(bool flag);
 
  // Write channelStatus info
  void writeChannelStatus();
  
  // Write html output
  void writeHtml();

  void PlotPedestalValues(const HcalDbService& cond);

private:
  // Event counters
  int ievt_; // all events
  int jevt_; // events in current run
  int run_;
  int evt_;
  bool begin_run_;
  bool end_run_;

  // parameter set inputs
  int debug_;
  std::string inputFile_;
  bool mergeRuns_;
  bool cloneME_;
  int prescaleFactor_;
  std::string prefixME_;
  bool enableCleanup_;
  std::vector<std::string > enabledClients_;

  int updateTime_; // update time for updating histograms 
  std::string baseHtmlDir_;
  int htmlUpdateTime_; //update time for updating histograms
  std::string databasedir_;
  int databaseUpdateTime_; //update time for dumping db info (offset by 10 minutes, so that always dump after 10 minutes)
  int databaseFirstUpdate_; // first update time (in minutes)
  int htmlFirstUpdate_; // first update for html
  
  int htmlcounter_;

  bool saveByLumiSection_;  //produces separate LS certification values when enabled
  bool Online_;  // fix to April 2011 problem where online DQM client crashes in endJob.  Is endRun perhaps not called?

  // time parameters
  time_t current_time_;
  time_t last_time_update_;
  time_t last_time_html_;
  time_t last_time_db_;

  std::vector<HcalBaseDQClient*> clients_;  

  DQMStore* dqmStore_;
  HcalChannelQuality* chanquality_;

  HcalSummaryClient* summaryClient_;
  EtaPhiHists* ChannelStatus;
  EtaPhiHists* ADC_PedestalFromDBByDepth;
  EtaPhiHists* ADC_WidthFromDBByDepth;
  EtaPhiHists* fC_PedestalFromDBByDepth;
  EtaPhiHists* fC_WidthFromDBByDepth;

};

#endif
