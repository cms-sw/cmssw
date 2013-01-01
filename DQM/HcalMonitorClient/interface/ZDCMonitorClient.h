#ifndef ZDCMonitorClient_H
#define ZDCMonitorClient_H

// Update on September 21, 2012 to match HcalMonitorClient
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"

#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"
#include "DQM/HcalMonitorClient/interface/HcalSummaryClient.h"

class DQMStore;
//class TH2F;
//class TH1F;
//class TFile;

class ZDCMonitorClient : public edm::EDAnalyzer{
  
public:
  
  /// Constructors
  //ZDCMonitorClient();
  ZDCMonitorClient(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~ZDCMonitorClient();

  /// Analyze
  void analyze(int LS=-1);
  void analyze(const edm::Event& evt, const edm::EventSetup& es);
  
  /// BeginJob
  void beginJob();

  /// EndJob
  void endJob(void);

  /// BeginRun
  void beginRun();
  void beginRun(const edm::Run& r, const edm::EventSetup & c);

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

 private:

  int ievt_; // all events
  int jevt_; // events in current run
  int run_;
  int evt_;
  bool begin_run_;
  bool end_run_;

  /////New Variables as of Fall 2012
  int LumiCounter;
  int PZDC_GoodLumiCounter;
  int NZDC_GoodLumiCounter;
  double PZDC_LumiRatio;
  double NZDC_LumiRatio;
  //////end new variables of Fall 2012////

  // parameter set inputs

  std::vector<double> ZDCGoodLumi_;

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
 std::string subdir_;

  // time parameters
  time_t current_time_;
  time_t last_time_update_;
  time_t last_time_html_;
  time_t last_time_db_;

  std::vector<HcalBaseDQClient*> clients_;  

  DQMStore* dqmStore_;
  HcalChannelQuality* chanquality_;

  HcalSummaryClient* summaryClient_;

  ///////////////////New plots as of Fall 2012/////////////
  MonitorElement* ZDCChannelSummary_;
  MonitorElement* ZDCHotChannelFraction_;
  MonitorElement* ZDCColdChannelFraction_;
  MonitorElement* ZDCDeadChannelFraction_;
  MonitorElement* ZDCDigiErrorFraction_;
  MonitorElement* ZDCReportSummary_;

  /////////////new plots as of Fall 2012//////////////////
};

#endif
