#ifndef HcalMonitorClient_GUARD_H
#define HcalMonitorClient_GUARD_H

/*
 * \file HcalMonitorClient.h
 * 
 * \author J. Temple
 * 
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMEDHarvester.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "DQM/HcalMonitorClient/interface/HcalBaseDQClient.h"
#include "DQM/HcalMonitorTasks/interface/HcalEtaPhiHists.h"

#include "DataFormats/Provenance/interface/RunLumiEventNumber.h"

class HcalChannelQuality;
class HcalSummaryClient;

class HcalMonitorClient: public DQMEDHarvester
{

public:

  // Constructor
  HcalMonitorClient(const edm::ParameterSet & ps);
  
  // Destructor
  virtual ~HcalMonitorClient();

 /// Analyze
  void analyze(DQMStore::IBooker &ib, DQMStore::IGetter &, int LS=-1);
  //void analyze(const edm::Event & e, const edm::EventSetup & c);
  
  /// EndJob
  void dqmEndJob(DQMStore::IBooker &, DQMStore::IGetter &);
  
  /// BeginRun
  //void beginRun();
  void beginRun(const edm::Run & r, const edm::EventSetup & c);
  
  /// EndRun
  void endRun(DQMStore::IBooker &, DQMStore::IGetter &);
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  
  
  /// EndLumiBlock
  void dqmEndLuminosityBlock(DQMStore::IBooker &, DQMStore::IGetter &, const edm::LuminosityBlock & l, const edm::EventSetup & c);
  
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
  void writeHtml(DQMStore::IBooker &, DQMStore::IGetter &);

  void PlotPedestalValues(const HcalDbService& cond);

private:
  // Event counters
  int ievt_; // all events
  int jevt_; // events in current run
  edm::RunNumber_t run_;
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

  const HcalTopology* hctopo_;
  const HcalChannelQuality* chanquality_;

  HcalSummaryClient* summaryClient_;
  EtaPhiHists* ChannelStatus;
  EtaPhiHists* ADC_PedestalFromDBByDepth;
  EtaPhiHists* ADC_WidthFromDBByDepth;
  EtaPhiHists* fC_PedestalFromDBByDepth;
  EtaPhiHists* fC_WidthFromDBByDepth;

  // -- 
  bool doPedSetup_;
  // This function creates the EtaPhiHists for pedestal monitoring
  // The doPedSetup_ flag is set to false during the execution of the function.
  void setupPedestalMon(DQMStore::IBooker &);

  bool doChanStatSetup_;
  // The setupChannelStatusMon function creates the EtaPhiHists to record ChannelStatus.
  // It will be retrieved by the HcalSummaryClient in the analysis(LS) method.
  // The doChanStatSetup_ flag is set to false during the xectuion of the function.
  void setupChannelStatusMon(DQMStore::IBooker &);

};

#endif
