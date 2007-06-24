#ifndef HcalMonitorClient_H
#define HcalMonitorClient_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
          
#include <DQM/HcalMonitorClient/interface/HcalTBClient.h>
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>
#include <DQM/HcalMonitorClient/interface/HcalLEDClient.h>
#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

#include "TROOT.h"
#include "TTree.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

using namespace cms;
using namespace std;

class HcalMonitorClient: public EDAnalyzer{
  
public:
  
  /// Constructor
  HcalMonitorClient();
  HcalMonitorClient(const ParameterSet& ps);
  HcalMonitorClient(const ParameterSet& ps, MonitorUserInterface* mui);
  
  /// Destructor
  ~HcalMonitorClient();
  
  /// Subscribe/Unsubscribe to Monitoring Elements
  void subscribe(void);
  void subscribeNew(void);
  void unsubscribe(void);
  
  // Initialize
  void initialize(const ParameterSet& ps);
  void offlineSetup();

  /// Analyze
  void analyze(const Event& evt, const EventSetup& es);
  
  /// BeginJob
  void beginJob(const EventSetup& c);
  
  /// EndJob
  void endJob(void);
  
  /// BeginRun
  void beginRun(void);
  
  /// EndRun
  void endRun(void);
  
  /// Setup
  void setup(void);
  
  /// Cleanup
  void cleanup(void);
  
  /// HtmlOutput
  void htmlOutput(void);

  /// Create reports
  void report(bool update);
  
  /// Create tests
  void createTests(void);

  /// reset all monitor elements
  void resetAllME(void);

  //Offline output functions
  void loadHistograms(TFile* infile, const char* fname);
  void dumpHistograms(int& runNum, vector<TH1F*> &hist1d, vector<TH2F*> &hist2d);
  
  void labelBins(TH1F* hist);


private:
  void removeAll();
  
  DaqMonitorBEInterface* m_dbe;
  MonitorUserInterface* mui_;

  int ievt_;
  int mon_evt_;
  int last_mon_evt_;
  int hostPort_;  
  int run_;
  int nTimeouts_;
  int last_update_;
  int last_reset_Evts_;
  int resetUpdate_;
  int resetEvents_;
  int resetTime_;
  int nUpdateEvents_;
  int timeoutThresh_;

  bool collateSources_;
  bool cloneME_;
  bool offline_;
  bool subscribed_;
  bool verbose_;
  bool begin_run_done_;
  bool end_run_done_;
  bool forced_begin_run_;
  bool forced_end_run_;
  bool enableExit_;

  string clientName_;
  string hostName_;  
  string outputFile_;
  string inputFile_;
  string baseHtmlDir_;
  string process_;
  string location_;
  string runtype_;
  string status_;

  TH1F* trigger_;

  timeval startTime_,updateTime_;


  HcalDataFormatClient* dataformat_client_;
  HcalDigiClient* digi_client_;
  HcalRecHitClient* rechit_client_;
  HcalPedestalClient* pedestal_client_;
  HcalLEDClient* led_client_;
  HcalTBClient* tb_client_;
  HcalHotCellClient* hot_client_;

};

#endif
