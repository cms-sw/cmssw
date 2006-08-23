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

#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace cms;
using namespace std;

class HcalMonitorClient: public EDAnalyzer{
  
public:
  
  
  /// Constructor
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

  
private:
  

  int ievt_;
  int jevt_;

  int timeout_;
  int timeout_thresh_;

  bool collateSources_;
  bool cloneME_;
  MonitorUserInterface* mui_;
  string clientName_;
  string hostName_;
  int hostPort_;
  
  bool verbose_;
  
  
  string outputFile_;
  bool enableSubRun_;
  int subrun_;
  string baseHtmlDir_;
  string process_;
  
  string location_;
  string runtype_;
  string status_;
  int run_;
  int mon_evt_;
  int report_;

  bool begin_run_done_;
  bool end_run_done_;
  bool forced_begin_run_;
  bool forced_end_run_;
  bool enableExit_;

  int last_update_;
  int update_freq_;
  int last_jevt_;
  int unknowns_;
  
  HcalDataFormatClient* dataformat_client_;
  HcalDigiClient* digi_client_;
  HcalRecHitClient* rechit_client_;
  HcalPedestalClient* pedestal_client_;
  HcalLEDClient* led_client_;
  HcalTBClient* tb_client_;

};

#endif
