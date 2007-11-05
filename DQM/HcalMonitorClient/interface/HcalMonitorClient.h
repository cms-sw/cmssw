#ifndef HcalMonitorClient_H
#define HcalMonitorClient_H


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Components/interface/DQMAnalyzer.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/CollateMonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
          
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalDataFormatClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDigiClient.h>
#include <DQM/HcalMonitorClient/interface/HcalRecHitClient.h>
#include <DQM/HcalMonitorClient/interface/HcalPedestalClient.h>
#include <DQM/HcalMonitorClient/interface/HcalLEDClient.h>
#include <DQM/HcalMonitorClient/interface/HcalHotCellClient.h>
#include <DQM/HcalMonitorClient/interface/HcalDeadCellClient.h>
#include <DQM/HcalMonitorModule/interface/HcalMonitorSelector.h>

#include <DQM/HcalMonitorClient/interface/HcalDQMDbInterface.h>

#include "TROOT.h"
#include "TTree.h"
#include "TGaxis.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>

using namespace std;

class HcalMonitorClient: public DQMAnalyzer{
  
public:
  
  /// Constructors
  HcalMonitorClient();
  HcalMonitorClient(const ParameterSet& ps);
  
  /// Destructor
  ~HcalMonitorClient();
  
  // Initialize
  void initialize(const ParameterSet& ps);
  void offlineSetup();

  /// Analyze
  void analyze(void);
  void analyze(const Event& evt, const EventSetup& es);
  
  /// BeginJob
  void beginJob(const EventSetup& c);
  /// BeginRun
  void beginRun(const Run& r, const edm::EventSetup & c);
  /// BeginLumiBlock
  void beginLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);


  /// EndJob
  void endJob(void);
  /// EndRun
  void endRun(const edm::Run & r, const edm::EventSetup & c);
  /// EndLumiBlock
  void endLuminosityBlock(const edm::LuminosityBlock & l, const edm::EventSetup & c);

  
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
  
private:
  void removeAllME(void);

  int ievt_;
  int resetUpdate_;
  int resetEvents_;
  int resetTime_;
  int lastResetTime_;
  int resetLS_;
  
  bool runningStandalone_;
  bool enableExit_;
  bool enableMonitorDaemon_;

  string inputFile_;
  string baseHtmlDir_;

  MonitorUserInterface* mui_;
  HcalDataFormatClient* dataformat_client_;
  HcalDigiClient* digi_client_;
  HcalRecHitClient* rechit_client_;
  HcalPedestalClient* pedestal_client_;
  HcalLEDClient* led_client_;
  HcalHotCellClient* hot_client_;
  HcalDeadCellClient* dead_client_;
  HcalHotCellDbInterface* dqm_db_;

};

#endif
