#ifndef GUARD_HcalDetDiagNoiseMonitorClient_H
#define GUARD_HcalDetDiagNoiseMonitorClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalDetDiagNoiseMonitorClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalDetDiagNoiseMonitorClient();
  /// Destructor
  ~HcalDetDiagNoiseMonitorClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob();

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
  void htmlOutput(int run, string htmlDir, string htmlName);
  void htmlExpertOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  ///process report
  void report();
  
  void resetAllME();
  void createTests();

private:

  MonitorElement* metall;
  MonitorElement* metnoise;
  MonitorElement* metphysics;
  MonitorElement* nLS;
  MonitorElement* jetetall;
  MonitorElement* jetetnoise;
  MonitorElement* Met_AllEvents_Rate;
  MonitorElement* Met_passingTrigger_HcalNoiseCategory_Rate;
  MonitorElement* Met_passingTrigger_PhysicsCategory_Rate;
  MonitorElement* Jets_Et_passing_selections_Rate;
  MonitorElement* Noise_Jets_Et_passing_selections_Rate;

};


#endif
