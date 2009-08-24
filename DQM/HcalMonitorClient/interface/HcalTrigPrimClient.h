#ifndef HcalTrigPrimClient_H
#define HcalTrigPrimClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

class HcalTrigPrimClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalTrigPrimClient();
  
  /// Destructor
  ~HcalTrigPrimClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);    

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob(void);
  
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
  
  void report();
  
  /// HtmlOutput
  void htmlOutput(int run, string htmlDir, string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);
  
  void resetAllME();
  void createTests();

 private:
  std::map< std::string, TH1* > histo1d;
  std::map< std::string, TH1* > histo2d;
};

#endif
