#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQM/HcalMonitorClient/interface/HcalClientUtils.h"
#include "DQM/HcalMonitorClient/interface/HcalHistoUtils.h"


class HcalDeadCellClient : public HcalBaseClient {
  
 public:
  
  /// Constructor
  HcalDeadCellClient();
  /// Destructor
  ~HcalDeadCellClient();

  void init(const edm::ParameterSet& ps, DQMStore* dbe, string clientName);

  /// Analyze
  void analyze(void);
  
  /// BeginJob
  void beginJob(const EventSetup& c);
  
  /// EndJob
  void endJob(std::map<HcalDetId, unsigned int>& myqual); 


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

  // Introduce temporary error/warning checks
  bool hasErrors_Temp();
  bool hasWarnings_Temp();
  bool hasOther_Temp() {return false;}
private:
  
  vector <std::string> subdets_;

  double minErrorFlag_;  // minimum error rate which causes problem cells to be dumped in client
  bool deadclient_makeDiagnostics_;

  bool deadclient_test_neverpresent_;
  bool deadclient_test_occupancy_;
  bool deadclient_test_energy_;
  bool dump2database_;

  int deadclient_checkNevents_;

  // Histograms

  TH2F* ProblemDeadCells;
  TH2F* ProblemDeadCellsByDepth[4];
  TH2F* UnoccupiedDeadCellsByDepth[4];
  TH2F* DigiNeverPresentByDepth[4];
  TH2F* BelowEnergyThresholdCellsByDepth[4];

  TH1F* NumberOfDeadCells;
  TH1F* NumberOfDeadCellsHB;
  TH1F* NumberOfDeadCellsHE;
  TH1F* NumberOfDeadCellsHO;
  TH1F* NumberOfDeadCellsHF;
  TH1F* NumberOfDeadCellsZDC;

  TH1F* NumberOfNeverPresentCells;
  TH1F* NumberOfNeverPresentCellsHB;
  TH1F* NumberOfNeverPresentCellsHE;
  TH1F* NumberOfNeverPresentCellsHO;
  TH1F* NumberOfNeverPresentCellsHF;
  TH1F* NumberOfNeverPresentCellsZDC;

  TH1F* NumberOfUnoccupiedCells;
  TH1F* NumberOfUnoccupiedCellsHB;
  TH1F* NumberOfUnoccupiedCellsHE;
  TH1F* NumberOfUnoccupiedCellsHO;
  TH1F* NumberOfUnoccupiedCellsHF;
  TH1F* NumberOfUnoccupiedCellsZDC;

  TH1F* NumberOfBelowEnergyCells;
  TH1F* NumberOfBelowEnergyCellsHB;
  TH1F* NumberOfBelowEnergyCellsHE;
  TH1F* NumberOfBelowEnergyCellsHO;
  TH1F* NumberOfBelowEnergyCellsHF;
  TH1F* NumberOfBelowEnergyCellsZDC;

};

#endif
