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

  TProfile* NumberOfDeadCells;
  TProfile* NumberOfDeadCellsHB;
  TProfile* NumberOfDeadCellsHE;
  TProfile* NumberOfDeadCellsHO;
  TProfile* NumberOfDeadCellsHF;
  TProfile* NumberOfDeadCellsZDC;

  TProfile* NumberOfNeverPresentCells;
  TProfile* NumberOfNeverPresentCellsHB;
  TProfile* NumberOfNeverPresentCellsHE;
  TProfile* NumberOfNeverPresentCellsHO;
  TProfile* NumberOfNeverPresentCellsHF;
  TProfile* NumberOfNeverPresentCellsZDC;

  TProfile* NumberOfUnoccupiedCells;
  TProfile* NumberOfUnoccupiedCellsHB;
  TProfile* NumberOfUnoccupiedCellsHE;
  TProfile* NumberOfUnoccupiedCellsHO;
  TProfile* NumberOfUnoccupiedCellsHF;
  TProfile* NumberOfUnoccupiedCellsZDC;

  TProfile* NumberOfBelowEnergyCells;
  TProfile* NumberOfBelowEnergyCellsHB;
  TProfile* NumberOfBelowEnergyCellsHE;
  TProfile* NumberOfBelowEnergyCellsHO;
  TProfile* NumberOfBelowEnergyCellsHF;
  TProfile* NumberOfBelowEnergyCellsZDC;

};

#endif
