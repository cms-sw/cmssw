#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

struct DeadCellHists{
  int type;
  TH2F* deadADC_OccMap;
  TH1F* deadADC_Eta;
  TH2F* badCAPID_OccMap;
  TH1F* badCAPID_Eta;
  TH1F* ADCdist;
  TH2F* NADACoolCellMap;
  TH2F* digiCheck;
  TH2F* cellCheck;
  TH2F* AbovePed;
  TH2F* CoolCellBelowPed;
  std::vector<TH2F*> DeadCap;
};

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
  

  ///process report
  void report();
  
  /// WriteDB
  void htmlOutput(int run, std::string htmlDir, std::string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  void resetAllME();
  void createTests();
  void createSubDetTests(DeadCellHists& hist);

  // Clear histograms
  void clearHists(DeadCellHists& hist);
  void deleteHists(DeadCellHists& hist);

  void getSubDetHistograms(DeadCellHists& hist);
  void resetSubDetHistograms(DeadCellHists& hist);
  void getSubDetHistogramsFromFile(DeadCellHists& hist, TFile* infile);
  void htmlSubDetOutput(DeadCellHists& hist, int runNo, 
			std::string htmlDir, 
			std::string htmlName);

private:

  ofstream htmlFile;

  DeadCellHists hbhists, hehists, hohists, hfhists, hcalhists;
  
};

#endif
