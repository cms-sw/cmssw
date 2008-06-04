#ifndef HcalDeadCellClient_H
#define HcalDeadCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"

struct DeadCellHists{
  int type;
  TH2F* problemDeadCells;
  TH2F* problemDeadCells_depth1;
  TH2F* problemDeadCells_depth2;
  TH2F* problemDeadCells_depth3;
  TH2F* problemDeadCells_depth4;

  std::vector <TH2F*> problemDeadCells_DEPTH;
 

  // Dead cell routine #1:  low ADC counts for cell
  TH2F* deadADC_map;
  std::vector<TH2F*> deadADC_map_depth; // individual depth plots
  TH1F* deadADC_eta;
  TH1F* ADCdist;
  std::vector<TH2F*> deadcapADC_map; // plots for individual CAPIDs

  // Dead cell routine #2:  cell cool compared to neighbors
  double floor, mindiff;
  TH2F* NADA_cool_cell_map;
  std::vector<TH2F*> NADA_cool_cell_map_depth; // individual depth plots

  // Dead cell routine #3:  cell consistently less than pedestal + N sigma
  TH2F* coolcell_below_pedestal;
  TH2F* above_pedestal;
  
  std::vector<TH2F*> coolcell_below_pedestal_depth;
  std::vector<TH2F*> above_pedestal_depth;

// extra diagnostic plots - could be removed?  
  // Should already have these in DigiMonitor, RecHitMonitor
  TH2F* digiCheck;
  TH2F* cellCheck;
  std::vector<TH2F*> digiCheck_depth;
  std::vector<TH2F*> cellCheck_depth;


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
  void htmlADCSubDetOutput(DeadCellHists& hist, int runNo, 
			   std::string htmlDir, 
			   std::string htmlName);
  
  void htmlBelowPedSubDetOutput(DeadCellHists& hist, int runNo, 
				std::string htmlDir, 
				std::string htmlName);
private:

  ofstream htmlFile;

  DeadCellHists hbhists, hehists, hohists, hfhists, hcalhists;
  double errorFrac_; // minimum fraction of events that must be bad to cause error
  int checkNevents_;
};

#endif
