#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"

struct HotCellHists{
  int type;
  int thresholds;

  TH2F* maxCellOccMap;
  TH2F* maxCellEnergyMap;
  TH1F* maxCellEnergy;
  TH1F* maxCellTime;
  TH1F* maxCellID;
  std::vector<TH2F*> threshOccMap;
  std::vector<TH2F*> threshEnergyMap;
  
  std::vector<TH2F*>  threshOccMapDepth1;
  std::vector<TH2F*>  threshEnergyMapDepth1;
  std::vector<TH2F*>  threshOccMapDepth2;
  std::vector<TH2F*>  threshEnergyMapDepth2;
  std::vector<TH2F*>  threshOccMapDepth3;
  std::vector<TH2F*>  threshEnergyMapDepth3;
  std::vector<TH2F*>  threshOccMapDepth4;
  std::vector<TH2F*>  threshEnergyMapDepth4;

  // NADA histograms
  TH2F* nadaOccMap;
  TH2F* nadaEnergyMap;
  TH1F* nadaNumHotCells;
  TH1F* nadaTestCell;
  TH1F* nadaEnergy;
  TH1F* nadaNumNegCells;
  TH2F* nadaNegOccMap;
  TH2F* nadaNegEnergyMap;
};

class HcalHotCellClient : public HcalBaseClient{
  
 public:
  
  /// Constructor
  HcalHotCellClient();
  /// Destructor
   ~HcalHotCellClient();

   void init(const edm::ParameterSet& ps, DaqMonitorBEInterface* dbe, string clientName);    

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
  void htmlSubDetOutput(HotCellHists& hist, int run, std::string htmlDir, std::string htmlName);
  void getHistograms();
  void loadHistograms(TFile* f);

  void resetAllME();

  void createTests();
  void createSubDetTests(HotCellHists& hist);

  // Clear histograms
  void clearHists(HotCellHists& hist);
  void deleteHists(HotCellHists& hist);

  void getSubDetHistograms(HotCellHists& hist);
  void resetSubDetHistograms(HotCellHists& hist);
  void getSubDetHistogramsFromFile(HotCellHists& hist, TFile* infile);

  // Count hot cell thresholds
  void getSubDetThresholds(HotCellHists& hist);
  void drawSubDetThresholds(HotCellHists& hist);
  
private:

  // Can we get threshold information from same .cfi file that HotCellMonitor uses?  
  int thresholds_;

  
  HotCellHists hbhists;
  HotCellHists hehists;
  HotCellHists hohists;
  HotCellHists hfhists;
  HotCellHists hcalhists;

  ofstream htmlFile;
};

#endif
