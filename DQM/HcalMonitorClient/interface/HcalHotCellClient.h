#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"

struct HotCellHists{
  int type;
  int thresholds;
  TH2F* OCC_MAP_GEO_Max;
  TH2F* EN_MAP_GEO_Max;
  TH1F* MAX_E;
  TH1F* MAX_T;
  TH1F* MAX_ID;
  std::vector<TH2F*> OCCmap;
  std::vector<TH2F*> ENERGYmap;
  // NADA histograms
  TH2F* NADA_OCC_MAP;
  TH2F* NADA_EN_MAP;
  TH1F* NADA_NumHotCells;
  TH1F* NADA_testcell;
  TH1F* NADA_Energy;
  TH1F* NADA_NumNegCells;
  TH2F* NADA_NEG_OCC_MAP;
  TH2F* NADA_NEG_EN_MAP;
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

private:

  // Can we get threshold information from same .cfi file that HotCellMonitor uses?  
  int thresholds_;

  TH2F* gl_geo_[4];
  TH2F* gl_en_[4];

  TH2F* occ_geo_[4][2];
  TH2F* occ_en_[4][2];
  TH1F* max_en_[4];
  TH1F* max_t_[4];
  
  HotCellHists hbhists;
  HotCellHists hehists;
  HotCellHists hohists;
  HotCellHists hfhists;
  HotCellHists hcalhists;

  ofstream htmlFile;
};

#endif
