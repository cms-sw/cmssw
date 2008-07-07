#ifndef HcalHotCellClient_H
#define HcalHotCellClient_H

#include "DQM/HcalMonitorClient/interface/HcalBaseClient.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <DQM/HcalMonitorClient/interface/HcalClientUtils.h>
#include <DQM/HcalMonitorClient/interface/HcalHistoUtils.h>

struct HotCellHists{
  int type;
  int Nthresholds;

  TH2F* problemHotCells;
 
  TH2F* problemHotCells_DEPTH[4];

  TH2F* maxCellOccMap;
  TH2F* maxCellEnergyMap;
  TH1F* maxCellEnergy;
  TH1F* maxCellTime;
  TH1F* maxCellID;
  std::vector<std::string> thresholds;

  // replace vectors with fixed-size arrays
  TH2F* threshOccMap[5];
  TH2F* threshEnergyMap[5];
  

  TH2F*  threshOccMapDepth1[5];
  TH2F*  threshEnergyMapDepth1[5];
  TH2F*  threshOccMapDepth2[5];
  TH2F*  threshEnergyMapDepth2[5];
  TH2F*  threshOccMapDepth3[5];
  TH2F*  threshEnergyMapDepth3[5];
  TH2F*  threshOccMapDepth4[5];
  TH2F*  threshEnergyMapDepth4[5];

   // Digi Plots
  TH2F* abovePedSigma;
  //std::vector <std::vector<TH2F*> > digiPedestalPlots_depth;


  // NADA histograms
  TH2F* nadaOccMap;
  TH2F* nadaEnergyMap;
  TH1F* nadaNumHotCells;
  TH1F* nadaEnergy;
  TH1F* nadaNumNegCells;
  TH2F* nadaNegOccMap;
  TH2F* nadaNegEnergyMap;

  TH2F* nadaOccMapDepth[4];
  TH2F* nadaNegOccMapDepth[4];
  TH2F* nadaEnergyMapDepth[4];
  TH2F* nadaNegEnergyMapDepth[4];
  

};

class HcalHotCellClient : public HcalBaseClient{
  
 public:
  
  /// Constructor
  HcalHotCellClient();
  /// Destructor
   ~HcalHotCellClient();

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
  void htmlSubDetOutput(HotCellHists& hist, int run, std::string htmlDir, std::string htmlName);
  void drawThresholdPlots(HotCellHists& hist, int run, string htmlDir);
  void drawDigiPlots(HotCellHists& hist, int run, string htmlDir);
  void drawNADAPlots(HotCellHists& hist, int run, string htmlDir);
  void drawProblemPlots(HotCellHists& hist, int run, string htmlDir); 

  void getHistograms();
  void loadHistograms(TFile* f);

  void resetAllME();

  void createTests();
  void createSubDetTests(HotCellHists& hist);

  // Clear histograms
  void clearHists(HotCellHists& hist);
  void deleteHists(HotCellHists& hist);

  void combineSubDetHistograms(HotCellHists& hcal, HotCellHists& hb, HotCellHists& he, HotCellHists& ho, HotCellHists& hf);
  void getSubDetHistograms(HotCellHists& hist);
  void resetSubDetHistograms(HotCellHists& hist);
  void getSubDetHistogramsFromFile(HotCellHists& hist, TFile* infile);

  // Count hot cell thresholds
  void getSubDetThresholds(HotCellHists& hist);
  void drawSubDetThresholds(HotCellHists& hist);


  
private:

  // Can we get threshold information from same .cfi file that HotCellMonitor uses?  
  int thresholds_;
  
  float errorFrac_;
  
  HotCellHists hbhists;
  HotCellHists hehists;
  HotCellHists hohists;
  HotCellHists hfhists;
  HotCellHists hcalhists;

  ofstream htmlFile;
};

#endif
