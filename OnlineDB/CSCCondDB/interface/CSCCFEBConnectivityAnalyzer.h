/** 
 * Analyzer for calculating CFEB ADC counts for connectivity.
 * author S.Durkin, O.Boeriu 
 * runs over multiple DDUs
 * takes variable size chambers & layers 
 * produces histograms & ntuple 
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/CSCMap.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

class TCalibCFEBConnectEvt {
  public:
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t ddu;
  Float_t adcMax;
  Float_t adcMin;
  Float_t diff;
  Float_t RMS;
};

class CSCCFEBConnectivityAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCFEBConnectivityAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
   
#define CHAMBERS_con 18
#define LAYERS_con 6
#define STRIPS_con 80
#define TIMEBINS_con 8
#define DDU_con 2
#define TOTALSTRIPS_con 480
#define TOTALEVENTS_con 320

  ~CSCCFEBConnectivityAnalyzer();

 private:
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int chamber,layer,reportedChambers,chamber_num,sector,record,NChambers,first_strip_index,strips_per_layer, chamber_index;
  int dmbID[CHAMBERS_con],crateID[CHAMBERS_con],size[CHAMBERS_con];
  float adcMin[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float adcMax[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float adcMean_max[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float adcMean_min[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  float diff[DDU_con][CHAMBERS_con][LAYERS_con][STRIPS_con];
  std::vector<int> adc;
  std::string chamber_id;
  int lines;
  float my_diff,my_diffSquare,theRMS;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  int flag;

 //root ntuple
  TCalibCFEBConnectEvt calib_evt;
  TBranch *calibevt;
  TTree *calibtree;
  TFile *calibfile;
};


