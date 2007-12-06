/** 
 * Analyzer for calculating CFEB SCA pedestal.
 * author O.Boeriu 
 * runs over multiple DDUs
 * takes variable size chambers & layers 
 * produces histograms & ntuple 
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/CSCMap1.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

class TCalibSCAEvt {
  public:
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t ddu;
  Float_t scaMeanVal;
  Int_t id;
  Int_t scanumber;
};

class CSCscaAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCscaAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
   
#define CHAMBERS_sca 9
#define LAYERS_sca 6
#define STRIPS_sca 80
#define TIMEBINS_sca 8
#define DDU_sca 2
#define Number_sca 96
#define TOTALSTRIPS_sca 480
#define TOTALEVENTS_sca 10000

  ~CSCscaAnalyzer();

  
 private:
  
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int chamber,layer,reportedChambers,chamber_num,sector,run,NChambers,first_strip_index,strips_per_layer,chamber_index ;
  int dmbID[CHAMBERS_sca],crateID[CHAMBERS_sca],size[CHAMBERS_sca];
  int value_adc[DDU_sca][CHAMBERS_sca][LAYERS_sca][STRIPS_sca][Number_sca];
  int scaNr[DDU_sca][CHAMBERS_sca][LAYERS_sca][STRIPS_sca];
  float value_adc_mean[DDU_sca][CHAMBERS_sca][LAYERS_sca][STRIPS_sca][Number_sca];
  int count_adc_mean[DDU_sca][CHAMBERS_sca][LAYERS_sca][STRIPS_sca][Number_sca];
  float div[DDU_sca][CHAMBERS_sca][LAYERS_sca][STRIPS_sca][Number_sca];
  std::vector<int> adc;
  std::string chamber_id;
  int lines,myIndex;
  std::ifstream filein;
  std::string PSet,name,chamber_type;
  bool debug;
  int flag,my_scaValue,counterzero,maxStrip,counter;
  unsigned int maxDDU;
  float pedMean,my_scaValueMean;
  int scaBlock,trigTime,lctPhase,power,cap,scaNumber,myNcham;

  //root ntuple
  TCalibSCAEvt calib_evt;
  TBranch *calibevt;
  TTree *calibtree;
  TFile *calibfile;
};
