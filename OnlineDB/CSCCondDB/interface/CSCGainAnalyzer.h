/** 
 * Analyzer for reading gains information
 * author S. Durkin, O.Boeriu
 *   
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>
#include "OnlineDB/CSCCondDB/interface/CSCMap.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCGains.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH2F.h"

class TCalibGainEvt {
 public:
  Float_t slope;
  Float_t intercept;
  Float_t chi2;
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t id;
  Int_t flagGain;
  Int_t flagIntercept;
};

class CSCGainAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCGainAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_ga 13
#define LAYERS_ga 6
#define STRIPS_ga 80
#define NUMBERPLOTTED_ga 20
#define PULSES_ga 25
#define FITNUMBERS_ga 10 
#define NUMMODTEN_ga 500
#define DDU_ga 4

  ~CSCGainAnalyzer();
  
 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,counterzero,chamber_num,sector,i_chamber,i_layer,reportedChambers;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu,record;
  time_t rawtime;
  int dmbID[CHAMBERS_ga],crateID[CHAMBERS_ga],size[CHAMBERS_ga]; 
  float gainSlope,gainIntercept;
  float adcMax[DDU_ga][CHAMBERS_ga][LAYERS_ga][STRIPS_ga];
  float adcMean_max[DDU_ga][CHAMBERS_ga][LAYERS_ga][STRIPS_ga];
  float maxmodten[NUMMODTEN_ga][CHAMBERS_ga][LAYERS_ga][STRIPS_ga];
  float newGain[480];
  float newIntercept[480];
  float newChi2[480];
  float myCharge[20];
  int lines,flagGain,flagIntercept;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  TH2F adcCharge_ch0;
  TH2F adcCharge_ch1;
  TH2F adcCharge_ch2;
  TH2F adcCharge_ch3;
  TH2F adcCharge_ch4;
  TH2F adcCharge_ch5;
  TH2F adcCharge_ch6;
  TH2F adcCharge_ch7;
  TH2F adcCharge_ch8;
  TH2F adcCharge_ch9;
  TH2F adcCharge_ch10;
  TH2F adcCharge_ch11;
  TH2F adcCharge_ch12;
};
