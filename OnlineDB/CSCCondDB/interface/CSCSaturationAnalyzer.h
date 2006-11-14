/** 
 * Analyzer for reading gains saturation information
 * author S. Durkin, O.Boeriu 18/03/06 
 *   
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
#include "TH2F.h"

class TCalibSaturationEvt {
  public:
  Float_t slope;
  Float_t intercept;
  Float_t chi2;
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t id;
};

class CSCSaturationAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCSaturationAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_sat 9
#define LAYERS_sat 6
#define STRIPS_sat 80
#define NUMBERPLOTTED_sat 24 
#define NUMMODTEN_sat 480
#define DDU_sat 2

  ~CSCSaturationAnalyzer();
  float (*charge_ptr)[NUMBERPLOTTED_sat];
  float (*adc_ptr)[NUMMODTEN_sat];
 
 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,chamber_num,sector,i_chamber,i_layer,reportedChambers;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu,record;
  time_t rawtime;
  int dmbID[CHAMBERS_sat],crateID[CHAMBERS_sat],size[CHAMBERS_sat]; 
  float adcMax[DDU_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  float adcMean_max[DDU_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  float maxmodten[NUMMODTEN_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  int lines;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  float myCharge[24],mySatADC[24],aVar,bVar;
  TH2F gain_vs_charge;
  TH2F gain01_vs_charge;
  TH2F gain02_vs_charge;
  TH2F gain03_vs_charge;
  TH2F gain04_vs_charge;
  TH2F gain05_vs_charge;
  TH2F gain11_vs_charge;
  TH2F gain12_vs_charge;
  TH2F gain13_vs_charge;
  TH2F gain14_vs_charge;
  TH2F gain15_vs_charge;
  TH2F gain21_vs_charge;
  TH2F gain22_vs_charge;
  TH2F gain23_vs_charge;
  TH2F gain24_vs_charge;
  TH2F gain25_vs_charge;
  TH2F gain31_vs_charge;
  TH2F gain32_vs_charge;
  TH2F gain33_vs_charge;
  TH2F gain34_vs_charge;
  TH2F gain35_vs_charge;
  TH2F gain41_vs_charge;
  TH2F gain42_vs_charge;
  TH2F gain43_vs_charge;
  TH2F gain44_vs_charge;
  TH2F gain45_vs_charge;
  TH2F gain51_vs_charge;
  TH2F gain52_vs_charge;
  TH2F gain53_vs_charge;
  TH2F gain54_vs_charge;
  TH2F gain55_vs_charge;
  TH2F gain61_vs_charge;
  TH2F gain62_vs_charge;
  TH2F gain63_vs_charge;
  TH2F gain64_vs_charge;
  TH2F gain65_vs_charge;
  TH2F gain71_vs_charge;
  TH2F gain72_vs_charge;
  TH2F gain73_vs_charge;
  TH2F gain74_vs_charge;
  TH2F gain75_vs_charge;
  TH2F gain81_vs_charge;
  TH2F gain82_vs_charge;
  TH2F gain83_vs_charge;
  TH2F gain84_vs_charge;
  TH2F gain85_vs_charge;

};
