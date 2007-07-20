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
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t id;
  Float_t N;
  Float_t a;
  Float_t b;
  Float_t c;
};

class CSCSaturationAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCSaturationAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_sat 13
#define LAYERS_sat 6
#define STRIPS_sat 80
#define NUMBERPLOTTED_sat 20 
#define NUMMODTEN_sat 500
#define PULSES_sat 25
#define DDU_sat 4

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
  float myCharge[NUMBERPLOTTED_sat],mySatADC[NUMBERPLOTTED_sat],aVar,bVar;
  TH2F adc_vs_charge;
  TH2F adc00_vs_charge;
  TH2F adc01_vs_charge;
  TH2F adc02_vs_charge;
  TH2F adc03_vs_charge;
  TH2F adc04_vs_charge;
  TH2F adc05_vs_charge;
  TH2F adc06_vs_charge;
  TH2F adc07_vs_charge;
  TH2F adc08_vs_charge;
  TH2F adc09_vs_charge;
  TH2F adc10_vs_charge;
  TH2F adc11_vs_charge;
  TH2F adc12_vs_charge;
  
};
