/** 
 * Analyzer for reading gains information
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
#define NUMBERPLOTTED_sat 25 
#define NUMMODTEN_sat 500
#define DDU_sat 2

  ~CSCSaturationAnalyzer();
  
 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,chamber_num,sector,i_chamber,i_layer,reportedChambers;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu,record;
  time_t rawtime;
  int dmbID[CHAMBERS_sat],crateID[CHAMBERS_sat],size[CHAMBERS_sat]; 
  float gainSlope,gainIntercept;
  float adcMax[DDU_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  float adcMean_max[DDU_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  float maxmodten[NUMMODTEN_sat][CHAMBERS_sat][LAYERS_sat][STRIPS_sat];
  float newGain[480];
  float newIntercept[480];
  float newChi2[480];
  int lines;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  float myCharge[25];
  TH2F gain_vs_charge;
};
