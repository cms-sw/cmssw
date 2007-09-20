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

class TCalibOldGainEvt {
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

class CSCOldGainAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCOldGainAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_oldga 10
#define LAYERS_oldga 6
#define STRIPS_oldga 80
#define NUMBERPLOTTED_oldga 10 
#define NUMMODTEN_oldga 200
#define DDU_oldga 4

  ~CSCOldGainAnalyzer();
  
 private:
  std::vector<int> newadc; 
  std::string chamber_id;
  int eventNumber,evt,chamber_num,sector,i_chamber,i_layer,reportedChambers,first_strip_index,strips_per_layer,chamber_index;
  int fff,ret_code,length,strip,misMatch,NChambers,Nddu,record;
  time_t rawtime;
  int dmbID[CHAMBERS_oldga],crateID[CHAMBERS_oldga],size[CHAMBERS_oldga]; 
  float gainSlope,gainIntercept;
  float adcMax[DDU_oldga][CHAMBERS_oldga][LAYERS_oldga][STRIPS_oldga];
  float adcMean_max[DDU_oldga][CHAMBERS_oldga][LAYERS_oldga][STRIPS_oldga];
  float maxmodten[NUMMODTEN_oldga][CHAMBERS_oldga][LAYERS_oldga][STRIPS_oldga];
  float newGain[480];
  float newIntercept[480];
  float newChi2[480];
  float myCharge[10];
  int lines,flagGain,flagIntercept,counter,myIndex;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  TH2F adcCharge;
};
