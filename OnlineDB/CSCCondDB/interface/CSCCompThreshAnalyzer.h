/** 
 * Analyzer for reading CFEB comparator information
 * author O.Boeriu 
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

class TCalibComparatorEvt {
  public:
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t id;
};

class CSCCompThreshAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCompThreshAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_ct 9
#define LAYERS_ct 6
#define STRIPS_ct 80
#define TOTALSTRIPS_ct 480
#define DDU_ct 2
#define NUMMOD_ct 875
#define NUMBERPLOTTED_ct 35

  ~CSCCompThreshAnalyzer();
  
 private:
  
  std::string chamber_id;
  int eventNumber,evt,event,pedSum, strip, misMatch,fff,ret_code,NChambers,Nddu;
  int length,i_chamber,i_layer,reportedChambers,chamber_num,sector,first_strip_index,strips_per_layer,chamber_index; 
  int timebin,mycompstrip,comparator,compstrip,compadc;
  int dmbID[CHAMBERS_ct],crateID[CHAMBERS_ct],size[CHAMBERS_ct]; 
  float meanThresh;
  float theMeanThresh[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	arrayMeanThresh[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	mean[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float	meanTot[CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  float meanmod[NUMMOD_ct][CHAMBERS_ct][LAYERS_ct][STRIPS_ct];
  int lines;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  float myCharge[35],myCompProb[35];
  TH2F adc_vs_charge;
};
