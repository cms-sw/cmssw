/** 
 * Analyzer for reading bin by bin ADC information
 * author S.Durkin, O.Boeriu 
 *   
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "OnlineDB/CSCCondDB/interface/CSCMap.h"
#include "OnlineDB/CSCCondDB/interface/AutoCorrMat.h"
#include "OnlineDB/CSCCondDB/interface/CSCOnlineDB.h"
#include "CondFormats/CSCObjects/interface/CSCNoiseMatrix.h"
#include "CondFormats/CSCObjects/interface/CSCobject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TDirectory.h"
#include "TCanvas.h"

class TCalibNoiseMatrixEvt {
  public:
  Float_t elem[12];
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t id;
  Int_t flagMatrix;
};

class CSCNoiseMatrixAnalyzer : public edm::EDAnalyzer {

 public:
  explicit CSCNoiseMatrixAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  
#define CHAMBERS_ma 468
#define LAYERS_ma 6
#define STRIPS_ma 80
#define DDU_ma 36

  ~CSCNoiseMatrixAnalyzer();
  
 private:
 // variables persistent across events should be declared here.
 std::vector<int> adc;
 std::string chamber_id;
 int eventNumber,evt,counterzero,strip,misMatch,NChambers,Nddu;
 int i_chamber,i_layer,reportedChambers,fff,ret_code,length,chamber_num,sector,record;
 int dmbID[CHAMBERS_ma],crateID[CHAMBERS_ma],size[CHAMBERS_ma];
 int lines,flagMatrix;
 std::ifstream filein;
 std::string PSet,name;
 bool debug;
 float *tmp, corrmat[12];
 float newMatrix1[480];
 float newMatrix2[480];
 float newMatrix3[480];
 float newMatrix4[480];
 float newMatrix5[480];
 float newMatrix6[480];
 float newMatrix7[480];
 float newMatrix8[480];
 float newMatrix9[480];
 float newMatrix10[480];
 float newMatrix11[480];
 float newMatrix12[480];

 Chamber_AutoCorrMat cam[CHAMBERS_ma];
};
