/** 
 * Analyzer for calculating CFEB cross-talk & pedestal.
 * author S.Durkin, O.Boeriu, A. Roe 
 * runs over multiple DDUs
 * takes variable size chambers & layers 
 * produces histograms & ntuple 
 */

#include <iostream>
#include <time.h>
#include <sys/stat.h>	
#include <unistd.h>
#include <fstream>

#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"

class TCalibCrossTalkEvt {
  public:
  Float_t xtalk_slope_left;
  Float_t xtalk_slope_right;
  Float_t xtalk_int_left;
  Float_t xtalk_int_right;
  Float_t xtalk_chi2_left;
  Float_t xtalk_chi2_right;
  Float_t peakTime;
  Int_t strip;
  Int_t layer;
  Int_t cham;
  Int_t ddu;
  Float_t pedMean;
  Float_t pedRMS;
  Float_t peakRMS;
  Float_t maxADC;
  Float_t sum;
  Int_t id;
  Int_t flagRMS;
  Int_t flagNoise;
  Float_t MaxPed[9];
  Float_t MaxRMS[9];
  Float_t MaxPeakTime[9];
  Float_t MinPeakTime[9];
  Float_t MaxPeakADC[9];
  Float_t MinPeakADC[9];
};

class CSCCrossTalkAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCCrossTalkAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  ~CSCCrossTalkAnalyzer();

#define CHAMBERS_xt 13
#define LAYERS_xt 6
#define STRIPS_xt 80
#define TIMEBINS_xt 8
#define DDU_xt 2
#define TOTALSTRIPS_xt 480
#define TOTALEVENTS_xt 320
#define PULSES 25 
#define NUMMODTEN_xt 10 

 private:
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt;
  int chamber,layer,reportedChambers,chamber_num,sector,record,NChambers ;
  int dmbID[CHAMBERS_xt],crateID[CHAMBERS_xt],size[CHAMBERS_xt];
  std::vector<int> adc;
  std::string chamber_id;
  int thebins[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*PULSES];
  int theadccountsc[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*PULSES];
  int theadccountsl[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*PULSES];
  int theadccountsr[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*PULSES];
  float pedMean,pedMean1,time,max1,max2,min1, aPeak,sumFive,maxRMS,maxPed;
  float maxPeakTime, minPeakTime, maxPeakADC, minPeakADC;
  float meanPedestal,meanPeak,meanPeakSquare,meanPedestalSquare,theRMS;
  float thePeak,thePeakMin, thePeakRMS,theSumFive,thePedestal,theRSquare;
  float thetime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt][TIMEBINS_xt*PULSES];
  float xtalk_intercept_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_intercept_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_slope_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_slope_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_chi2_left[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float xtalk_chi2_right[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myMeanPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float array_meanPeakTime[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPed[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPedSquare[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayPed[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayPeak[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayPeakMin[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPeak[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arrayOfPeakSquare[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float arraySumFive[DDU_xt][CHAMBERS_xt][LAYERS_xt][STRIPS_xt];
  float myTime[TIMEBINS_xt];
  float myADC[TIMEBINS_xt];
  int myTbin[TIMEBINS_xt];
  float newPed[TOTALSTRIPS_xt];
  float newRMS[TOTALSTRIPS_xt];
  float newPeakRMS[TOTALSTRIPS_xt];
  float newPeak[TOTALSTRIPS_xt];
  float newPeakMin[TOTALSTRIPS_xt];
  float newSumFive[TOTALSTRIPS_xt];
  float new_xtalk_intercept_right[TOTALSTRIPS_xt];
  float new_xtalk_intercept_left[TOTALSTRIPS_xt];
  float new_xtalk_slope_right[TOTALSTRIPS_xt];
  float new_xtalk_slope_left[TOTALSTRIPS_xt];
  float new_rchi2[TOTALSTRIPS_xt];
  float new_lchi2[TOTALSTRIPS_xt];
  float newPeakTime[TOTALSTRIPS_xt];
  float newMeanPeakTime[TOTALSTRIPS_xt];
  int lines;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  int flagRMS,flagNoise;

  //root ntuple
  TCalibCrossTalkEvt calib_evt;
  TBranch *calibevt;
  TTree *calibtree;
  TFile *calibfile;
  ofstream* outfile;
  TH1F xtime;
  TH1F ped_mean_all; 
  TH1F ped_RMS_all;
  TH1F maxADC;
  TH2F pulseshape_ch0;
  TH2F pulseshape_ch1;
  TH2F pulseshape_ch2;
  TH2F pulseshape_ch3;
  TH2F pulseshape_ch4;
  TH2F pulseshape_ch5;
  TH2F pulseshape_ch6;
  TH2F pulseshape_ch7;
  TH2F pulseshape_ch8;
  TH2F pulseshape_ch9;
  TH2F pulseshape_ch10;
  TH2F pulseshape_ch11;
  TH2F pulseshape_ch12;
};

