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

class TCalibOldCrossTalkEvt {
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
  Float_t MaxPed[13];
  Float_t MaxRMS[13];
  Float_t MaxPeakTime[13];
  Float_t MinPeakTime[13];
  Float_t MaxPeakADC[13];
  Float_t MinPeakADC[13];
};

class CSCOldCrossTalkAnalyzer : public edm::EDAnalyzer {
 public:
  explicit CSCOldCrossTalkAnalyzer(edm::ParameterSet const& conf);
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);
  ~CSCOldCrossTalkAnalyzer();

#define CHAMBERS_oldxt 13
#define LAYERS_oldxt 6
#define STRIPS_oldxt 80
#define TIMEBINS_oldxt 8
#define DDU_oldxt 4
#define TOTALSTRIPS_oldxt 480
#define PULSES_oldxt 20

  //TH2F *g1=new TH2F("g1","Baseline RMS",100,0,80,100,0,200);
 private:
  int eventNumber,evt,strip,misMatch,fff,ret_code,length,Nddu,myevt,first_strip_index,strips_per_layer,chamber_index;
  int chamber,layer,reportedChambers,chamber_num,sector,record,NChambers ;
  int dmbID[CHAMBERS_oldxt],crateID[CHAMBERS_oldxt],size[CHAMBERS_oldxt];
  std::vector<int> adc;
  std::string chamber_id;
  int thebins[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt][TIMEBINS_oldxt*PULSES_oldxt];
  int theadccountsc[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt][TIMEBINS_oldxt*PULSES_oldxt];
  int theadccountsl[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt][TIMEBINS_oldxt*PULSES_oldxt];
  int theadccountsr[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt][TIMEBINS_oldxt*PULSES_oldxt];
  float pedMean,pedMean1,time,max1,max2,min1, aPeak,sumFive,maxRMS,maxPed,adcMAX;
  float maxPeakTime, minPeakTime, maxPeakADC, minPeakADC;
  float meanPedestal,meanPeak,meanPeakSquare,meanPedestalSquare,theRMS;
  float thePeak,thePeakMin, thePeakRMS,theSumFive,thePedestal,theRSquare;
  float thetime[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt][TIMEBINS_oldxt*PULSES_oldxt];
  float xtalk_intercept_left[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float xtalk_intercept_right[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float xtalk_slope_left[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float xtalk_slope_right[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float xtalk_chi2_left[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float xtalk_chi2_right[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float myPeakTime[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float myMeanPeakTime[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float array_meanPeakTime[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayOfPed[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayOfPedSquare[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayPed[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayPeak[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayPeakMin[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayOfPeak[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arrayOfPeakSquare[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float arraySumFive[DDU_oldxt][CHAMBERS_oldxt][LAYERS_oldxt][STRIPS_oldxt];
  float myTime[TIMEBINS_oldxt];
  float myADC[TIMEBINS_oldxt];
  int myTbin[TIMEBINS_oldxt];
  float newPed[TOTALSTRIPS_oldxt];
  float newRMS[TOTALSTRIPS_oldxt];
  float newPeakRMS[TOTALSTRIPS_oldxt];
  float newPeak[TOTALSTRIPS_oldxt];
  float newPeakMin[TOTALSTRIPS_oldxt];
  float newSumFive[TOTALSTRIPS_oldxt];
  float new_xtalk_intercept_right[TOTALSTRIPS_oldxt];
  float new_xtalk_intercept_left[TOTALSTRIPS_oldxt];
  float new_xtalk_slope_right[TOTALSTRIPS_oldxt];
  float new_xtalk_slope_left[TOTALSTRIPS_oldxt];
  float new_rchi2[TOTALSTRIPS_oldxt];
  float new_lchi2[TOTALSTRIPS_oldxt];
  float newPeakTime[TOTALSTRIPS_oldxt];
  float newMeanPeakTime[TOTALSTRIPS_oldxt];
  int lines,myIndex;
  std::ifstream filein;
  std::string PSet,name;
  bool debug;
  int flagRMS,flagNoise;

  //root ntuple
  TCalibOldCrossTalkEvt calib_evt;
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

