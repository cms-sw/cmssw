#ifndef RecoLocalMuon_CSCValHists_H
#define RecoLocalMuon_CSCValHists_H


/** \class CSCValHists
 *
 *  Manages Histograms for CSCValidation
 *
 */


// system include files
#include <memory>
#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>

#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TFile.h"
#include "TTree.h"


class CSCValHists{

  public:

  // constructor
  CSCValHists();


  // destructor
  ~CSCValHists();

  // book histograms
  void bookHists();

  // write histograms
  void writeHists(TFile* theFile, bool isSimulation);

  // setup trees
  void setupTrees();

  // fill hists
  void fillWireHistos(int wire, int TB, int codeN, int codeB,
                      int en, int st, int ri, int ch, int la);

  void fillStripHistos(int strip, int codeN, int codeB,
                       int en, int st, int ri, int ch, int la);

  void fillNoiseHistos(float ADC, int  globalStrip, int kStation, int kRing);

  void fillCalibHistos(float gain, float xl, float xr, float xil, float xir, float pedp,
                       float pedr, float n33, float n34, float n35, float n44, float n45,
                       float n46, float n55, float n56, float n57, float n66, float n67,
                       float n77, int bin);

  void fillRechitHistos(int codeN, int codeB, float x, float y, float gx, float gy,
                        float sQ, float rQ, float time, float simres,
                        int en, int st, int ri, int ch, int la);

  void fillRechitTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch, int la);

  void fillSegmentHistos(int codeN, int codeB, int nhits, float theta, float gx, float gy,
                         float resid, float chi2p, float gTheta, float gPhi,
                         int en, int st, int ri, int ch);

  void fillSegmentTree(float x, float y, float gx, float gy, int en, int st, int ri, int ch);

  void fillEventHistos(int nWire, int nStrip, int nrH,int nSeg);

  // these functions handle Stoyan's efficiency code
  void fillEfficiencyHistos(int bin, int flag);
  void getEfficiency(float bin, float Norm, std::vector<float> &eff);
  void histoEfficiency(TH1F *readHisto, TH1F *writeHisto);

  protected:

  private:

  // my histograms
  TH1F *hCalibGainsS;
  TH1F *hCalibXtalkSL;
  TH1F *hCalibXtalkSR;
  TH1F *hCalibXtalkIL;
  TH1F *hCalibXtalkIR;
  TH1F *hCalibPedsP;
  TH1F *hCalibPedsR;
  TH1F *hCalibNoise33;
  TH1F *hCalibNoise34;
  TH1F *hCalibNoise35;
  TH1F *hCalibNoise44;
  TH1F *hCalibNoise45;
  TH1F *hCalibNoise46;
  TH1F *hCalibNoise55;
  TH1F *hCalibNoise56;
  TH1F *hCalibNoise57;
  TH1F *hCalibNoise66;
  TH1F *hCalibNoise67;
  TH1F *hCalibNoise77;


  TH1F *hWireAll;
  TH1F *hWireTBinAll;
  TH1F *hWirenGroupsTotal;
  TH1F *hWireCodeBroad;
  TH1F *hWireCodeNarrow1;
  TH1F *hWireCodeNarrow2;
  TH1F *hWireCodeNarrow3;
  TH1F *hWireCodeNarrow4;
  TH1F *hWireLayer11b;
  TH1F *hWireLayer12;
  TH1F *hWireLayer13;
  TH1F *hWireLayer11a;
  TH1F *hWireLayer21;
  TH1F *hWireLayer22;
  TH1F *hWireLayer31;
  TH1F *hWireLayer32;
  TH1F *hWireLayer41;
  TH1F *hWireLayer42;
  TH1F *hWireWire11b;
  TH1F *hWireWire12;
  TH1F *hWireWire13;
  TH1F *hWireWire11a;
  TH1F *hWireWire21;
  TH1F *hWireWire22;
  TH1F *hWireWire31;
  TH1F *hWireWire32;
  TH1F *hWireWire41;
  TH1F *hWireWire42;

  TH1F *hStripAll;
  TH1F *hStripNFired;
  TH1F *hStripCodeBroad;
  TH1F *hStripCodeNarrow1;
  TH1F *hStripCodeNarrow2;
  TH1F *hStripCodeNarrow3;
  TH1F *hStripCodeNarrow4;
  TH1F *hStripLayer11b;
  TH1F *hStripLayer12;
  TH1F *hStripLayer13;
  TH1F *hStripLayer11a;
  TH1F *hStripLayer21;
  TH1F *hStripLayer22;
  TH1F *hStripLayer31;
  TH1F *hStripLayer32;
  TH1F *hStripLayer41;
  TH1F *hStripLayer42;
  TH1F *hStripStrip11b;
  TH1F *hStripStrip12;
  TH1F *hStripStrip13;
  TH1F *hStripStrip11a;
  TH1F *hStripStrip21;
  TH1F *hStripStrip22;
  TH1F *hStripStrip31;
  TH1F *hStripStrip32;
  TH1F *hStripStrip41;
  TH1F *hStripStrip42;

  TH1F *hStripPed; 
  TH1F *hStripPedME11; 
  TH1F *hStripPedME12; 
  TH1F *hStripPedME13; 
  TH1F *hStripPedME14; 
  TH1F *hStripPedME21; 
  TH1F *hStripPedME22; 
  TH1F *hStripPedME31; 
  TH1F *hStripPedME32; 
  TH1F *hStripPedME41; 
  TH1F *hStripPedME42; 
  TH2F *hPedvsStrip;

  TH1F *hRHCodeBroad;
  TH1F *hRHCodeNarrow1;
  TH1F *hRHCodeNarrow2;
  TH1F *hRHCodeNarrow3;
  TH1F *hRHCodeNarrow4;
  TH1F *hRHLayer11b;
  TH1F *hRHLayer12;
  TH1F *hRHLayer13;
  TH1F *hRHLayer11a;
  TH1F *hRHLayer21;
  TH1F *hRHLayer22;
  TH1F *hRHLayer31;
  TH1F *hRHLayer32;
  TH1F *hRHLayer41;
  TH1F *hRHLayer42;
  TH1F *hRHX11b;
  TH1F *hRHX12;
  TH1F *hRHX13;
  TH1F *hRHX11a;
  TH1F *hRHX21;
  TH1F *hRHX22;
  TH1F *hRHX31;
  TH1F *hRHX32;
  TH1F *hRHX41;
  TH1F *hRHX42;
  TH1F *hRHY11b;
  TH1F *hRHY12;
  TH1F *hRHY13;
  TH1F *hRHY11a;
  TH1F *hRHY21;
  TH1F *hRHY22;
  TH1F *hRHY31;
  TH1F *hRHY32;
  TH1F *hRHY41;
  TH1F *hRHY42;
  TH2F *hRHGlobal1;
  TH2F *hRHGlobal2;
  TH2F *hRHGlobal3;
  TH2F *hRHGlobal4;
  TH1F *hRHEff;
  TH1F *hRHResid11b;
  TH1F *hRHResid12;
  TH1F *hRHResid13;
  TH1F *hRHResid11a;
  TH1F *hRHResid21;
  TH1F *hRHResid22;
  TH1F *hRHResid31;
  TH1F *hRHResid32;
  TH1F *hRHResid41;
  TH1F *hRHResid42;
  TH1F *hSResid11b;
  TH1F *hSResid12;
  TH1F *hSResid13;
  TH1F *hSResid11a;
  TH1F *hSResid21;
  TH1F *hSResid22;
  TH1F *hSResid31;
  TH1F *hSResid32;
  TH1F *hSResid41;
  TH1F *hSResid42;
  TH1F *hRHSumQ11b;
  TH1F *hRHSumQ12;
  TH1F *hRHSumQ13;
  TH1F *hRHSumQ11a;
  TH1F *hRHSumQ21;
  TH1F *hRHSumQ22;
  TH1F *hRHSumQ31;
  TH1F *hRHSumQ32;
  TH1F *hRHSumQ41;
  TH1F *hRHSumQ42;
  TH1F *hRHRatioQ11b;
  TH1F *hRHRatioQ12;
  TH1F *hRHRatioQ13;
  TH1F *hRHRatioQ11a;
  TH1F *hRHRatioQ21;
  TH1F *hRHRatioQ22;
  TH1F *hRHRatioQ31;
  TH1F *hRHRatioQ32;
  TH1F *hRHRatioQ41;
  TH1F *hRHRatioQ42;
  TH1F *hRHTiming11a;
  TH1F *hRHTiming12;
  TH1F *hRHTiming13;
  TH1F *hRHTiming11b;
  TH1F *hRHTiming21;
  TH1F *hRHTiming22;
  TH1F *hRHTiming31;
  TH1F *hRHTiming32;
  TH1F *hRHTiming41;
  TH1F *hRHTiming42;
  TH1F *hRHnrechits;


  TH1F *hSCodeBroad;
  TH1F *hSCodeNarrow1;
  TH1F *hSCodeNarrow2;
  TH1F *hSCodeNarrow3;
  TH1F *hSCodeNarrow4;
  TH1F *hSnHits11b;
  TH1F *hSnHits12;
  TH1F *hSnHits13;
  TH1F *hSnHits11a;
  TH1F *hSnHits21;
  TH1F *hSnHits22;
  TH1F *hSnHits31;
  TH1F *hSnHits32;
  TH1F *hSnHits41;
  TH1F *hSnHits42;
  TH1F *hSTheta11b;
  TH1F *hSTheta12;
  TH1F *hSTheta13;
  TH1F *hSTheta11a;
  TH1F *hSTheta21;
  TH1F *hSTheta22;
  TH1F *hSTheta31;
  TH1F *hSTheta32;
  TH1F *hSTheta41;
  TH1F *hSTheta42;
  TH2F *hSGlobal1;
  TH2F *hSGlobal2;
  TH2F *hSGlobal3;
  TH2F *hSGlobal4;
  TH1F *hSnhits;
  TH1F *hSEff;
  TH1F *hSChiSqProb;
  TH1F *hSGlobalTheta;
  TH1F *hSGlobalPhi;
  TH1F *hSnSegments;

  // tmp histos for Efficiency
  TH1F *hSSTE;
  TH1F *hRHSTE;


  //
  //
  // A struct for creating a Tree/Branch of position info
  struct posRecord {
    int endcap;
    int station;
    int ring;
    int chamber;
    int layer;
    float localx;
    float localy;
    float globalx;
    float globaly;
  } rHpos, segpos;

  //
  //
  // The root tree
  TTree *rHTree;
  TTree *segTree;



};

#endif   
