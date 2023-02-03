// How to run:
//root -b -q -l RemoteMonitoringMAP.C+
//root -b -q -l 'RemoteMonitoringMAP.C+("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214513.root","/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214512.root")'
//root -b -q -l 'RemoteMonitoringMAP.C+(" /afs/cern.ch/work/d/dtlisov/private/Monitoring/histos/LED_211659.root","/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214512.root")'
//
//
//
#include "LogEleMapdb.h"

#include <iostream>
#include <fstream>

#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"
#include "TROOT.h"
#include <TMath.h>
#include "TStyle.h"
#include "TSystem.h"
#include "TLegend.h"
#include "TText.h"
#include "TAxis.h"
#include "TFile.h"
#include "TLine.h"
#include "TGraph.h"
#include <TPaveText.h>
//#####

#include <TChain.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TPaveStats.h>
#include <vector>
#include <string>
#include <iomanip>
//#####
#include <TClass.h>

//
// https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT
// https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/GlobalRMT

using namespace std;
//inline void HERE(const char *msg) { std::cout << msg << std::endl; }

int main(int argc, char *argv[]) {
  std::string dirnm = "Analyzer";
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(1);

  gStyle->SetStatX(0.91);
  gStyle->SetStatY(0.75);
  gStyle->SetStatW(0.20);
  gStyle->SetStatH(0.10);
  // gStyle->SetStatH(0.35);
  //

  //        Float_t LeftOffset = 0.12;
  //Float_t TopOffset = 0.12;
  Float_t LeftOffset = 0.12;
  Float_t TopOffset = 0.22;

  gStyle->SetLineWidth(1);
  gStyle->SetErrorX(0);

  //---=[ Titles,Labels ]=-----------
  gStyle->SetOptTitle(0);  // title on/off
  //      gStyle->SetTitleColor(0);           // title color
  gStyle->SetTitleColor(1);  // title color
  //      gStyle->SetTitleX(0.35);            // title x-position
  gStyle->SetTitleX(0.15);  // title x-position
  gStyle->SetTitleH(0.15);  // title height
  //      gStyle->SetTitleW(0.53);            // title width
  gStyle->SetTitleW(0.60);         // title width
  gStyle->SetTitleFont(42);        // title font
  gStyle->SetTitleFontSize(0.07);  // title font size

  gStyle->SetPalette(1);
  //---=[ Pad style ]=----------------
  gStyle->SetPadTopMargin(TopOffset);
  gStyle->SetPadBottomMargin(LeftOffset);
  gStyle->SetPadRightMargin(TopOffset);
  gStyle->SetPadLeftMargin(LeftOffset);

  if (argc < 2)
    return 1;
  char fname[300];
  char refname[300];
  sprintf(fname, "%s", argv[1]);
  sprintf(refname, "%s", argv[2]);

  cout << fname << " " << refname << std::endl;

  //

  //======================================================================
  // Connect the input files, parameters and get the 2-d histogram in memory
  //    TFile *hfile= new TFile("GlobalHist.root", "READ");
  string promt = (string)fname;
  string runnumber = "";
  for (unsigned int i = promt.size() - 11; i < promt.size() - 5; i++)
    runnumber += fname[i];
  string refrunnumber = "";
  promt = (string)refname;
  for (unsigned int i = promt.size() - 11; i < promt.size() - 5; i++)
    refrunnumber += refname[i];

  TFile *hfile = new TFile(fname, "READ");
  hfile->ls();
  TDirectory *dir = (TDirectory *)hfile->FindObjectAny(dirnm.c_str());
  //TFile *hreffile = new TFile(refname, "READ");
  //megatile channels
  //CUTS:    [test][subdetector]             CapID(Test=1;  ADC amplitude Am(Test= 2);  Width for Wm(Test=3);     Ratio cut for Rm(Test=4);  TS mean for TNm(test=5);   TS max  for TXm(Test=6);
  double MIN_M[7][5] = {{0., 0., 0., 0., 0.},
                        {0., 0., 0., 0., 0.},
                        {0, 0., 0., 0., 0.},
                        {0, 1.0, 1.0, 0.2, 0.1},
                        {0, 0.10, 0.10, 0.18, 0.30},
                        {0, 0.8, 0.8, 0.8, 0.1},
                        {0, -0.5, -0.5, -0.5, -0.5}};
  double MAX_M[7][5] = {{0., 0., 0., 0., 0.},
                        {0., 0., 0., 0., 0.},
                        {0, 900, 900, 9000, 3000},
                        {0, 3.9, 3.9, 4.4, 2.0},
                        {0, 0.95, 0.98, 0.96, 1.04},
                        {0, 8.0, 8.0, 8.0, 2.8},
                        {0, 6.5, 6.5, 6.5, 3.5}};

  // calibration channels:
  double MIN_C[7][5] = {{0., 0., 0., 0., 0.},
                        {0., 0., 0., 0., 0.},
                        {0, 120., 120., 120., 60.},
                        {0, 1.0, 1.0, 0.50, 0.2},
                        {0, 0.6, 0.64, 0.25, 0.25},
                        {0, 1.0, 1.0, 1.0, 1.0},
                        {0, 0.5, 0.5, 0.5, 0.5}};
  double MAX_C[7][5] = {{0., 0., 0., 0., 0.},
                        {0., 0., 0., 0., 0.},
                        {0, 1E20, 1E20, 1E20, 1E20},
                        {0, 2.3, 2.3, 3.0, 2.3},
                        {0, 1., 1., 1., 1.00},
                        {0, 5.5, 5.5, 3.5, 5.2},
                        {0, 8.5, 8.5, 8.5, 9.5}};
  double porog[5] = {0., 2., 2., 5., 1.};  // Cut for GS test in pro cents
  //    double porog[5] = {0., 200., 200., 100., 100.}; // Cut for GS test in pro cents
  double Pedest[2][5] = {{0., 0.2, 0.9, 0.1, 0.2}, {0., 0.2, 0.2, 0.1, 0.16}};  //Cuts for Pedestal  and pedestal  Width
  //======================================================================
  // with TfileService implementation, change everywhere below:     hfile->Get     to     dir->FindObjectAny
  //======================================================================
  // Prepare histograms and plot them to .png files

  //TCanvas *cHB = new TCanvas("cHB","cHB",1000,500);
  TCanvas *cHB = new TCanvas("cHB", "cHB", 1000, 1000);
  //TCanvas *cHE = new TCanvas("cHE","cHE",1500,500);
  TCanvas *cHE = new TCanvas("cHE", "cHE", 1500, 1500);
  //  TCanvas *cONE = new TCanvas("cONE","cONE",500,500);
  TCanvas *cONE = new TCanvas("cONE", "cONE", 1500, 500);
  TCanvas *cHO = new TCanvas("cHO", "cHO", 500, 500);
  TCanvas *cPED = new TCanvas("cPED", "cPED", 1000, 500);
  //TCanvas *cHF = new TCanvas("cHF","cHF",1000,1000);
  TCanvas *cHF = new TCanvas("cHF", "cHF", 1000, 1000);

  char *str = (char *)alloca(10000);

  // before upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3; HO depth4; HF depth1,2
  // 5 depthes:  0(empty),   1,2,3,4

  // upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 8 depthes:  0(empty),   1,2,3,4,5,6,7

  // upgrade 2019:
  // depth: HB depth1,2,3,4; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 10 depthes:  0(empty),   1,2,3,4,5,6,7,8,9

  //  Int_t ALLDEPTH = 5;
  //  Int_t ALLDEPTH = 8;
  Int_t ALLDEPTH = 10;
  //massive_indx=1  2  3  4  5
  //             0, HB,HE,HO,HF
  int k_min[5] = {0, 1, 1, 4, 1};  // minimum depth for each subdet

  //int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
  //int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
  int k_max[5] = {0, 4, 7, 4, 4};  // maximum depth for each subdet

  TH2F *Map_Ampl[33][5][ALLDEPTH];  // 2D histogramm for test,subdet,depth
  //AZ2023  TH2F *Map_SUBGOOD[5][ALLDEPTH];        // 2d histogramm for subdet, depth
  TH2F *Map_SUB[5][ALLDEPTH];            // 2d histogramm for subdet, depth
  TH1F *HistAmplDepth[22][5][ALLDEPTH];  // 1d histogramm for test,subdet, depth
  TH1F *HistAmpl[22][5];                 // 1d histogramm for test,subdet
  TH2F *Map_SUBTS[5][ALLDEPTH];          // 2d histogramm for subdet, depth in different TSs

  TH1F *HistPed[3][5][4];           // 1d  histogramm for test,subdet, CapID
  TH2F *Map_Ped[3][5];              // 2d  histogramm for test,subdet -> test 33
  TH1F *hist_GoodTSshape[5];        // 1d  histogramm for TS shape subdet -> test 41
  TH1F *hist_GoodTSshape0[5];       // 1d  histogramm for TS shape subdet -> test 41
  TH1F *hist_BadTSshape[5];         // 1d  histogramm for TS shape subdet -> test 41
  TH1F *hist_BadTSshape0[5];        // 1d  histogramm for TS shape subdet -> test 41
  TH1F *hist_ADC_All[5];            // 1d  histogramm for TS shape subdet -> test 42
  TH1F *hist_ADC_DS[5][ALLDEPTH];   // 1d  histogramm for TS shape subdet, depth -> test 42
  TH1F *hist_SumADC[5][ALLDEPTH];   // 1d  histogramm for TS shape subdet, depth -> test 43
  TH1F *hist_SumADC0[5][ALLDEPTH];  // 1d  histogramm for TS shape subdet, depth -> test 43
  TH1F *hist_SumADC1[5][ALLDEPTH];  // 1d  histogramm for TS shape subdet, depth -> test 43

  Map_SUB[1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  Map_SUB[1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  Map_SUB[1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HB");
  Map_SUB[1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HB");
  Map_SUB[2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  Map_SUB[2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  Map_SUB[2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  Map_SUB[2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HE");
  Map_SUB[2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5_HE");
  Map_SUB[2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6_HE");
  Map_SUB[2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7_HE");
  Map_SUB[3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  Map_SUB[4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  Map_SUB[4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  Map_SUB[4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HF");
  Map_SUB[4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HF");
  //AZ2023:
  /*
  Map_SUBGOOD[1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  Map_SUBGOOD[1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  Map_SUBGOOD[1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HB");
  Map_SUBGOOD[1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HB");
  Map_SUBGOOD[2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  Map_SUBGOOD[2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  Map_SUBGOOD[2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  Map_SUBGOOD[2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HE");
  Map_SUBGOOD[2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5_HE");
  Map_SUBGOOD[2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6_HE");
  Map_SUBGOOD[2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7_HE");
  Map_SUBGOOD[3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  Map_SUBGOOD[4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  Map_SUBGOOD[4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  Map_SUBGOOD[4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3_HF");
  Map_SUBGOOD[4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4_HF");
*/
  //+++++++++++++++++++++++++++++
  //Test 0 Entries
  //+++++++++++++++++++++++++++++

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //     if (sub==1) cHB->Divide(2,1);
    if (sub == 1)
      cHB->Divide(2, 2);
    //     if (sub==2) cHE->Divide(3,1);
    if (sub == 2)
      cHE->Divide(3, 3);
    if (sub == 3)
      cONE->Divide(1, 1);
    //     if (sub==4) cHF->Divide(2,1);
    if (sub == 4)
      cHF->Divide(2, 2);
    //     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      Map_SUB[sub][k]->SetTitle(str);
      Map_SUB[sub][k]->SetXTitle("#eta \b");
      Map_SUB[sub][k]->SetYTitle("#phi \b");
      Map_SUB[sub][k]->SetZTitle("Number of events \b");
      if (sub == 3)
        Map_SUB[sub][k]->SetTitleOffset(0.8, "Z");
      Map_SUB[sub][k]->Draw("COLZ");
      Map_SUB[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      //            Map_SUB[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
      if (sub == 1) {
        cHB->Modified();
        cHB->Update();
      }
      if (sub == 2) {
        cHE->Modified();
        cHE->Update();
      }
      if (sub == 3) {
        cONE->Modified();
        cONE->Update();
      }
      if (sub == 4) {
        cHF->Modified();
        cHF->Update();
      }
    }  //end depth

    if (sub == 1) {
      cHB->Print("MapRateEntryHB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MapRateEntryHE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateEntryHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MapRateEntryHF.png");
      cHF->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++
  //Test 1 (Cm) Rate of Cap ID errors
  //+++++++++++++++++++++++++++++

  Map_Ampl[1][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Error_HB");
  Map_Ampl[1][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Error_HB");
  Map_Ampl[1][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Error_HE");
  Map_Ampl[1][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Error_HE");
  Map_Ampl[1][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Error_HE");
  Map_Ampl[1][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Error_HO");
  Map_Ampl[1][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Error_HF");
  Map_Ampl[1][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Error_HF");

  Map_Ampl[1][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Error_HE");
  Map_Ampl[1][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5Error_HE");
  Map_Ampl[1][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6Error_HE");
  Map_Ampl[1][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7Error_HE");
  Map_Ampl[1][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Error_HF");
  Map_Ampl[1][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Error_HF");

  Map_Ampl[1][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Error_HB");
  Map_Ampl[1][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Error_HB");

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //     if (sub==1) cHB->Divide(2,1);
    if (sub == 1)
      cHB->Divide(2, 2);
    //     if (sub==2) cHE->Divide(3,1);
    if (sub == 2)
      cHE->Divide(3, 3);
    if (sub == 3)
      cONE->Divide(1, 1);
    //     if (sub==4) cHF->Divide(2,1);
    if (sub == 4)
      cHF->Divide(2, 2);
    //     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      Map_Ampl[1][sub][k]->Divide(Map_Ampl[1][sub][k], Map_SUB[sub][k], 1, 1, "B");
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      Map_Ampl[1][sub][k]->SetTitle(str);
      Map_Ampl[1][sub][k]->SetXTitle("#eta \b");
      Map_Ampl[1][sub][k]->SetYTitle("#phi \b");
      Map_Ampl[1][sub][k]->SetZTitle("Rate \b");
      if (sub == 3)
        Map_Ampl[1][sub][k]->SetTitleOffset(0.8, "Z");
      Map_Ampl[1][sub][k]->Draw("COLZ");
      Map_Ampl[1][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      Map_Ampl[1][sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
      if (sub == 1) {
        cHB->Modified();
        cHB->Update();
      }
      if (sub == 2) {
        cHE->Modified();
        cHE->Update();
      }
      if (sub == 3) {
        cONE->Modified();
        cONE->Update();
      }
      if (sub == 4) {
        cHF->Modified();
        cHF->Update();
      }
    }  //end depth

    if (sub == 1) {
      cHB->Print("MapRateCapIDHB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MapRateCapIDHE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateCapIDHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MapRateCapIDHF.png");
      cHF->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++
  //Test 2 (Am) ADC amplitude
  //+++++++++++++++++++++++++++++

  Map_Ampl[2][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1ADCAmpl225_HB");
  Map_Ampl[2][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2ADCAmpl225_HB");
  Map_Ampl[2][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1ADCAmpl225_HE");
  Map_Ampl[2][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2ADCAmpl225_HE");
  Map_Ampl[2][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3ADCAmpl225_HE");
  Map_Ampl[2][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4ADCAmpl225_HO");
  Map_Ampl[2][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1ADCAmpl225_HF");
  Map_Ampl[2][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2ADCAmpl225_HF");

  Map_Ampl[2][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4ADCAmpl225_HE");
  Map_Ampl[2][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5ADCAmpl225_HE");
  Map_Ampl[2][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6ADCAmpl225_HE");
  Map_Ampl[2][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7ADCAmpl225_HE");
  Map_Ampl[2][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3ADCAmpl225_HF");
  Map_Ampl[2][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4ADCAmpl225_HF");

  Map_Ampl[2][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3ADCAmpl225_HB");
  Map_Ampl[2][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4ADCAmpl225_HB");

  HistAmpl[2][1] = (TH1F *)dir->FindObjectAny("h_ADCAmpl_HB");
  HistAmpl[2][2] = (TH1F *)dir->FindObjectAny("h_ADCAmpl_HE");
  HistAmpl[2][3] = (TH1F *)dir->FindObjectAny("h_ADCAmpl_HO");
  HistAmpl[2][4] = (TH1F *)dir->FindObjectAny("h_ADCAmpl_HF");

  //+++++++++++++++++++++++++++++
  //Test 3 (Wm) Rate of RMS
  //+++++++++++++++++++++++++++++

  Map_Ampl[3][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Amplitude225_HB");
  Map_Ampl[3][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Amplitude225_HB");
  Map_Ampl[3][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Amplitude225_HE");
  Map_Ampl[3][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Amplitude225_HE");
  Map_Ampl[3][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Amplitude225_HE");
  Map_Ampl[3][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Amplitude225_HO");
  Map_Ampl[3][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Amplitude225_HF");
  Map_Ampl[3][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Amplitude225_HF");

  Map_Ampl[3][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Amplitude225_HE");
  Map_Ampl[3][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5Amplitude225_HE");
  Map_Ampl[3][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6Amplitude225_HE");
  Map_Ampl[3][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7Amplitude225_HE");
  Map_Ampl[3][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Amplitude225_HF");
  Map_Ampl[3][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Amplitude225_HF");

  Map_Ampl[3][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Amplitude225_HB");
  Map_Ampl[3][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Amplitude225_HB");

  HistAmpl[3][1] = (TH1F *)dir->FindObjectAny("h_Amplitude_HB");
  HistAmpl[3][2] = (TH1F *)dir->FindObjectAny("h_Amplitude_HE");
  HistAmpl[3][3] = (TH1F *)dir->FindObjectAny("h_Amplitude_HO");
  HistAmpl[3][4] = (TH1F *)dir->FindObjectAny("h_Amplitude_HF");

  //+++++++++++++++++++++++++++++
  //Test 4 (Rm) Rate of ratio 4 near max TS/ All TS
  //+++++++++++++++++++++++++++++

  Map_Ampl[4][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Ampl047_HB");
  Map_Ampl[4][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Ampl047_HB");
  Map_Ampl[4][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Ampl047_HE");
  Map_Ampl[4][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Ampl047_HE");
  Map_Ampl[4][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Ampl047_HE");
  Map_Ampl[4][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Ampl047_HO");
  Map_Ampl[4][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1Ampl047_HF");
  Map_Ampl[4][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2Ampl047_HF");

  Map_Ampl[4][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Ampl047_HE");
  Map_Ampl[4][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5Ampl047_HE");
  Map_Ampl[4][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6Ampl047_HE");
  Map_Ampl[4][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7Ampl047_HE");
  Map_Ampl[4][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Ampl047_HF");
  Map_Ampl[4][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Ampl047_HF");

  Map_Ampl[4][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3Ampl047_HB");
  Map_Ampl[4][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4Ampl047_HB");

  HistAmpl[4][1] = (TH1F *)dir->FindObjectAny("h_Ampl_HB");
  HistAmpl[4][2] = (TH1F *)dir->FindObjectAny("h_Ampl_HE");
  HistAmpl[4][3] = (TH1F *)dir->FindObjectAny("h_Ampl_HO");
  HistAmpl[4][4] = (TH1F *)dir->FindObjectAny("h_Ampl_HF");

  //+++++++++++++++++++++++++++++
  //Test 5 (TNm) Mean position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[5][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HB");
  Map_Ampl[5][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HB");
  Map_Ampl[5][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HE");
  Map_Ampl[5][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HE");
  Map_Ampl[5][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA225_HE");
  Map_Ampl[5][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA225_HO");
  Map_Ampl[5][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HF");
  Map_Ampl[5][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HF");

  Map_Ampl[5][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA225_HE");
  Map_Ampl[5][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5TSmeanA225_HE");
  Map_Ampl[5][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6TSmeanA225_HE");
  Map_Ampl[5][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7TSmeanA225_HE");
  Map_Ampl[5][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA225_HF");
  Map_Ampl[5][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA225_HF");

  Map_Ampl[5][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA225_HB");
  Map_Ampl[5][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA225_HB");

  HistAmpl[5][1] = (TH1F *)dir->FindObjectAny("h_TSmeanA_HB");
  HistAmpl[5][2] = (TH1F *)dir->FindObjectAny("h_TSmeanA_HE");
  HistAmpl[5][3] = (TH1F *)dir->FindObjectAny("h_TSmeanA_HO");
  HistAmpl[5][4] = (TH1F *)dir->FindObjectAny("h_TSmeanA_HF");

  //+++++++++++++++++++++++++++++
  //Test 6 (TXm) Maximum position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[6][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmaxA225_HB");
  Map_Ampl[6][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmaxA225_HB");
  Map_Ampl[6][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmaxA225_HE");
  Map_Ampl[6][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmaxA225_HE");
  Map_Ampl[6][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmaxA225_HE");
  Map_Ampl[6][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmaxA225_HO");
  Map_Ampl[6][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmaxA225_HF");
  Map_Ampl[6][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmaxA225_HF");

  Map_Ampl[6][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmaxA225_HE");
  Map_Ampl[6][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5TSmaxA225_HE");
  Map_Ampl[6][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6TSmaxA225_HE");
  Map_Ampl[6][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7TSmaxA225_HE");
  Map_Ampl[6][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmaxA225_HF");
  Map_Ampl[6][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmaxA225_HF");

  Map_Ampl[6][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmaxA225_HB");
  Map_Ampl[6][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmaxA225_HB");

  HistAmpl[6][1] = (TH1F *)dir->FindObjectAny("h_TSmaxA_HB");
  HistAmpl[6][2] = (TH1F *)dir->FindObjectAny("h_TSmaxA_HE");
  HistAmpl[6][3] = (TH1F *)dir->FindObjectAny("h_TSmaxA_HO");
  HistAmpl[6][4] = (TH1F *)dir->FindObjectAny("h_TSmaxA_HF");

  for (int test = 2; test <= 6; test++) {  //Test: 2-Am, 3-Wm, 4-Rm, 5-TNm, 6-TXm,
    for (int sub = 1; sub <= 4; sub++) {   //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                           //        if (sub==1) cHB->Divide(2,1);
      if (sub == 1)
        cHB->Divide(2, 2);
      //        if (sub==2) cHE->Divide(3,1);
      if (sub == 2)
        cHE->Divide(3, 3);
      if (sub == 3)
        cONE->Divide(1, 1);
      //        if (sub==4) cHF->Divide(2,1);
      if (sub == 4)
        cHF->Divide(2, 2);
      //        int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
      //        int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
      //        int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
      for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
        if (sub == 1)
          cHB->cd(k);
        if (sub == 2)
          cHE->cd(k);
        if (sub == 3)
          cONE->cd(k - 3);
        if (sub == 4)
          cHF->cd(k);
        Map_Ampl[test][sub][k]->Divide(Map_Ampl[test][sub][k], Map_SUB[sub][k], 1, 1, "B");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogz();
        if (sub == 1)
          sprintf(str, "HB, Depth%d \b", k);
        if (sub == 2)
          sprintf(str, "HE, Depth%d \b", k);
        if (sub == 3)
          sprintf(str, "HO, Depth%d \b", k);
        if (sub == 4)
          sprintf(str, "HF, Depth%d \b", k);
        Map_Ampl[test][sub][k]->SetTitle(str);
        Map_Ampl[test][sub][k]->SetXTitle("#eta \b");
        Map_Ampl[test][sub][k]->SetYTitle("#phi \b");
        Map_Ampl[test][sub][k]->SetZTitle("Rate \b");
        if (sub == 3)
          Map_Ampl[test][sub][k]->SetTitleOffset(0.8, "Z");
        Map_Ampl[test][sub][k]->Draw("COLZ");
        Map_Ampl[test][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
        Map_Ampl[test][sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
        if (sub == 1) {
          cHB->Modified();
          cHB->Update();
        }
        if (sub == 2) {
          cHE->Modified();
          cHE->Update();
        }
        if (sub == 3) {
          cONE->Modified();
          cONE->Update();
        }
        if (sub == 4) {
          cHF->Modified();
          cHF->Update();
        }
      }  //end depth
      if (test == 2) {
        if (sub == 1) {
          cHB->Print("MapRateAmplHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRateAmplHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateAmplHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRateAmplHF.png");
          cHF->Clear();
        }
      }
      if (test == 3) {
        if (sub == 1) {
          cHB->Print("MapRateRMSHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRateRMSHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateRMSHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRateRMSHF.png");
          cHF->Clear();
        }
      }
      if (test == 4) {
        if (sub == 1) {
          cHB->Print("MapRate43TStoAllTSHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRate43TStoAllTSHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRate43TStoAllTSHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRate43TStoAllTSHF.png");
          cHF->Clear();
        }
      }
      if (test == 5) {
        if (sub == 1) {
          cHB->Print("MapRateMeanPosHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRateMeanPosHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateMeanPosHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRateMeanPosHF.png");
          cHF->Clear();
        }
      }
      if (test == 6) {
        if (sub == 1) {
          cHB->Print("MapRateMaxPosHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRateMaxPosHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateMaxPosHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRateMaxPosHF.png");
          cHF->Clear();
        }
      }

      //          cONE->Divide(1,1);
      /*
          cONE->Divide(2,1);
	  if(test == 2 && sub == 2 ) {
	    cONE->cd(2);
	    TH1F *kjkjkhj2= (TH1F*)dir->FindObjectAny("h_ADCAmpl_HE");kjkjkhj2->Draw("");kjkjkhj2->SetTitle("HE, All Depth: shunt6");
	  }
	  if(test == 2 && sub == 1 ) {
	    cONE->cd(2);
	    TH1F *kjkjkhj1= (TH1F*)dir->FindObjectAny("h_ADCAmpl_HB");kjkjkhj1->Draw("");kjkjkhj1->SetTitle("HB, All Depth: shunt6");
	  }
*/

      cONE->Divide(3, 1);
      if (test == 2 && sub == 2) {
        cONE->cd(2);
        TH1F *kjkjkhj2 = (TH1F *)dir->FindObjectAny("h_AmplitudeHEtest1");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogy();
        kjkjkhj2->Draw("");
        kjkjkhj2->SetXTitle("HE, All Depth: shunt1");
        cONE->cd(3);
        TH1F *kjkjkhj3 = (TH1F *)dir->FindObjectAny("h_AmplitudeHEtest6");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogy();
        kjkjkhj3->Draw("");
        kjkjkhj3->SetXTitle("HE, All Depth: shunt6");
      }
      if (test == 2 && sub == 1) {
        cONE->cd(2);
        TH1F *kjkjkhb2 = (TH1F *)dir->FindObjectAny("h_AmplitudeHBtest1");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogy();
        kjkjkhb2->Draw("");
        kjkjkhb2->SetXTitle("HB, All Depth: shunt1");
        cONE->cd(3);
        TH1F *kjkjkhb3 = (TH1F *)dir->FindObjectAny("h_AmplitudeHBtest6");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogy();
        kjkjkhb3->Draw("");
        kjkjkhb3->SetXTitle("HB, All Depth: shunt6");
      }

      cONE->cd(1);
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogy();
      if (sub == 1)
        HistAmpl[test][sub]->SetTitle("HB, All Depth: shunt6");
      if (sub == 2)
        HistAmpl[test][sub]->SetTitle("HE, All Depth: shunt6");
      if (sub == 3)
        HistAmpl[test][sub]->SetTitle("HO, All Depth");
      if (sub == 4)
        HistAmpl[test][sub]->SetTitle("HF, All Depth");
      if (test == 2)
        HistAmpl[test][sub]->SetXTitle("ADC Amlitude in each event & cell \b");
      if (test == 3)
        HistAmpl[test][sub]->SetXTitle("RMS in each event & cell \b");
      if (test == 4)
        HistAmpl[test][sub]->SetXTitle("Ratio in each event & cell \b");
      if (test == 5)
        HistAmpl[test][sub]->SetXTitle("Mean TS position in each event & cell \b");
      if (test == 6)
        HistAmpl[test][sub]->SetXTitle("Max TS position in each event & cell \b");
      HistAmpl[test][sub]->SetYTitle("Number of cell-events \b");
      HistAmpl[test][sub]->SetLineColor(4);
      HistAmpl[test][sub]->SetLineWidth(2);
      HistAmpl[test][sub]->SetTitleOffset(1.4, "Y");
      HistAmpl[test][sub]->Draw("");
      // //        HistAmpl[test][sub]->GetYaxis()->SetRangeUser(1., 100.);
      //          if (test==2) {gPad->SetLogx(); HistAmpl[test][sub]->GetXaxis()->SetRangeUser(1., 10000.);}
      if (test == 2) {
        gPad->SetLogx();
      }
      if (test == 3)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 5.);  // width
      if (test == 4)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 1.);  // R
      if (test == 5)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);  // Tn
      if (test == 6)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);  // Tx
      cONE->Modified();
      cONE->Update();
      double min_x[] = {MIN_M[test][sub], MIN_M[test][sub]};
      double min_y[] = {0., 100000000.};
      TGraph *MIN = new TGraph(2, min_x, min_y);
      MIN->SetLineStyle(2);
      MIN->SetLineColor(2);
      MIN->SetLineWidth(2 + 100 * 100);
      MIN->SetFillStyle(3005);
      MIN->SetFillColor(2);
      MIN->Draw("L");
      double max_x[] = {MAX_M[test][sub], MAX_M[test][sub]};
      double max_y[] = {0., 100000000.};
      TGraph *MAX = new TGraph(2, max_x, max_y);
      MAX->SetLineStyle(2);
      MAX->SetLineColor(2);
      MAX->SetLineWidth(-2 - 100 * 100);
      MAX->SetFillStyle(3004);
      MAX->SetFillColor(2);
      MAX->Draw("L");
      if (test == 2) {
        if (sub == 1) {
          cONE->Print("HistAmplHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistAmplHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistAmplHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistAmplHF.png");
          cONE->Clear();
        }
      }
      if (test == 3) {
        if (sub == 1) {
          cONE->Print("HistRMSHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistRMSHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistRMSHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistRMSHF.png");
          cONE->Clear();
        }
      }
      if (test == 4) {
        if (sub == 1) {
          cONE->Print("Hist43TStoAllTSHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("Hist43TStoAllTSHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("Hist43TStoAllTSHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("Hist43TStoAllTSHF.png");
          cONE->Clear();
        }
      }
      if (test == 5) {
        if (sub == 1) {
          cONE->Print("HistMeanPosHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistMeanPosHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistMeanPosHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistMeanPosHF.png");
          cONE->Clear();
        }
      }
      if (test == 6) {
        if (sub == 1) {
          cONE->Print("HistMaxPosHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistMaxPosHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistMaxPosHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistMaxPosHF.png");
          cONE->Clear();
        }
      }
    }  // end sub
  }    //end test

  //+++++++++++++++++++++++++++++++++++
  //Test 31, 32 Pedestal, pedestalWidths
  //++++++++++++++++++++++++++++++++++++

  Map_Ampl[31][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestal_HB");
  Map_Ampl[31][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestal_HB");
  Map_Ampl[31][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestal_HB");
  Map_Ampl[31][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestal_HB");
  Map_Ampl[31][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestal_HE");
  Map_Ampl[31][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestal_HE");
  Map_Ampl[31][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestal_HE");
  Map_Ampl[31][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestal_HE");
  Map_Ampl[31][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5pedestal_HE");
  Map_Ampl[31][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6pedestal_HE");
  Map_Ampl[31][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7pedestal_HE");
  Map_Ampl[31][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestal_HO");
  Map_Ampl[31][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestal_HF");
  Map_Ampl[31][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestal_HF");
  Map_Ampl[31][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestal_HF");
  Map_Ampl[31][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestal_HF");

  Map_Ampl[32][1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestalw_HB");
  Map_Ampl[32][1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestalw_HB");
  Map_Ampl[32][1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestalw_HB");
  Map_Ampl[32][1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestalw_HB");
  Map_Ampl[32][2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestalw_HE");
  Map_Ampl[32][2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestalw_HE");
  Map_Ampl[32][2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestalw_HE");
  Map_Ampl[32][2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestalw_HE");
  Map_Ampl[32][2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5pedestalw_HE");
  Map_Ampl[32][2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6pedestalw_HE");
  Map_Ampl[32][2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7pedestalw_HE");
  Map_Ampl[32][3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestalw_HO");
  Map_Ampl[32][4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1pedestalw_HF");
  Map_Ampl[32][4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2pedestalw_HF");
  Map_Ampl[32][4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3pedestalw_HF");
  Map_Ampl[32][4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4pedestalw_HF");

  HistPed[1][1][0] = (TH1F *)dir->FindObjectAny("h_pedestal0_HB");
  HistPed[1][1][1] = (TH1F *)dir->FindObjectAny("h_pedestal1_HB");
  HistPed[1][1][2] = (TH1F *)dir->FindObjectAny("h_pedestal2_HB");
  HistPed[1][1][3] = (TH1F *)dir->FindObjectAny("h_pedestal3_HB");
  HistPed[2][1][0] = (TH1F *)dir->FindObjectAny("h_pedestalw0_HB");
  HistPed[2][1][1] = (TH1F *)dir->FindObjectAny("h_pedestalw1_HB");
  HistPed[2][1][2] = (TH1F *)dir->FindObjectAny("h_pedestalw2_HB");
  HistPed[2][1][3] = (TH1F *)dir->FindObjectAny("h_pedestalw3_HB");

  HistPed[1][2][0] = (TH1F *)dir->FindObjectAny("h_pedestal0_HE");
  HistPed[1][2][1] = (TH1F *)dir->FindObjectAny("h_pedestal1_HE");
  HistPed[1][2][2] = (TH1F *)dir->FindObjectAny("h_pedestal2_HE");
  HistPed[1][2][3] = (TH1F *)dir->FindObjectAny("h_pedestal3_HE");
  HistPed[2][2][0] = (TH1F *)dir->FindObjectAny("h_pedestalw0_HE");
  HistPed[2][2][1] = (TH1F *)dir->FindObjectAny("h_pedestalw1_HE");
  HistPed[2][2][2] = (TH1F *)dir->FindObjectAny("h_pedestalw2_HE");
  HistPed[2][2][3] = (TH1F *)dir->FindObjectAny("h_pedestalw3_HE");

  HistPed[1][3][0] = (TH1F *)dir->FindObjectAny("h_pedestal0_HO");
  HistPed[1][3][1] = (TH1F *)dir->FindObjectAny("h_pedestal1_HO");
  HistPed[1][3][2] = (TH1F *)dir->FindObjectAny("h_pedestal2_HO");
  HistPed[1][3][3] = (TH1F *)dir->FindObjectAny("h_pedestal3_HO");
  HistPed[2][3][0] = (TH1F *)dir->FindObjectAny("h_pedestalw0_HO");
  HistPed[2][3][1] = (TH1F *)dir->FindObjectAny("h_pedestalw1_HO");
  HistPed[2][3][2] = (TH1F *)dir->FindObjectAny("h_pedestalw2_HO");
  HistPed[2][3][3] = (TH1F *)dir->FindObjectAny("h_pedestalw3_HO");

  HistPed[1][4][0] = (TH1F *)dir->FindObjectAny("h_pedestal0_HF");
  HistPed[1][4][1] = (TH1F *)dir->FindObjectAny("h_pedestal1_HF");
  HistPed[1][4][2] = (TH1F *)dir->FindObjectAny("h_pedestal2_HF");
  HistPed[1][4][3] = (TH1F *)dir->FindObjectAny("h_pedestal3_HF");
  HistPed[2][4][0] = (TH1F *)dir->FindObjectAny("h_pedestalw0_HF");
  HistPed[2][4][1] = (TH1F *)dir->FindObjectAny("h_pedestalw1_HF");
  HistPed[2][4][2] = (TH1F *)dir->FindObjectAny("h_pedestalw2_HF");
  HistPed[2][4][3] = (TH1F *)dir->FindObjectAny("h_pedestalw3_HF");

  for (int test = 31; test <= 32; test++) {  //Test: 31-Pedestals, 32-pedestal Widths,
    for (int sub = 1; sub <= 4; sub++) {     //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
                                             //        if (sub==1) cHB->Divide(2,1);
      if (sub == 1)
        cHB->Divide(2, 2);
      //        if (sub==2) cHE->Divide(3,1);
      if (sub == 2)
        cHE->Divide(3, 3);
      if (sub == 3)
        cONE->Divide(1, 1);
      //        if (sub==4) cHF->Divide(2,1);
      if (sub == 4)
        cHF->Divide(2, 2);
      //        int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
      //        int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
      //        int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
      for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depths
        if (sub == 1)
          cHB->cd(k);
        if (sub == 2)
          cHE->cd(k);
        if (sub == 3)
          cONE->cd(k - 3);
        if (sub == 4)
          cHF->cd(k);
        Map_Ampl[test][sub][k]->Divide(Map_Ampl[test][sub][k], Map_SUB[sub][k], 1, 1, "B");
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogz();
        if (sub == 1)
          sprintf(str, "HB, Depth%d \b", k);
        if (sub == 2)
          sprintf(str, "HE, Depth%d \b", k);
        if (sub == 3)
          sprintf(str, "HO, Depth%d \b", k);
        if (sub == 4)
          sprintf(str, "HF, Depth%d \b", k);
        Map_Ampl[test][sub][k]->SetTitle(str);
        Map_Ampl[test][sub][k]->SetXTitle("#eta \b");
        Map_Ampl[test][sub][k]->SetYTitle("#phi \b");
        Map_Ampl[test][sub][k]->SetZTitle("Rate \b");
        if (sub == 3)
          Map_Ampl[test][sub][k]->SetTitleOffset(0.8, "Z");
        Map_Ampl[test][sub][k]->Draw("COLZ");
        Map_Ampl[test][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
        Map_Ampl[test][sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
        if (sub == 1) {
          cHB->Modified();
          cHB->Update();
        }
        if (sub == 2) {
          cHE->Modified();
          cHE->Update();
        }
        if (sub == 3) {
          cONE->Modified();
          cONE->Update();
        }
        if (sub == 4) {
          cHF->Modified();
          cHF->Update();
        }
      }  //end depth
      if (test == 31) {
        if (sub == 1) {
          cHB->Print("MapRatePedHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRatePedHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRatePedHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRatePedHF.png");
          cHF->Clear();
        }
      }
      if (test == 32) {
        if (sub == 1) {
          cHB->Print("MapRatePedWidthsHB.png");
          cHB->Clear();
        }
        if (sub == 2) {
          cHE->Print("MapRatePedWidthsHE.png");
          cHE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRatePedWidthsHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cHF->Print("MapRatePedWidthsHF.png");
          cHF->Clear();
        }
      }

      ///////////////////////////////////////////////

      cPED->Divide(2, 2);
      for (int cap = 0; cap <= 3; cap++) {
        cPED->cd(cap + 1);
        gPad->SetGridy();
        gPad->SetGridx();
        gPad->SetLogy();

        if (sub == 1)
          sprintf(str, "HB, Cap%d, all depth\b", cap);
        if (sub == 2)
          sprintf(str, "HE, Cap%d, all depth\b", cap);
        if (sub == 3)
          sprintf(str, "HO, Cap%d, all depth\b", cap);
        if (sub == 4)
          sprintf(str, "HF, Cap%d, all depth\b", cap);

        HistPed[test - 30][sub][cap]->SetTitle(str);

        if (test == 31)
          HistPed[test - 30][sub][cap]->SetXTitle("Pedestals in each event & cell \b");
        if (test == 32)
          HistPed[test - 30][sub][cap]->SetXTitle("Pedestal Widths in each event & cell \b");

        HistPed[test - 30][sub][cap]->SetYTitle("Number of channel-events \b");
        HistPed[test - 30][sub][cap]->SetLineColor(4);
        HistPed[test - 30][sub][cap]->SetLineWidth(2);
        HistPed[test - 30][sub][cap]->SetTitleOffset(1.4, "Y");
        HistPed[test - 30][sub][cap]->Draw("");
        //            HistPed[test-30][sub][cap]->GetYaxis()->SetRangeUser(1., 100.);
        //            if (test==31) {gPad->SetLogx(); HistPed[test-30][sub][cap]->GetXaxis()->SetRangeUser(1., 10000.);}
        //   	      if (test==32) HistPed[test-30][sub][cap]->GetXaxis()->SetRangeUser(0., 5.);

        cPED->Modified();
        cPED->Update();
        double min_x[] = {Pedest[test - 31][sub], Pedest[test - 31][sub]};
        double min_y[] = {0., 100000000.};
        TGraph *MIN = new TGraph(2, min_x, min_y);
        MIN->SetLineStyle(2);
        MIN->SetLineColor(2);
        MIN->SetLineWidth(2 + 100 * 100);
        MIN->SetFillStyle(3005);
        MIN->SetFillColor(2);
        MIN->Draw("L");
      }
      if (test == 31) {
        if (sub == 1) {
          cPED->Print("HistPedestalsHB.png");
          cPED->Clear();
        }
        if (sub == 2) {
          cPED->Print("HistPedestalsHE.png");
          cPED->Clear();
        }
        if (sub == 3) {
          cPED->Print("HistPedestalsHO.png");
          cPED->Clear();
        }
        if (sub == 4) {
          cPED->Print("HistPedestalsHF.png");
          cPED->Clear();
        }
      }
      if (test == 32) {
        if (sub == 1) {
          cPED->Print("HistPedestalWidthsHB.png");
          cPED->Clear();
        }
        if (sub == 2) {
          cPED->Print("HistPedestalWidthsHE.png");
          cPED->Clear();
        }
        if (sub == 3) {
          cPED->Print("HistPedestalWidthsHO.png");
          cPED->Clear();
        }
        if (sub == 4) {
          cPED->Print("HistPedestalWidthsHF.png");
          cPED->Clear();
        }
      }
    }  // end sub
  }    //end test 31,32

  //+++++++++++++++++++++++++++++++++++
  //Test 33 Correlation of Pedestal, pedestalWidths Vs fullAmplitude
  //++++++++++++++++++++++++++++++++++++

  cPED->Clear();
  Map_Ped[1][1] = (TH2F *)dir->FindObjectAny("h2_pedvsampl_HB");
  Map_Ped[1][2] = (TH2F *)dir->FindObjectAny("h2_pedvsampl_HE");
  Map_Ped[1][3] = (TH2F *)dir->FindObjectAny("h2_pedvsampl_HO");
  Map_Ped[1][4] = (TH2F *)dir->FindObjectAny("h2_pedvsampl_HF");
  Map_Ped[2][1] = (TH2F *)dir->FindObjectAny("h2_pedwvsampl_HB");
  Map_Ped[2][2] = (TH2F *)dir->FindObjectAny("h2_pedwvsampl_HE");
  Map_Ped[2][3] = (TH2F *)dir->FindObjectAny("h2_pedwvsampl_HO");
  Map_Ped[2][4] = (TH2F *)dir->FindObjectAny("h2_pedwvsampl_HF");
  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
    cPED->Divide(2, 1);
    for (int test = 1; test <= 2; test++) {
      cPED->cd(test);
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      if (test == 1)
        Map_Ped[test][sub]->SetXTitle("Pedestal, fC \b");
      if (test == 2)
        Map_Ped[test][sub]->SetXTitle("pedestal Width, fC \b");
      Map_Ped[test][sub]->SetYTitle("Amplitude, fC \b");
      Map_Ped[test][sub]->SetZTitle("entries  \b");
      if (test == 1)
        sprintf(str, "Cap0 Pedestal vs Amplitude \b");
      if (test == 2)
        sprintf(str, "Cap0 pedestalWidth vs Amplitude \b");
      Map_Ped[test][sub]->SetTitle(str);
      Map_Ped[test][sub]->Draw("COLZ");
      // Map_Ped[test][sub]->GetYaxis()->SetRangeUser(0, 72.);
      //      Map_Ped[test][sub]->GetZaxis()->SetRangeUser(0.0001, 1.);
      cPED->Modified();
      cPED->Update();
    }  // test 1,2
    if (sub == 1) {
      cPED->Print("CorrelationsMapPedestalVsfullAmplitudeHB.png");
      cPED->Clear();
    }
    if (sub == 2) {
      cPED->Print("CorrelationsMapPedestalVsfullAmplitudeHE.png");
      cPED->Clear();
    }
    if (sub == 3) {
      cPED->Print("CorrelationsMapPedestalVsfullAmplitudeHO.png");
      cPED->Clear();
    }
    if (sub == 4) {
      cPED->Print("CorrelationsMapPedestalVsfullAmplitudeHF.png");
      cPED->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++++++++
  //Test 41 Time Slices shape for good and bad channels
  //++++++++++++++++++++++++++++++++++++

  cONE->Clear();
  hist_GoodTSshape[1] = (TH1F *)dir->FindObjectAny("h_shape_good_channels_HB");
  hist_GoodTSshape[2] = (TH1F *)dir->FindObjectAny("h_shape_good_channels_HE");
  hist_GoodTSshape[3] = (TH1F *)dir->FindObjectAny("h_shape_good_channels_HO");
  hist_GoodTSshape[4] = (TH1F *)dir->FindObjectAny("h_shape_good_channels_HF");

  hist_GoodTSshape0[1] = (TH1F *)dir->FindObjectAny("h_shape0_good_channels_HB");
  hist_GoodTSshape0[2] = (TH1F *)dir->FindObjectAny("h_shape0_good_channels_HE");
  hist_GoodTSshape0[3] = (TH1F *)dir->FindObjectAny("h_shape0_good_channels_HO");
  hist_GoodTSshape0[4] = (TH1F *)dir->FindObjectAny("h_shape0_good_channels_HF");

  hist_BadTSshape[1] = (TH1F *)dir->FindObjectAny("h_shape_bad_channels_HB");
  hist_BadTSshape[2] = (TH1F *)dir->FindObjectAny("h_shape_bad_channels_HE");
  hist_BadTSshape[3] = (TH1F *)dir->FindObjectAny("h_shape_bad_channels_HO");
  hist_BadTSshape[4] = (TH1F *)dir->FindObjectAny("h_shape_bad_channels_HF");

  hist_BadTSshape0[1] = (TH1F *)dir->FindObjectAny("h_shape0_bad_channels_HB");
  hist_BadTSshape0[2] = (TH1F *)dir->FindObjectAny("h_shape0_bad_channels_HE");
  hist_BadTSshape0[3] = (TH1F *)dir->FindObjectAny("h_shape0_bad_channels_HO");
  hist_BadTSshape0[4] = (TH1F *)dir->FindObjectAny("h_shape0_bad_channels_HF");

  cONE->cd(1);

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF

    gPad->SetGridy();
    gPad->SetGridx();
    gPad->SetLogz();
    hist_GoodTSshape[sub]->Divide(hist_GoodTSshape[sub], hist_GoodTSshape0[sub], 1, 1, "B");
    hist_GoodTSshape[sub]->SetXTitle("Time slice \b");
    hist_GoodTSshape[sub]->SetYTitle("ADC counts \b");
    sprintf(str, "Mean ADC Shape \b");
    hist_GoodTSshape[sub]->SetTitle(str);
    hist_GoodTSshape[sub]->Draw("");
    // hist_GoodTSshape[sub]->GetYaxis()->SetRangeUser(0, 72.);
    // hist_GoodTSshape[sub]->GetZaxis()->SetRangeUser(0.0001, 1.);
    cONE->Modified();
    cONE->Update();
    if (sub == 1) {
      cONE->Print("HistGoodTSshapesHB.png");
      cONE->Clear();
    }
    if (sub == 2) {
      cONE->Print("HistGoodTSshapesHE.png");
      cONE->Clear();
    }
    if (sub == 3) {
      cONE->Print("HistGoodTSshapesHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cONE->Print("HistGoodTSshapesHF.png");
      cONE->Clear();
    }
  }  // end sub

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF

    gPad->SetGridy();
    gPad->SetGridx();
    gPad->SetLogz();
    hist_BadTSshape[sub]->Divide(hist_BadTSshape[sub], hist_BadTSshape0[sub], 1, 1, "B");
    hist_BadTSshape[sub]->SetXTitle("Time slice \b");
    hist_BadTSshape[sub]->SetYTitle("ADC counts \b");
    sprintf(str, "Mean ADC Shape \b");
    hist_BadTSshape[sub]->SetTitle(str);
    hist_BadTSshape[sub]->Draw("");
    // hist_BadTSshape[sub]->GetYaxis()->SetRangeUser(0, 72.);
    // hist_BadTSshape[sub]->GetZaxis()->SetRangeUser(0.0001, 1.);
    cONE->Modified();
    cONE->Update();
    if (sub == 1) {
      cONE->Print("HistBadTSshapesHB.png");
      cONE->Clear();
    }
    if (sub == 2) {
      cONE->Print("HistBadTSshapesHE.png");
      cONE->Clear();
    }
    if (sub == 3) {
      cONE->Print("HistBadTSshapesHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cONE->Print("HistBadTSshapesHF.png");
      cONE->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++
  //Entries in different TSs:
  //+++++++++++++++++++++++++++++
  Map_SUBTS[1][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TS2_HB");
  Map_SUBTS[1][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TS2_HB");
  Map_SUBTS[1][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TS2_HB");
  Map_SUBTS[1][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TS2_HB");

  Map_SUBTS[2][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TS2_HE");
  Map_SUBTS[2][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TS2_HE");
  Map_SUBTS[2][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TS2_HE");
  Map_SUBTS[2][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TS2_HE");
  Map_SUBTS[2][5] = (TH2F *)dir->FindObjectAny("h_mapDepth5TS2_HE");
  Map_SUBTS[2][6] = (TH2F *)dir->FindObjectAny("h_mapDepth6TS2_HE");
  Map_SUBTS[2][7] = (TH2F *)dir->FindObjectAny("h_mapDepth7TS2_HE");

  Map_SUBTS[3][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TS012_HO");

  Map_SUBTS[4][1] = (TH2F *)dir->FindObjectAny("h_mapDepth1TS1_HF");
  Map_SUBTS[4][2] = (TH2F *)dir->FindObjectAny("h_mapDepth2TS1_HF");
  Map_SUBTS[4][3] = (TH2F *)dir->FindObjectAny("h_mapDepth3TS1_HF");
  Map_SUBTS[4][4] = (TH2F *)dir->FindObjectAny("h_mapDepth4TS1_HF");

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //     if (sub==1) cHB->Divide(2,1);
    if (sub == 1)
      cHB->Divide(2, 2);
    //     if (sub==2) cHE->Divide(3,1);
    if (sub == 2)
      cHE->Divide(3, 3);
    if (sub == 3)
      cONE->Divide(1, 1);
    //     if (sub==4) cHF->Divide(2,1);
    if (sub == 4)
      cHF->Divide(2, 2);
    //     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      Map_SUBTS[sub][k]->Divide(Map_SUBTS[sub][k], Map_SUB[sub][k], 1, 1, "B");
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      Map_SUBTS[sub][k]->SetTitle(str);
      Map_SUBTS[sub][k]->SetXTitle("#eta \b");
      Map_SUBTS[sub][k]->SetYTitle("#phi \b");
      Map_SUBTS[sub][k]->SetZTitle("Number of events \b");
      if (sub == 3)
        Map_SUBTS[sub][k]->SetTitleOffset(0.8, "Z");
      Map_SUBTS[sub][k]->Draw("COLZ");
      Map_SUBTS[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      //            Map_SUBTS[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
      if (sub == 1) {
        cHB->Modified();
        cHB->Update();
      }
      if (sub == 2) {
        cHE->Modified();
        cHE->Update();
      }
      if (sub == 3) {
        cONE->Modified();
        cONE->Update();
      }
      if (sub == 4) {
        cHF->Modified();
        cHF->Update();
      }
    }  //end depth

    if (sub == 1) {
      cHB->Print("Hist_mapDepthAllTS2_HB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("Hist_mapDepthAllTS2_HE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("Hist_mapDepthAllTS012_HO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("Hist_mapDepthAllTS1_HF.png");
      cHF->Clear();
    }
  }  // end sub

  //======================================================================

  //AZ2023:  std::cout << " We are here to print general 2D MAP " << std::endl;

  //======================================================================

  //======================================================================
  /// Prepare maps of good/bad channels:
  // i - Eta; j - Phi
  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
  //	     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
  //	     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet before upgrade
  //	     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
  //k-Depth

  //  TH2F *Map_ALL = new TH2F("Map_All", "Map_all", 82, -41, 40, 72, 0, 71);
  //AZ2023:
  //AZ2023  TH2F *Map_ALL = new TH2F("Map_All", "Map_all", 82, -41, 41, 72, 0, 72);

  /*
  int nx = Map_ALL->GetXaxis()->GetNbins();
  int ny = Map_ALL->GetYaxis()->GetNbins();
  cout << " nx= " << nx << " ny= " << ny << endl;
*/
  //  int NBad = 0;
  //  int NWarn = 0;
  //  int NCalib = 0;
  //  int NPed = 0;
  //  //    int Eta[3][10000]={0};
  //  int Eta[4][10000] = {0};
  //  int Phi[4][10000] = {0};
  //  int Sub[4][10000] = {0};
  //  int Depth[4][10000] = {0};
  //  string Comment[4][10000] = {""};
  //  string Text[33] = {"", "Cm", "Am", "Wm", "Rm", "TNm", "TXm", "", "", "", "", "Cc", "Ac", "Wc", "Rc", "TNc", "TXc",
  //                     "", "",   "",   "",   "GS", "",    "",    "", "", "", "", "",   "",   "",   "Pm", "pWm"};
  //  int flag_W = 0;
  //  int flag_B = 0;
  //  int flag_P = 0;
  //AZ2023:
  /*
  int fffffflag = 0;
  std::cout << " Map_ALL   SUBGOOD update " << std::endl;
  for (int sub = 1; sub <= 4; sub++) {
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {
      for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
          if (Map_SUB[sub][k]->GetBinContent(i, j) != 0) {
       //AZ2023     Map_SUBGOOD[sub][k]->SetBinContent(i, j, 0.5);
        //AZ2023    Map_ALL->SetBinContent(i, j, 0.5);
          }
        }
      }
    }
  }
*/
  //AZ2023:
  /*
  std::cout << " Map_ALL   SUBGOOD filling............... " << std::endl;
  for (int sub = 1; sub <= 4; sub++) {
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {
      for (int i = 1; i <= nx; i++) {
        for (int j = 1; j <= ny; j++) {
          //          flag_W = 0;
          //          flag_B = 0;
          //          flag_P = 0;
          //       CapID(Test=1;  ADC amplitude Am(Test= 2);  Width for Wm(Test=3);     Ratio cut for Rm(Test=4);  TS mean for TNm(test=5);   TS max  for TXm(Test=6);
          for (int test = 3; test <= 6; test++) {
            //	    cout<<" test= "<<test<<" sbd= "<<sub<<" depth= "<<k<<" eta= "<<i<<" , phi= "<<j<<endl;
            //	    cout<<" initial content Map_Ampl[test][sub][k]->GetBinContent(i, j)= "<<                  Map_Ampl[test][sub][k]->GetBinContent(i, j)       <<endl;

            //Bad
            //Rate 0.1 for displaying  on whole detector map and subdetector map
            if (Map_Ampl[test][sub][k]->GetBinContent(i, j) > 0.1) {
       //AZ2023       Map_ALL->SetBinContent(i, j, 1.);
           //AZ2023   Map_SUBGOOD[sub][k]->SetBinContent(i, j, 1.);
              fffffflag = 1;
            }

            if ((Map_Ampl[test][sub][k]->GetBinContent(i, j) != 0.) &&
                (Map_Ampl[test][sub][k]->GetBinContent(i, j) < 0.001)) {
              if (Map_SUBGOOD[sub][k]->GetBinContent(i, j) != 1.)
              //AZ2023  Map_SUBGOOD[sub][k]->SetBinContent(i, j, 0.75);
          //AZ2023     if (Map_ALL->GetBinContent(i, j) != 1.)Map_ALL->SetBinContent(i, j, 0.75);
              fffffflag = 2;
            }
            ////

            //	    if(fffffflag != 0)   cout<<"Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)  << "fffffflag = "<< fffffflag    <<endl;

          }  //end test

          //	  std::cout << " RUN3 2022 MAPS_SUB: Pedestals......"<< std::endl;
          //Pedestals
          for (int test = 31; test <= 32; test++) {
            //	    cout<<"Pedestals test= "<<test<<" sbd= "<<sub<<" depth= "<<k<<" eta= "<<i<<" , phi= "<<j<<endl;
            if (Map_Ampl[test][sub][k]->GetBinContent(i, j) > 0.9) {
            //AZ2023   if (Map_SUBGOOD[sub][k]->GetBinContent(i, j) != 1.0)   Map_SUBGOOD[sub][k]->SetBinContent(i, j, 0.15);
	//AZ2023	if (Map_ALL->GetBinContent(i, j) != 1.)	  Map_ALL->SetBinContent(i, j, 0.15);
            }
            //	    cout<<"Pedestals Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)<<endl;
          }  //end test
        }
      }
    }
  }
*/
  //AZ2023:
  /*
  std::cout << " RUN3: 2022 Plots with MAPS_SUB: start ..............................." << std::endl;
  // subdet maps
  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO

    std::cout << " RUN3: 2022 MAPS_SUB= " << sub << std::endl;
    //     if (sub==1) cHB->Divide(2,1);
    if (sub == 1)
      cHB->Divide(2, 2);
    //     if (sub==2) cHE->Divide(3,1);
    if (sub == 2)
      cHE->Divide(3, 3);
    if (sub == 3)
      cONE->Divide(1, 1);
    //     if (sub==4) cHB->Divide(2,1);
    if (sub == 4)
      cHF->Divide(2, 2);
    //     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
    //k = Depth
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      //          gStyle->SetTitleOffset(0.5, "Y");
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      Map_SUBGOOD[sub][k]->SetTitle(str);
      Map_SUBGOOD[sub][k]->SetXTitle("#eta \b");
      Map_SUBGOOD[sub][k]->SetYTitle("#phi \b");
      Map_SUBGOOD[sub][k]->Draw("COLZ");
      Map_SUBGOOD[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      Map_SUBGOOD[sub][k]->GetZaxis()->SetRangeUser(0., 1.);

      if (sub == 1) {
        cHB->Modified();
        cHB->Update();
      }
      if (sub == 2) {
        cHE->Modified();
        cHE->Update();
      }
      if (sub == 3) {
        cONE->Modified();
        cONE->Update();
      }
      if (sub == 4) {
        cHF->Modified();
        cHF->Update();
      }
    }  //end depth
    if (sub == 1) {
      cHB->Print("MAPHB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MAPHE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MAPHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MAPHF.png");
      cHF->Clear();
    }
  }  // end sub
*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  TCanvas *cmain1 = new TCanvas("cmain1", "cmain1", 200, 10, 1400, 1800);
  cmain1->Divide(2, 2);

  cmain1->cd(1);
  TH1F *JDBEYESJ0 = (TH1F *)dir->FindObjectAny("h_totalAmplitudeHBperEvent");
  JDBEYESJ0->SetStats(0);
  JDBEYESJ0->SetMarkerStyle(20);
  JDBEYESJ0->SetMarkerSize(0.8);
  JDBEYESJ0->GetYaxis()->SetLabelSize(0.04);
  JDBEYESJ0->SetXTitle("iEvent \b");
  JDBEYESJ0->SetYTitle("totalAmplitude perEvent \b");
  JDBEYESJ0->SetTitle("HB \b");
  JDBEYESJ0->SetMarkerColor(2);
  JDBEYESJ0->SetLineColor(1);
  JDBEYESJ0->SetMinimum(0.8);
  JDBEYESJ0->Draw("HIST same P0");
  //JDBEYESJ0->Clear();

  cmain1->cd(2);
  TH1F *JDBEYESJ1 = (TH1F *)dir->FindObjectAny("h_totalAmplitudeHEperEvent");
  JDBEYESJ1->SetStats(0);
  JDBEYESJ1->SetMarkerStyle(20);
  JDBEYESJ1->SetMarkerSize(0.8);
  JDBEYESJ1->GetYaxis()->SetLabelSize(0.04);
  JDBEYESJ1->SetXTitle("iEvent \b");
  JDBEYESJ1->SetYTitle("totalAmplitude perEvent \b");
  JDBEYESJ1->SetTitle("HE \b");
  JDBEYESJ1->SetMarkerColor(2);
  JDBEYESJ1->SetLineColor(1);
  JDBEYESJ1->SetMinimum(0.8);
  JDBEYESJ1->Draw("HIST same P0");
  //JDBEYESJ1->Clear();

  cmain1->cd(3);
  TH1F *JDBEYESJ2 = (TH1F *)dir->FindObjectAny("h_totalAmplitudeHFperEvent");
  JDBEYESJ2->SetStats(0);
  JDBEYESJ2->SetMarkerStyle(20);
  JDBEYESJ2->SetMarkerSize(0.8);
  JDBEYESJ2->GetYaxis()->SetLabelSize(0.04);
  JDBEYESJ2->SetXTitle("iEvent \b");
  JDBEYESJ2->SetYTitle("totalAmplitude perEvent \b");
  JDBEYESJ2->SetTitle("HF \b");
  JDBEYESJ2->SetMarkerColor(2);
  JDBEYESJ2->SetLineColor(1);
  JDBEYESJ2->SetMinimum(0.8);
  JDBEYESJ2->Draw("HIST same P0");
  //JDBEYESJ2->Clear();

  cmain1->cd(4);
  TH1F *JDBEYESJ3 = (TH1F *)dir->FindObjectAny("h_totalAmplitudeHOperEvent");
  JDBEYESJ3->SetStats(0);
  JDBEYESJ3->SetMarkerStyle(20);
  JDBEYESJ3->SetMarkerSize(0.8);
  JDBEYESJ3->GetYaxis()->SetLabelSize(0.04);
  JDBEYESJ3->SetXTitle("iEvent \b");
  JDBEYESJ3->SetYTitle("totalAmplitude perEvent \b");
  JDBEYESJ3->SetTitle("HO \b");
  JDBEYESJ3->SetMarkerColor(2);
  JDBEYESJ3->SetLineColor(1);
  JDBEYESJ3->SetMinimum(0.8);
  JDBEYESJ3->Draw("HIST same P0");
  //JDBEYESJ3->Clear();
  cmain1->Modified();
  cmain1->Update();
  cmain1->Print("EVENTDEPENDENCE.png");
  cmain1->Clear();
  //  std::cout << " EVENTDEPENDENCE " << std::endl;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //AZ2023:
  /*
  // ALL SubDet
  gStyle->SetOptTitle(0);
  TCanvas *cmain = new TCanvas("cmain", "MAP", 1000, 1000);
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Map_ALL->SetTitleOffset(1.3, "Y");
  Map_ALL->SetXTitle("#eta \b");
  Map_ALL->SetYTitle("#phi \b");
  Map_ALL->Draw("COLZ");
  Map_ALL->GetYaxis()->SetRangeUser(0, 72.);
  Map_ALL->GetZaxis()->SetRangeUser(0, 1.);
  cmain->Modified();
  cmain->Update();
  cmain->Print("MAP.png");
  cmain->Clear();
  std::cout << " MAP_ALL " << std::endl;
*/

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // 7. Correlation of Charge(=Amplitude) vs timing, fc (=TSN * 25 ns)
  // three plots -----------------------------------------------------------------

  TCanvas *corravstsn = new TCanvas("corravstsn", "corravstsn", 200, 10, 1400, 1800);
  // three plots for HB:
  corravstsn->Divide(2, 2);
  corravstsn->cd(1);
  TH2F *two11 = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two11->SetMarkerStyle(20);
  two11->SetMarkerSize(0.4);
  two11->SetYTitle("timing HB \b");
  two11->SetXTitle("Q,fc HB\b");
  two11->SetMarkerColor(1);
  two11->SetLineColor(1);
  two11->Draw("BOX");
  corravstsn->cd(2);
  TH1F *TSNvsQ_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HB->SetMarkerStyle(20);
  TSNvsQ_HB->SetMarkerSize(0.6);
  TSNvsQ_HB->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HB->SetXTitle("Q,fc HB \b");
  TSNvsQ_HB->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HB->SetMarkerColor(4);
  TSNvsQ_HB->SetLineColor(0);
  TSNvsQ_HB->SetMinimum(0.8);
  TSNvsQ_HB->Draw("E");
  corravstsn->cd(3);
  TH1F *twod1_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HB");
  TH1F *twod0_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HB");
  //  twod1_HB->Sumw2();
  //  twod0_HB->Sumw2();
  TH1F *Ceff_HB = (TH1F *)twod1_HB->Clone("Ceff_HB");
  for (int x = 1; x <= twod1_HB->GetXaxis()->GetNbins(); x++) {
    twod1_HB->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HB->Divide(twod1_HB, twod0_HB, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HB->SetMarkerStyle(20);
  Ceff_HB->SetMarkerSize(0.4);
  Ceff_HB->SetXTitle("Q,fc \b");
  Ceff_HB->SetYTitle("<timing>HB \b");
  Ceff_HB->SetMarkerColor(2);
  Ceff_HB->SetLineColor(2);
  Ceff_HB->SetMaximum(140.);
  Ceff_HB->SetMinimum(30.);
  Ceff_HB->Draw("E");
  corravstsn->Modified();
  corravstsn->Update();
  corravstsn->Print("corravstsnPLOTSHB.png");
  corravstsn->Clear();
  //  std::cout << " corravstsnPLOTSHB.png created " << std::endl;
  // three plots for HE:
  corravstsn->Divide(2, 2);
  corravstsn->cd(1);
  TH2F *twoHE = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  twoHE->SetMarkerStyle(20);
  twoHE->SetMarkerSize(0.4);
  twoHE->SetYTitle("timing HE \b");
  twoHE->SetXTitle("Q,fc HE\b");
  twoHE->SetMarkerColor(1);
  twoHE->SetLineColor(1);
  twoHE->Draw("BOX");
  corravstsn->cd(2);
  TH1F *TSNvsQ_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HE->SetMarkerStyle(20);
  TSNvsQ_HE->SetMarkerSize(0.6);
  TSNvsQ_HE->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HE->SetXTitle("Q,fc HE \b");
  TSNvsQ_HE->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HE->SetMarkerColor(4);
  TSNvsQ_HE->SetLineColor(0);
  TSNvsQ_HE->SetMinimum(0.8);
  TSNvsQ_HE->Draw("E");
  corravstsn->cd(3);
  TH1F *twod1_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HE");
  TH1F *twod0_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HE");
  //  twod1_HE->Sumw2();
  //  twod0_HE->Sumw2();
  TH1F *Ceff_HE = (TH1F *)twod1_HE->Clone("Ceff_HE");
  for (int x = 1; x <= twod1_HE->GetXaxis()->GetNbins(); x++) {
    twod1_HE->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HE->Divide(twod1_HE, twod0_HE, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HE->SetMarkerStyle(20);
  Ceff_HE->SetMarkerSize(0.4);
  Ceff_HE->SetXTitle("Q,fc \b");
  Ceff_HE->SetYTitle("<timing>HE \b");
  Ceff_HE->SetMarkerColor(2);
  Ceff_HE->SetLineColor(2);
  Ceff_HE->SetMaximum(150.);
  Ceff_HE->SetMinimum(25.);
  Ceff_HE->Draw("E");
  corravstsn->Modified();
  corravstsn->Update();
  corravstsn->Print("corravstsnPLOTSHE.png");
  corravstsn->Clear();
  //  std::cout << " corravstsnPLOTSHE.png created " << std::endl;
  // three plots for HF:
  corravstsn->Divide(2, 2);
  corravstsn->cd(1);
  TH2F *twoHF = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HF");
  gPad->SetGridy();
  gPad->SetGridx();
  twoHF->SetMarkerStyle(20);
  twoHF->SetMarkerSize(0.4);
  twoHF->SetYTitle("timing HF \b");
  twoHF->SetXTitle("Q,fc HF\b");
  twoHF->SetMarkerColor(1);
  twoHF->SetLineColor(1);
  twoHF->Draw("BOX");
  corravstsn->cd(2);
  TH1F *TSNvsQ_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HF");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HF->SetMarkerStyle(20);
  TSNvsQ_HF->SetMarkerSize(0.6);
  TSNvsQ_HF->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HF->SetXTitle("Q,fc HF \b");
  TSNvsQ_HF->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HF->SetMarkerColor(4);
  TSNvsQ_HF->SetLineColor(0);
  TSNvsQ_HF->SetMinimum(0.8);
  TSNvsQ_HF->Draw("E");
  corravstsn->cd(3);
  TH1F *twod1_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HF");
  TH1F *twod0_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HF");
  //  twod1_HF->Sumw2();
  //  twod0_HF->Sumw2();
  TH1F *Ceff_HF = (TH1F *)twod1_HF->Clone("Ceff_HF");
  for (int x = 1; x <= twod1_HF->GetXaxis()->GetNbins(); x++) {
    twod1_HF->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HF->Divide(twod1_HF, twod0_HF, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HF->SetMarkerStyle(20);
  Ceff_HF->SetMarkerSize(0.4);
  Ceff_HF->SetXTitle("Q,fc \b");
  Ceff_HF->SetYTitle("<timing>HF \b");
  Ceff_HF->SetMarkerColor(2);
  Ceff_HF->SetLineColor(2);
  Ceff_HF->SetMaximum(50.);
  Ceff_HF->SetMinimum(0.);
  Ceff_HF->Draw("E");
  corravstsn->Modified();
  corravstsn->Update();
  corravstsn->Print("corravstsnPLOTSHF.png");
  corravstsn->Clear();
  //  std::cout << " corravstsnPLOTSHF.png created " << std::endl;
  // three plots for HO:
  corravstsn->Divide(2, 2);
  corravstsn->cd(1);
  TH2F *twoHO = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HO");
  gPad->SetGridy();
  gPad->SetGridx();
  twoHO->SetMarkerStyle(20);
  twoHO->SetMarkerSize(0.4);
  twoHO->SetYTitle("timing HO \b");
  twoHO->SetXTitle("Q,fc HO\b");
  twoHO->SetMarkerColor(1);
  twoHO->SetLineColor(1);
  twoHO->Draw("BOX");
  corravstsn->cd(2);
  TH1F *TSNvsQ_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HO");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HO->SetMarkerStyle(20);
  TSNvsQ_HO->SetMarkerSize(0.6);
  TSNvsQ_HO->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HO->SetXTitle("Q,fc HO \b");
  TSNvsQ_HO->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HO->SetMarkerColor(4);
  TSNvsQ_HO->SetLineColor(0);
  TSNvsQ_HO->SetMinimum(0.8);
  TSNvsQ_HO->Draw("E");
  corravstsn->cd(3);
  TH1F *twod1_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HO");
  TH1F *twod0_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HO");
  //  twod1_HO->Sumw2();
  //  twod0_HO->Sumw2();
  TH1F *Ceff_HO = (TH1F *)twod1_HO->Clone("Ceff_HO");
  for (int x = 1; x <= twod1_HO->GetXaxis()->GetNbins(); x++) {
    twod1_HO->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HO->Divide(twod1_HO, twod0_HO, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HO->SetMarkerStyle(20);
  Ceff_HO->SetMarkerSize(0.4);
  Ceff_HO->SetXTitle("Q,fc \b");
  Ceff_HO->SetYTitle("<timing>HO \b");
  Ceff_HO->SetMarkerColor(2);
  Ceff_HO->SetLineColor(2);
  Ceff_HO->SetMaximum(150.);
  Ceff_HO->SetMinimum(70.);
  Ceff_HO->Draw("E");
  corravstsn->Modified();
  corravstsn->Update();
  corravstsn->Print("corravstsnPLOTSHO.png");
  corravstsn->Clear();
  //  std::cout << " corravstsnPLOTSHO.png created " << std::endl;

  // 2D plots (from 1 to 7) <TSn>    -----------------------------------------------------------------
  TCanvas *cHBnew = new TCanvas("cHBnew", "cHBnew", 0, 10, 1400, 1800);
  // 4 plots for HB:
  cHBnew->Clear();
  cHBnew->Divide(2, 2);
  cHBnew->cd(1);
  TH2F *dva1_HBDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HB");
  TH2F *dva0_HBDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  //  dva1_HBDepth1->Sumw2();
  //  dva0_HBDepth1->Sumw2();
  TH2F *Seff_HBDepth1 = (TH2F *)dva1_HBDepth1->Clone("Seff_HBDepth1");
  Seff_HBDepth1->Divide(dva1_HBDepth1, dva0_HBDepth1, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HBDepth1->SetMarkerStyle(20);
  Seff_HBDepth1->SetMarkerSize(0.4);
  Seff_HBDepth1->SetXTitle("#eta \b");
  Seff_HBDepth1->SetYTitle("#phi \b");
  Seff_HBDepth1->SetZTitle("<timing> HB Depth1 \b");
  Seff_HBDepth1->SetMarkerColor(2);
  Seff_HBDepth1->SetLineColor(2);
  Seff_HBDepth1->SetMaximum(100.);
  Seff_HBDepth1->SetMinimum(80.);
  Seff_HBDepth1->Draw("COLZ");
  cHBnew->cd(2);
  TH2F *dva1_HBDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HB");
  TH2F *dva0_HBDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  TH2F *Seff_HBDepth2 = (TH2F *)dva1_HBDepth2->Clone("Seff_HBDepth2");
  Seff_HBDepth2->Divide(dva1_HBDepth2, dva0_HBDepth2, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HBDepth2->SetMarkerStyle(20);
  Seff_HBDepth2->SetMarkerSize(0.4);
  Seff_HBDepth2->SetXTitle("#eta \b");
  Seff_HBDepth2->SetYTitle("#phi \b");
  Seff_HBDepth2->SetZTitle("<timing> HB Depth2 \b");
  Seff_HBDepth2->SetMarkerColor(2);
  Seff_HBDepth2->SetLineColor(2);
  Seff_HBDepth2->SetMaximum(100.);
  Seff_HBDepth2->SetMinimum(80.);
  Seff_HBDepth2->Draw("COLZ");
  cHBnew->cd(3);
  TH2F *dva1_HBDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HB");
  TH2F *dva0_HBDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HB");
  TH2F *Seff_HBDepth3 = (TH2F *)dva1_HBDepth3->Clone("Seff_HBDepth3");
  Seff_HBDepth3->Divide(dva1_HBDepth3, dva0_HBDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HBDepth3->SetMarkerStyle(20);
  Seff_HBDepth3->SetMarkerSize(0.4);
  Seff_HBDepth3->SetXTitle("#eta \b");
  Seff_HBDepth3->SetYTitle("#phi \b");
  Seff_HBDepth3->SetZTitle("<timing> HB Depth3 \b");
  Seff_HBDepth3->SetMarkerColor(2);
  Seff_HBDepth3->SetLineColor(2);
  Seff_HBDepth3->SetMaximum(100.);
  Seff_HBDepth3->SetMinimum(80.);
  Seff_HBDepth3->Draw("COLZ");
  cHBnew->cd(4);
  TH2F *dva1_HBDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HB");
  TH2F *dva0_HBDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HB");
  TH2F *Seff_HBDepth4 = (TH2F *)dva1_HBDepth4->Clone("Seff_HBDepth4");
  Seff_HBDepth4->Divide(dva1_HBDepth4, dva0_HBDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HBDepth4->SetMarkerStyle(20);
  Seff_HBDepth4->SetMarkerSize(0.4);
  Seff_HBDepth4->SetXTitle("#eta \b");
  Seff_HBDepth4->SetYTitle("#phi \b");
  Seff_HBDepth4->SetZTitle("<timing> HB Depth4 \b");
  Seff_HBDepth4->SetMarkerColor(2);
  Seff_HBDepth4->SetLineColor(2);
  Seff_HBDepth4->SetMaximum(100.);
  Seff_HBDepth4->SetMinimum(80.);
  Seff_HBDepth4->Draw("COLZ");
  cHBnew->Modified();
  cHBnew->Update();
  cHBnew->Print("2DcorravstsnPLOTSHB.png");
  cHBnew->Clear();
  //  std::cout << " 2DcorravstsnPLOTSHB.png created " << std::endl;
  // clean-up
  if (dva1_HBDepth1)
    delete dva1_HBDepth1;
  if (dva0_HBDepth1)
    delete dva0_HBDepth1;
  if (Seff_HBDepth1)
    delete Seff_HBDepth1;
  if (dva1_HBDepth2)
    delete dva1_HBDepth2;
  if (dva0_HBDepth2)
    delete dva0_HBDepth2;
  if (Seff_HBDepth2)
    delete Seff_HBDepth2;
  if (dva1_HBDepth3)
    delete dva1_HBDepth3;
  if (dva0_HBDepth3)
    delete dva0_HBDepth3;
  if (Seff_HBDepth3)
    delete Seff_HBDepth3;
  if (dva1_HBDepth4)
    delete dva1_HBDepth4;
  if (dva0_HBDepth4)
    delete dva0_HBDepth4;
  if (Seff_HBDepth4)
    delete Seff_HBDepth4;

  // 7 plots for HE:
  TCanvas *cHEnew = new TCanvas("cHEnew", "cHEnew", 5, 10, 1400, 1800);
  cHEnew->Clear();
  cHEnew->Divide(2, 4);
  cHEnew->cd(1);
  TH2F *dva1_HEDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HE");
  TH2F *dva0_HEDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  //  dva1_HEDepth1->Sumw2();
  //  dva0_HEDepth1->Sumw2();
  TH2F *Seff_HEDepth1 = (TH2F *)dva1_HEDepth1->Clone("Seff_HEDepth1");
  Seff_HEDepth1->Divide(dva1_HEDepth1, dva0_HEDepth1, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth1->SetMarkerStyle(20);
  Seff_HEDepth1->SetMarkerSize(0.4);
  Seff_HEDepth1->SetXTitle("#eta \b");
  Seff_HEDepth1->SetYTitle("#phi \b");
  Seff_HEDepth1->SetZTitle("<timing> HE Depth1 \b");
  Seff_HEDepth1->SetMarkerColor(2);
  Seff_HEDepth1->SetLineColor(2);
  Seff_HEDepth1->SetMaximum(100.);
  Seff_HEDepth1->SetMinimum(80.);
  Seff_HEDepth1->Draw("COLZ");
  cHEnew->cd(2);
  TH2F *dva1_HEDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HE");
  TH2F *dva0_HEDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  TH2F *Seff_HEDepth2 = (TH2F *)dva1_HEDepth2->Clone("Seff_HEDepth2");
  Seff_HEDepth2->Divide(dva1_HEDepth2, dva0_HEDepth2, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth2->SetMarkerStyle(20);
  Seff_HEDepth2->SetMarkerSize(0.4);
  Seff_HEDepth2->SetXTitle("#eta \b");
  Seff_HEDepth2->SetYTitle("#phi \b");
  Seff_HEDepth2->SetZTitle("<timing> HE Depth2 \b");
  Seff_HEDepth2->SetMarkerColor(2);
  Seff_HEDepth2->SetLineColor(2);
  Seff_HEDepth2->SetMaximum(100.);
  Seff_HEDepth2->SetMinimum(80.);
  Seff_HEDepth2->Draw("COLZ");
  cHEnew->cd(3);
  TH2F *dva1_HEDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HE");
  TH2F *dva0_HEDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  TH2F *Seff_HEDepth3 = (TH2F *)dva1_HEDepth3->Clone("Seff_HEDepth3");
  Seff_HEDepth3->Divide(dva1_HEDepth3, dva0_HEDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth3->SetMarkerStyle(20);
  Seff_HEDepth3->SetMarkerSize(0.4);
  Seff_HEDepth3->SetXTitle("#eta \b");
  Seff_HEDepth3->SetYTitle("#phi \b");
  Seff_HEDepth3->SetZTitle("<timing> HE Depth3 \b");
  Seff_HEDepth3->SetMarkerColor(2);
  Seff_HEDepth3->SetLineColor(2);
  Seff_HEDepth3->SetMaximum(100.);
  Seff_HEDepth3->SetMinimum(80.);
  Seff_HEDepth3->Draw("COLZ");
  cHEnew->cd(4);
  TH2F *dva1_HEDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HE");
  TH2F *dva0_HEDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HE");
  TH2F *Seff_HEDepth4 = (TH2F *)dva1_HEDepth4->Clone("Seff_HEDepth4");
  Seff_HEDepth4->Divide(dva1_HEDepth4, dva0_HEDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth4->SetMarkerStyle(20);
  Seff_HEDepth4->SetMarkerSize(0.4);
  Seff_HEDepth4->SetXTitle("#eta \b");
  Seff_HEDepth4->SetYTitle("#phi \b");
  Seff_HEDepth4->SetZTitle("<timing> HE Depth4 \b");
  Seff_HEDepth4->SetMarkerColor(2);
  Seff_HEDepth4->SetLineColor(2);
  Seff_HEDepth4->SetMaximum(100.);
  Seff_HEDepth4->SetMinimum(80.);
  Seff_HEDepth4->Draw("COLZ");
  cHEnew->cd(5);
  TH2F *dva1_HEDepth5 = (TH2F *)dir->FindObjectAny("h_mapDepth5TSmeanA_HE");
  TH2F *dva0_HEDepth5 = (TH2F *)dir->FindObjectAny("h_mapDepth5_HE");
  TH2F *Seff_HEDepth5 = (TH2F *)dva1_HEDepth5->Clone("Seff_HEDepth5");
  Seff_HEDepth5->Divide(dva1_HEDepth5, dva0_HEDepth5, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth5->SetMarkerStyle(20);
  Seff_HEDepth5->SetMarkerSize(0.4);
  Seff_HEDepth5->SetXTitle("#eta \b");
  Seff_HEDepth5->SetYTitle("#phi \b");
  Seff_HEDepth5->SetZTitle("<timing> HE Depth5 \b");
  Seff_HEDepth5->SetMarkerColor(2);
  Seff_HEDepth5->SetLineColor(2);
  Seff_HEDepth5->SetMaximum(100.);
  Seff_HEDepth5->SetMinimum(80.);
  Seff_HEDepth5->Draw("COLZ");
  cHEnew->cd(6);
  TH2F *dva1_HEDepth6 = (TH2F *)dir->FindObjectAny("h_mapDepth6TSmeanA_HE");
  TH2F *dva0_HEDepth6 = (TH2F *)dir->FindObjectAny("h_mapDepth6_HE");
  TH2F *Seff_HEDepth6 = (TH2F *)dva1_HEDepth6->Clone("Seff_HEDepth6");
  Seff_HEDepth6->Divide(dva1_HEDepth6, dva0_HEDepth6, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth6->SetMarkerStyle(20);
  Seff_HEDepth6->SetMarkerSize(0.4);
  Seff_HEDepth6->SetXTitle("#eta \b");
  Seff_HEDepth6->SetYTitle("#phi \b");
  Seff_HEDepth6->SetZTitle("<timing> HE Depth6 \b");
  Seff_HEDepth6->SetMarkerColor(2);
  Seff_HEDepth6->SetLineColor(2);
  Seff_HEDepth6->SetMaximum(100.);
  Seff_HEDepth6->SetMinimum(80.);
  Seff_HEDepth6->Draw("COLZ");
  cHEnew->cd(7);
  TH2F *dva1_HEDepth7 = (TH2F *)dir->FindObjectAny("h_mapDepth7TSmeanA_HE");
  TH2F *dva0_HEDepth7 = (TH2F *)dir->FindObjectAny("h_mapDepth7_HE");
  TH2F *Seff_HEDepth7 = (TH2F *)dva1_HEDepth7->Clone("Seff_HEDepth7");
  Seff_HEDepth7->Divide(dva1_HEDepth7, dva0_HEDepth7, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HEDepth7->SetMarkerStyle(20);
  Seff_HEDepth7->SetMarkerSize(0.4);
  Seff_HEDepth7->SetXTitle("#eta \b");
  Seff_HEDepth7->SetYTitle("#phi \b");
  Seff_HEDepth7->SetZTitle("<timing> HE Depth7 \b");
  Seff_HEDepth7->SetMarkerColor(2);
  Seff_HEDepth7->SetLineColor(2);
  Seff_HEDepth7->SetMaximum(100.);
  Seff_HEDepth7->SetMinimum(80.);
  Seff_HEDepth7->Draw("COLZ");
  cHEnew->Modified();
  cHEnew->Update();
  cHEnew->Print("2DcorravstsnPLOTSHE.png");
  cHEnew->Clear();
  //  std::cout << " 2DcorravstsnPLOTSHE.png created " << std::endl;
  // clean-up
  if (dva1_HEDepth1)
    delete dva1_HEDepth1;
  if (dva0_HEDepth1)
    delete dva0_HEDepth1;
  if (Seff_HEDepth1)
    delete Seff_HEDepth1;
  if (dva1_HEDepth2)
    delete dva1_HEDepth2;
  if (dva0_HEDepth2)
    delete dva0_HEDepth2;
  if (Seff_HEDepth2)
    delete Seff_HEDepth2;
  if (dva1_HEDepth3)
    delete dva1_HEDepth3;
  if (dva0_HEDepth3)
    delete dva0_HEDepth3;
  if (Seff_HEDepth3)
    delete Seff_HEDepth3;
  if (dva1_HEDepth4)
    delete dva1_HEDepth4;
  if (dva0_HEDepth4)
    delete dva0_HEDepth4;
  if (Seff_HEDepth4)
    delete Seff_HEDepth4;
  if (dva1_HEDepth5)
    delete dva1_HEDepth5;
  if (dva0_HEDepth5)
    delete dva0_HEDepth5;
  if (Seff_HEDepth5)
    delete Seff_HEDepth5;
  if (dva1_HEDepth6)
    delete dva1_HEDepth6;
  if (dva0_HEDepth6)
    delete dva0_HEDepth6;
  if (Seff_HEDepth6)
    delete Seff_HEDepth6;
  if (dva1_HEDepth7)
    delete dva1_HEDepth7;
  if (dva0_HEDepth7)
    delete dva0_HEDepth7;
  if (Seff_HEDepth7)
    delete Seff_HEDepth7;
  // 4 plots for HF:
  TCanvas *cHFnew = new TCanvas("cHFnew", "cHFnew", 200, 10, 1400, 1800);
  cHFnew->Clear();
  cHFnew->Divide(2, 2);
  cHFnew->cd(1);
  TH2F *dva1_HFDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HF");
  TH2F *dva0_HFDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  //  dva1_HFDepth1->Sumw2();
  //  dva0_HFDepth1->Sumw2();
  TH2F *Seff_HFDepth1 = (TH2F *)dva1_HFDepth1->Clone("Seff_HFDepth1");
  Seff_HFDepth1->Divide(dva1_HFDepth1, dva0_HFDepth1, 25., 1., "B");
  /*
    for (int i=1;i<=Seff_HFDepth1->GetXaxis()->GetNbins();i++) {
      for (int j=1;j<=Seff_HFDepth1->GetYaxis()->GetNbins();j++) {
	  double ccc1 =  Seff_HFDepth1->GetBinContent(i,j)   ;
	  //	  if(ccc1 >  0.) std::cout << "********************   i =  " << i  << " j =  " << j  << " ccc1 =  " << ccc1 << std::endl;

	  Seff_HFDepth1->SetBinContent(i,j,0.);
	  if(ccc1 >  0.)  Seff_HFDepth1->SetBinContent(i,j,ccc1);
      }
    }
*/
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HFDepth1->SetMarkerStyle(20);
  Seff_HFDepth1->SetMarkerSize(0.4);
  Seff_HFDepth1->SetXTitle("#eta \b");
  Seff_HFDepth1->SetYTitle("#phi \b");
  Seff_HFDepth1->SetZTitle("<timing> HF Depth1 \b");
  Seff_HFDepth1->SetMarkerColor(2);
  Seff_HFDepth1->SetLineColor(2);
  Seff_HFDepth1->SetMaximum(50.);
  Seff_HFDepth1->SetMinimum(20.);
  Seff_HFDepth1->Draw("COLZ");
  cHFnew->cd(2);
  TH2F *dva1_HFDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HF");
  TH2F *dva0_HFDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  TH2F *Seff_HFDepth2 = (TH2F *)dva1_HFDepth2->Clone("Seff_HFDepth2");
  Seff_HFDepth2->Divide(dva1_HFDepth2, dva0_HFDepth2, 25., 1., "B");
  /*
    for (int i=1;i<=Seff_HFDepth2->GetXaxis()->GetNbins();i++) {
      for (int j=1;j<=Seff_HFDepth2->GetYaxis()->GetNbins();j++) {
	  double ccc1 =  Seff_HFDepth2->GetBinContent(i,j)   ;
	  Seff_HFDepth2->SetBinContent(i,j,0.);
	  if(ccc1 >  0. )  Seff_HFDepth2->SetBinContent(i,j,ccc1);
      }
    }
  */
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HFDepth2->SetMarkerStyle(20);
  Seff_HFDepth2->SetMarkerSize(0.4);
  Seff_HFDepth2->SetXTitle("#eta \b");
  Seff_HFDepth2->SetYTitle("#phi \b");
  Seff_HFDepth2->SetZTitle("<timing> HF Depth2 \b");
  Seff_HFDepth2->SetMarkerColor(2);
  Seff_HFDepth2->SetLineColor(2);
  Seff_HFDepth2->SetMaximum(50.);
  Seff_HFDepth2->SetMinimum(20.);
  Seff_HFDepth2->Draw("COLZ");
  cHFnew->cd(3);
  TH2F *dva1_HFDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HF");
  TH2F *dva0_HFDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HF");
  TH2F *Seff_HFDepth3 = (TH2F *)dva1_HFDepth3->Clone("Seff_HFDepth3");
  Seff_HFDepth3->Divide(dva1_HFDepth3, dva0_HFDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HFDepth3->SetMarkerStyle(20);
  Seff_HFDepth3->SetMarkerSize(0.4);
  Seff_HFDepth3->SetXTitle("#eta \b");
  Seff_HFDepth3->SetYTitle("#phi \b");
  Seff_HFDepth3->SetZTitle("<timing> HF Depth3 \b");
  Seff_HFDepth3->SetMarkerColor(2);
  Seff_HFDepth3->SetLineColor(2);
  Seff_HFDepth3->SetMaximum(50.);
  Seff_HFDepth3->SetMinimum(20.);
  Seff_HFDepth3->Draw("COLZ");
  cHFnew->cd(4);
  TH2F *dva1_HFDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HF");
  TH2F *dva0_HFDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HF");
  TH2F *Seff_HFDepth4 = (TH2F *)dva1_HFDepth4->Clone("Seff_HFDepth4");
  Seff_HFDepth4->Divide(dva1_HFDepth4, dva0_HFDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HFDepth4->SetMarkerStyle(20);
  Seff_HFDepth4->SetMarkerSize(0.4);
  Seff_HFDepth4->SetXTitle("#eta \b");
  Seff_HFDepth4->SetYTitle("#phi \b");
  Seff_HFDepth4->SetZTitle("<timing> HF Depth4 \b");
  Seff_HFDepth4->SetMarkerColor(2);
  Seff_HFDepth4->SetLineColor(2);
  Seff_HFDepth4->SetMaximum(50.);
  Seff_HFDepth4->SetMinimum(20.);
  Seff_HFDepth4->Draw("COLZ");
  cHFnew->Modified();
  cHFnew->Update();
  cHFnew->Print("2DcorravstsnPLOTSHF.png");
  cHFnew->Clear();
  //  std::cout << " 2DcorravstsnPLOTSHF.png created " << std::endl;
  if (dva1_HFDepth1)
    delete dva1_HFDepth1;
  if (dva0_HFDepth1)
    delete dva0_HFDepth1;
  if (Seff_HFDepth1)
    delete Seff_HFDepth1;
  if (dva1_HFDepth2)
    delete dva1_HFDepth2;
  if (dva0_HFDepth2)
    delete dva0_HFDepth2;
  if (Seff_HFDepth2)
    delete Seff_HFDepth2;
  if (dva1_HFDepth3)
    delete dva1_HFDepth3;
  if (dva0_HFDepth3)
    delete dva0_HFDepth3;
  if (Seff_HFDepth3)
    delete Seff_HFDepth3;
  if (dva1_HFDepth4)
    delete dva1_HFDepth4;
  if (dva0_HFDepth4)
    delete dva0_HFDepth4;
  if (Seff_HFDepth4)
    delete Seff_HFDepth4;
  // 1 plot for HO:
  TCanvas *cHOnew = new TCanvas("cHOnew", "cHOnew", 200, 10, 1400, 1800);
  //  TCanvas *cHOnew = new TCanvas("cHOnew", "cHOnew", 1500, 500);
  cHOnew->Clear();
  cHOnew->Divide(1, 1);
  cHOnew->cd(1);
  TH2F *dva1_HODepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HO");
  /*
  for (int i=1;i<=dva1_HODepth4->GetXaxis()->GetNbins();i++) {
    for (int j=1;j<=dva1_HODepth4->GetYaxis()->GetNbins();j++) {
      double ccc1 =  dva1_HODepth4->GetBinContent(i,j)   ;
      	  if(ccc1 >  0.) std::cout << "******    dva1_HODepth4   **************   i =  " << i  << " j =  " << j  << " ccc1 =  " << ccc1 << std::endl;
    }
  }
*/
  TH2F *dva0_HODepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  /*
  for (int i=1;i<=dva0_HODepth4->GetXaxis()->GetNbins();i++) {
    for (int j=1;j<=dva0_HODepth4->GetYaxis()->GetNbins();j++) {
      double ccc1 =  dva0_HODepth4->GetBinContent(i,j)   ;
      	  if(ccc1 >  0.) std::cout << "******   dva0_HODepth4   **************   i =  " << i  << " j =  " << j  << " ccc1 =  " << ccc1 << std::endl;
    }
  }
 */
  TH2F *Seff_HODepth4 = (TH2F *)dva1_HODepth4->Clone("Seff_HODepth4");
  /*
  for (int x = 1; x <= Seff_HODepth4->GetXaxis()->GetNbins(); x++) {
    for (int y = 1; y <= Seff_HODepth4->GetYaxis()->GetNbins(); y++) {
      //      dva1_HODepth4->SetBinError(float(x), float(y), 0.001);
      //      Seff_HODepth4->SetBinContent(float(x), float(y), 0.0);
    }    //end x
  }    //end y
*/
  Seff_HODepth4->Divide(dva1_HODepth4, dva0_HODepth4, 25., 1., "B");
  /*
  for (int i=1;i<=Seff_HODepth4->GetXaxis()->GetNbins();i++) {
    for (int j=1;j<=Seff_HODepth4->GetYaxis()->GetNbins();j++) {
      double ccc1 =  Seff_HODepth4->GetBinContent(i,j);
      if(ccc1 >  0.) std::cout << "******    Seff_HODepth4   **************   i =  " << i  << " j =  " << j  << " ccc1 =  " << ccc1 << std::endl;
    }
  }
*/
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HODepth4->SetMarkerStyle(20);
  Seff_HODepth4->SetMarkerSize(0.4);
  Seff_HODepth4->SetXTitle("#eta \b");
  Seff_HODepth4->SetYTitle("#phi \b");
  Seff_HODepth4->SetZTitle("<timing> HO Depth4 \b");
  Seff_HODepth4->SetMarkerColor(2);
  Seff_HODepth4->SetLineColor(2);
  Seff_HODepth4->SetMaximum(130.);
  Seff_HODepth4->SetMinimum(70.);
  Seff_HODepth4->Draw("COLZ");
  cHOnew->Modified();
  cHOnew->Update();
  cHOnew->Print("2DcorravstsnPLOTSHO.png");
  cHOnew->Clear();
  //  std::cout << " 2DcorravstsnPLOTSHO.png created " << std::endl;
  if (dva1_HODepth4)
    delete dva1_HODepth4;
  if (dva0_HODepth4)
    delete dva0_HODepth4;
  if (Seff_HODepth4)
    delete Seff_HODepth4;

  //  std::cout << " END OF 2023 " << std::endl;
  //
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //======================================================================

  //====================================================================== html pages  CREATING:
  std::cout << " html pages  CREATING: " << std::endl;
  //======================================================================
  // Creating each test kind for each subdet html pages:
  std::string raw_class, raw_class1, raw_class2, raw_class3;
  int ind = 0;

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
    ofstream htmlFileT, htmlFileC, htmlFileD, htmlFileP, htmlFileS;
    if (sub == 1) {
      htmlFileT.open("HB_Tile.html");
      htmlFileC.open("HB_Calib.html");
      htmlFileD.open("HB_Drift.html");
      htmlFileP.open("HB_Pedestals.html");
      htmlFileS.open("HB_Shapes.html");
    }
    if (sub == 2) {
      htmlFileT.open("HE_Tile.html");
      htmlFileC.open("HE_Calib.html");
      htmlFileD.open("HE_Drift.html");
      htmlFileP.open("HE_Pedestals.html");
      htmlFileS.open("HE_Shapes.html");
    }
    if (sub == 3) {
      htmlFileT.open("HO_Tile.html");
      htmlFileC.open("HO_Calib.html");
      htmlFileD.open("HO_Drift.html");
      htmlFileP.open("HO_Pedestals.html");
      htmlFileS.open("HO_Shapes.html");
    }
    if (sub == 4) {
      htmlFileT.open("HF_Tile.html");
      htmlFileC.open("HF_Calib.html");
      htmlFileD.open("HF_Drift.html");
      htmlFileP.open("HF_Pedestals.html");
      htmlFileS.open("HF_Shapes.html");
    }

    // Megatile channels
    htmlFileT << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileT << "<head>" << std::endl;
    htmlFileT << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileT << "<title> Remote Monitoring Tool Global</title>" << std::endl;
    htmlFileT << "<style type=\"text/css\">" << std::endl;
    htmlFileT << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileT << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileT << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileT << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileT << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileT << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileT << "</style>" << std::endl;
    htmlFileT << "<body>" << std::endl;

    if (sub == 1)
      htmlFileT << "<h1> Criteria for megatile channels for HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileT << "<h1> Criteria for megatile channels for HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileT << "<h1> Criteria for megatile channels for HO, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileT << "<h1> Criteria for megatile channels for HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Test Entries

    htmlFileT << "<h2> 0. Entries for each channel.</h3>" << std::endl;
    htmlFileT << "<h3> 0.A. Entries in each channel for each depth.</h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: color is number of hits in digi collection </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateEntryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateEntryHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateEntryHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateEntryHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Test Cm
    htmlFileT << "<h2> 1. Cm criterion: CapID errors for each channel.</h3>" << std::endl;
    htmlFileT << "<h3> 1.A. Rate of CapId failures in each channel for each depth.</h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateCapIDHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateCapIDHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateCapIDHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateCapIDHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Am
    htmlFileT << "<h2> 2. Am criterion: ADC amplitude collected over all TSs(Full Amplitude) for each channel. </h3>"
              << std::endl;
    htmlFileT << "<h3> 2.A. Full ADC amplitude distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileT << "<h4> Legend: Bins less " << MIN_M[2][sub] << " correpond to bad ADC amplitude </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"HistAmplHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"HistAmplHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"HistAmplHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"HistAmplHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;
    htmlFileT << "<h3> 2.B. Rate of bad ADC amplitude (<" << MIN_M[2][sub] << ") in each channel for each depth. </h3>"
              << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colours - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateAmplHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateAmplHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateAmplHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateAmplHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Test Wm
    htmlFileT << "<h2> 3. Wm criterion: RMS (width) of ADC amplutude for each channel.</h3>" << std::endl;
    htmlFileT << "<h3> 3.A. RMS distribution over all events, channel and depth.</h3>" << std::endl;
    htmlFileT << "<h4> Legend: Bins less " << MIN_M[3][sub] << " and more " << MAX_M[3][sub]
              << " correpond to bad RMS </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"HistRMSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"HistRMSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"HistRMSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"HistRMSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;
    htmlFileT << "<h3> 3.B. Rate of bad RMS (<" << MIN_M[3][sub] << ",>" << MAX_M[3][sub]
              << ") in each channel for each depth.</h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateRMSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateRMSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateRMSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateRMSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Rm
    htmlFileT << "<h2> 4. Rm criterion: Ratio ADC value sum over four near maximum (-2, -1, max, +1) TS to ADC value "
                 "sum over all TS for each channel. </h3>"
              << std::endl;
    htmlFileT << "<h3> 4.A. Ratio distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileT << "<h4> Legend: Bins less " << MIN_M[4][sub] << " and more " << MAX_M[4][sub]
              << " correpond to bad ratio </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"Hist43TStoAllTSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"Hist43TStoAllTSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"Hist43TStoAllTSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"Hist43TStoAllTSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;
    htmlFileT << "<h3> 4.B. Rate of bad ratio (<" << MIN_M[4][sub] << ", >" << MAX_M[4][sub]
              << ") in each channel for each depth.</h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRate43TStoAllTSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRate43TStoAllTSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRate43TStoAllTSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRate43TStoAllTSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // TNm
    htmlFileT << "<h2> 5. TNm criterion: Mean TS position for each channel.</h3>" << std::endl;
    htmlFileT << "<h3> 5.A. TN position distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileT << "<h4> Legend: Bins less " << MIN_M[5][sub] << " and more " << MAX_M[5][sub]
              << " correpond to bad mean position </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"HistMeanPosHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"HistMeanPosHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"HistMeanPosHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"HistMeanPosHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;
    htmlFileT << "<h3> 5.B. Rate of bad TN position  (<" << MIN_M[5][sub] << ", >" << MAX_M[5][sub]
              << ") in each channel for each depth. </h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateMeanPosHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateMeanPosHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateMeanPosHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateMeanPosHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // TXm
    htmlFileT << "<h2> 6.TXm criterion: Maximum TS position for each channel.</h3>" << std::endl;
    htmlFileT << "<h3> 6.A. TX position distribution over all events, channel and depth.</h3>" << std::endl;
    htmlFileT << "<h4> Legend: Bins less " << MIN_M[6][sub] << " and more " << MAX_M[6][sub]
              << " correpond to bad position </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"HistMaxPosHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"HistMaxPosHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"HistMaxPosHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"HistMaxPosHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;
    htmlFileT << "<h3> 6.B. Rate of bad TX position  (<" << MIN_M[6][sub] << ", >" << MAX_M[6][sub]
              << ") in each channel for each depth. </h3>" << std::endl;
    htmlFileT << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"MapRateMaxPosHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"MapRateMaxPosHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"MapRateMaxPosHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"MapRateMaxPosHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    // Correlation of A vs TSn done in 2023 for Run3 and for GlobalRMT only
    htmlFileT << "<h2> 7....... Correlation of A(=Q) vs timing(=25ns*MeanTSposition) </h3>" << std::endl;

    htmlFileT << "<h3> 7.A..... 1)2D-correlation of timing vs Q,fc;.......... 2)Q,fc;................ 3)mean timing vs "
                 "Q,fc .......  </h3>"
              << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"corravstsnPLOTSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"corravstsnPLOTSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"corravstsnPLOTSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"corravstsnPLOTSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    htmlFileT << "<h3> 7.B....... Mean timing in 2D space of eta-phi for different Depthes........ </h3>" << std::endl;
    if (sub == 1)
      htmlFileT << " <img src=\"2DcorravstsnPLOTSHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileT << " <img src=\"2DcorravstsnPLOTSHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileT << " <img src=\"2DcorravstsnPLOTSHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileT << " <img src=\"2DcorravstsnPLOTSHF.png\" />" << std::endl;
    htmlFileT << "<br>" << std::endl;

    htmlFileT << "</body> " << std::endl;
    htmlFileT << "</html> " << std::endl;
    htmlFileT.close();

    // Pedestals
    htmlFileP << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileP << "<head>" << std::endl;
    htmlFileP << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileP << "<title> Remote Monitoring Tool Global</title>" << std::endl;
    htmlFileP << "<style type=\"text/css\">" << std::endl;
    htmlFileP << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileP << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileP << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileP << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileP << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileP << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileP << "</style>" << std::endl;
    htmlFileP << "<body>" << std::endl;

    if (sub == 1)
      htmlFileP << "<h1> Pedestals for HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileP << "<h1> Pedestals for HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileP << "<h1> Pedestals for HO, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileP << "<h1> Pedestals for HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileP << "<br>" << std::endl;

    // Pedestal:
    htmlFileP << "<h2> 1.Pm criterion: Pedestals for each CapID .</h3>" << std::endl;
    htmlFileP << "<h3> 1.A. Pedestal distribution over all events, channels for each CapID and all depths.</h3>"
              << std::endl;
    htmlFileP << "<h4> Legend: Bins less " << Pedest[0][sub] << " correpond to bad Pedestals </h4>" << std::endl;
    if (sub == 1)
      htmlFileP << " <img src=\"HistPedestalsHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileP << " <img src=\"HistPedestalsHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileP << " <img src=\"HistPedestalsHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileP << " <img src=\"HistPedestalsHF.png\" />" << std::endl;
    htmlFileP << "<br>" << std::endl;
    htmlFileP << "<h3> 1.B. Rate of channels at very low Pedestals at least in one CapID for each depth.</h3>"
              << std::endl;
    htmlFileP << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileP << " <img src=\"MapRatePedHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileP << " <img src=\"MapRatePedHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileP << " <img src=\"MapRatePedHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileP << " <img src=\"MapRatePedHF.png\" />" << std::endl;

    // PedestalWidth:
    htmlFileP << "<h2> 2.pWm criterion: Pedestal Widths for each CapID .</h3>" << std::endl;
    htmlFileP << "<h3> 2.A. Pedestal Widths distribution over all events, channels for each CapID and all depths.</h3>"
              << std::endl;
    htmlFileP << "<h4> Legend: Bins less " << Pedest[1][sub] << " correpond to bad Pedestal Widths </h4>" << std::endl;
    if (sub == 1)
      htmlFileP << " <img src=\"HistPedestalWidthsHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileP << " <img src=\"HistPedestalWidthsHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileP << " <img src=\"HistPedestalWidthsHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileP << " <img src=\"HistPedestalWidthsHF.png\" />" << std::endl;
    htmlFileP << "<br>" << std::endl;
    htmlFileP << "<h3> 2.B. Rate of channels at very low Pedestal Widths at least in one CapID for each depth.</h3>"
              << std::endl;
    htmlFileP << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileP << " <img src=\"MapRatePedWidthsHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileP << " <img src=\"MapRatePedWidthsHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileP << " <img src=\"MapRatePedWidthsHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileP << " <img src=\"MapRatePedWidthsHF.png\" />" << std::endl;

    // Correlations of Pedestal(Width) and fullAmplitude:
    htmlFileP << "<h2> 3.Pedestal and pedestalWidths vs Amplitude .</h3>" << std::endl;
    htmlFileP << "<h3> 3.A. Correlation of Pedestal(pedestalWidths) and Amplitude over all channels and events .</h3>"
              << std::endl;
    htmlFileP << "<h4> Legend: colour - entries </h4>" << std::endl;
    if (sub == 1)
      htmlFileP << "<img src=\"CorrelationsMapPedestalVsfullAmplitudeHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileP << "<img src=\"CorrelationsMapPedestalVsfullAmplitudeHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileP << "<img src=\"CorrelationsMapPedestalVsfullAmplitudeHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileP << "<img src=\"CorrelationsMapPedestalVsfullAmplitudeHF.png\" />" << std::endl;
    htmlFileP << "<br>" << std::endl;

    // TSs Shapes:

    htmlFileS << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileS << "<head>" << std::endl;
    htmlFileS << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileS << "<title> Remote Monitoring Tool Global</title>" << std::endl;
    htmlFileS << "<style type=\"text/css\">" << std::endl;
    htmlFileS << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileS << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileS << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileS << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileS << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileS << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileS << "</style>" << std::endl;
    htmlFileS << "<body>" << std::endl;

    if (sub == 1)
      htmlFileS << "<h1> ADC Shape for HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileS << "<h1> ADC Shape for HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileS << "<h1> ADC Shape for HO, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileS << "<h1> ADC Shape for HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileP << "<br>" << std::endl;

    htmlFileS << "<h2> 1.Mean ADC Shape </h3>" << std::endl;
    htmlFileS << "<h3> 1.A. ADC shape averaged over all good channels, depth and events.</h3>" << std::endl;
    //     htmlFileS << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileS << " <img src=\"HistGoodTSshapesHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileS << " <img src=\"HistGoodTSshapesHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileS << " <img src=\"HistGoodTSshapesHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileS << " <img src=\"HistGoodTSshapesHF.png\" />" << std::endl;
    htmlFileS << "<br>" << std::endl;
    htmlFileS << "<h3> 1.B. ADC shape averaged over all bad channels, depth and events. Bad channels are selected by 5 "
                 "criteria: CapId, A, W, P, Pw</h3>"
              << std::endl;
    //     htmlFileS << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileS << " <img src=\"HistBadTSshapesHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileS << " <img src=\"HistBadTSshapesHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileS << " <img src=\"HistBadTSshapesHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileS << " <img src=\"HistBadTSshapesHF.png\" />" << std::endl;

    htmlFileS << "<h2> 2. Pattern of channels for Sub-Detector over depth,eta.phi </h3>" << std::endl;
    htmlFileS << "<h3> 2.A. reminder:.......................... for HBHE, TS=2;...................................... "
                 "for HF, TS=1;..................................... for HO, TS=0,1,2  </h3>"
              << std::endl;
    if (sub == 1)
      htmlFileS << " <img src=\"Hist_mapDepthAllTS2_HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileS << " <img src=\"Hist_mapDepthAllTS2_HE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileS << " <img src=\"Hist_mapDepthAllTS012_HO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileS << " <img src=\"Hist_mapDepthAllTS1_HF.png\" />" << std::endl;
    htmlFileS << "<br>" << std::endl;

    htmlFileS.close();
  }  // end sub

  //======================================================================

  //======================================================================
  // Creating subdet  html pages:

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
    ofstream htmlFile;
    if (sub == 1)
      htmlFile.open("HB.html");
    if (sub == 2)
      htmlFile.open("HE.html");
    if (sub == 3)
      htmlFile.open("HO.html");
    if (sub == 4)
      htmlFile.open("HF.html");

    htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFile << "<head>" << std::endl;
    htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFile << "<title> Remote Monitoring Tool </title>" << std::endl;
    htmlFile << "<style type=\"text/css\">" << std::endl;
    htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
             << std::endl;
    htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                "text-align: center;}"
             << std::endl;
    htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: #FF00FF; }" << std::endl;
    htmlFile << "   td.s6 { font-family: arial, arial ce, helvetica; background-color: #9ACD32; }" << std::endl;
    htmlFile << "   td.s7 { font-family: arial, arial ce, helvetica; background-color: #32CD32; }" << std::endl;
    htmlFile << "   td.s8 { font-family: arial, arial ce, helvetica; background-color: #00FFFF; }" << std::endl;
    htmlFile << "   td.s9 { font-family: arial, arial ce, helvetica; background-color: #FFE4E1; }" << std::endl;
    htmlFile << "   td.s10 { font-family: arial, arial ce, helvetica; background-color: #A0522D; }" << std::endl;
    htmlFile << "   td.s11 { font-family: arial, arial ce, helvetica; background-color: #1E90FF; }" << std::endl;
    htmlFile << "   td.s12 { font-family: arial, arial ce, helvetica; background-color: #00BFFF; }" << std::endl;
    htmlFile << "   td.s13 { font-family: arial, arial ce, helvetica; background-color: #FFFF00; }" << std::endl;
    htmlFile << "   td.s14 { font-family: arial, arial ce, helvetica; background-color: #B8860B; }" << std::endl;
    htmlFile << "</style>" << std::endl;
    htmlFile << "<body>" << std::endl;
    if (sub == 1)
      htmlFile << "<h1> HCAL BARREL, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFile << "<h1> HCAL ENDCAP, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFile << "<h1> HCAL OUTER, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFile << "<h1> HCAL FORWARD, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFile << "<br>" << std::endl;

    htmlFile << "<a name=\"Top\"></a>\n";
    htmlFile << "<b>Contents:<br>\n";
    htmlFile << "1. <a href=\"#AnalysisResults\">Analysis results</a><br>\n";
    htmlFile << "2. <a href=\"#Status\">Status</a><br>\n";
    htmlFile << "2A. <a href=\"#ChannelMap\">Channel map</a><br>\n";
    //   htmlFile << "2B. <a href=\"#BadChannels\">List of bad channels</a><br>\n";
    //   htmlFile << "2C. <a href=\"#BadPedestals\">List of channels with bad pedestals</a><br>\n";

    htmlFile << "<a name=\"AnalysisResults\"></a>\n";
    if (sub == 1)
      htmlFile << "<h2> 1. Analysis results for HB</h2>" << std::endl;
    if (sub == 2)
      htmlFile << "<h2> 1. Analysis results for HE</h2>" << std::endl;
    if (sub == 3)
      htmlFile << "<h2> 1. Analysis results for HO</h2>" << std::endl;
    if (sub == 4)
      htmlFile << "<h2> 1. Analysis results for HF</h2>" << std::endl;
    htmlFile << "<table width=\"400\">" << std::endl;
    htmlFile << "<tr>" << std::endl;

    if (sub == 1) {
      // AZ 12.03.2019
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HB_Tile.html\">Megatile Channels</a></td>" << std::endl;
      //     htmlFile << "  <td><a href=\"HB_Tile.html\">Megatile Channels</a></td>"<< std::endl;

      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HB_Calib.html\">Calibration Channels</a></td>"<< std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HB_Drift.html\">Gain Stability</a></td>"<< std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HB_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HB_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }
    if (sub == 2) {
      // AZ 12.03.2019
      htmlFile << " <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HE_Tile.html\">Megatile Channels</a></td>" << std::endl;
      //     htmlFile << "  <td><a href=\"HE_Tile.html\">Megatile Channels</a></td>"<< std::endl;

      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HE_Calib.html\">Calibration Channels</a></td>"<< std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HE_Drift.html\">Gain Stability</a></td>"<< std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HE_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HE_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }
    if (sub == 3) {
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HO_Tile.html\">Megatile Channels</a></td>" << std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HO_Calib.html\">Calibration Channels</a></td>"<< std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HO_Drift.html\">Gain Stability</a></td>"<< std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HO_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HO_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }
    if (sub == 4) {
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HF_Tile.html\">Megatile Channels</a></td>" << std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HF_Calib.html\">Calibration Channels</a></td>"<< std::endl;
      //       htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HF_Drift.html\">Gain Stability</a></td>"<< std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HF_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
               << runnumber << "/HF_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }

    htmlFile << "</tr>" << std::endl;
    htmlFile << "</table>" << std::endl;
    htmlFile << "<br>" << std::endl;

    //AZ2023:
    /*
    htmlFile << "<a name=\"Status\"></a>\n";
    if (sub == 1)
      htmlFile << "<h2> 2.Status HB over all criteria </h2>" << std::endl;
    if (sub == 2)
      htmlFile << "<h2> 2.Status HE over all criteria </h2>" << std::endl;
    if (sub == 3)
      htmlFile << "<h2> 2.Status HO over all criteria </h2>" << std::endl;
    if (sub == 4)
      htmlFile << "<h2> 2.Status HF over all criteria </h2>" << std::endl;

    htmlFile << "<a name=\"ChannelMap\"></a>\n";
    htmlFile << "<h3> 2.A.Channel map for each Depth </h3>" << std::endl;
    htmlFile << "<h4> Channel legend: yellow - good, white - "
                "not applicable or out of range </h4>"
             << std::endl;
    if (sub == 1)
      htmlFile << " <img src=\"MAPHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFile << " <img src=\"MAPHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFile << " <img src=\"MAPHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFile << " <img src=\"MAPHF.png\" />" << std::endl;
    htmlFile << "<br>" << std::endl;
    htmlFile << "<a href=\"#Top\">to top</a><br>\n";
*/
    /////////////////////////////////////////////////////////////////   AZ 19.03.2018
    /*     
//     htmlFile << "<h3> 2.B.List of Bad channels (rate > 0.1) and its rates for each RMT criteria (for GS - %) </h3>"<< std::endl;

     htmlFile << "<a name=\"BadChannels\"></a>\n";
     htmlFile << "<h3> 2.B.List of Bad channels (rate > 0.1) and its rates for each RMT criteria </h3>"<< std::endl;

     //htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
     //   htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

     htmlFile << "<table>"<< std::endl;     
     htmlFile << "<tr>";
     htmlFile << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Am</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>"   << std::endl; 
//     htmlFile << "<td class=\"s9\" align=\"center\">GS (%)</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>"   << std::endl;
     htmlFile << "</tr>"   << std::endl;     
   
     for (int i=1;i<=NBad;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
	   raw_class2="<td class=\"s11\" align=\"center\">";
	   raw_class3="<td class=\"s13\" align=\"center\">";
	   
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
	   raw_class2="<td class=\"s12\" align=\"center\">";
	   raw_class3="<td class=\"s14\" align=\"center\">";
        }
        const CellDB db;
        CellDB ce;
	if ((ce.size()>=1)&&(Sub[2][i]==sub)) {
	// AZ 19.03.2018
	
// AZ 19           if (Sub[2][i]==1) {
// AZ 19	      ce = db.find("subdet", "HB").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HB, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl; continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HB, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}
// AZ 19	      }
// AZ 19	   if (Sub[2][i]==2) {
// AZ 19	      ce = db.find("subdet", "HE").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HE, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HE, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}	   
// AZ 19	      }
// AZ 19	   if (Sub[2][i]==3) {
// AZ 19	      ce = db.find("subdet", "HO").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HO, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HO, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}	   
// AZ 19	      }	   
// AZ 19	   if (Sub[2][i]==4) {
// AZ 19	      ce = db.find("subdet", "HF").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HF, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HF, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}	   
// AZ 19	      }
	
	   htmlFile << "<tr>"<< std::endl;
           htmlFile << "<td class=\"s4\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[2][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[2][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[2][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[1][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
           htmlFile << raw_class1<< Map_Ampl[2][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[3][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[4][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[5][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[6][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[11][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//         htmlFile << raw_class2<< Map_Ampl[12][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;	 
//	   htmlFile << raw_class2<< Map_Ampl[13][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[14][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[15][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[16][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class3<< Map_Ampl[21][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[31][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[32][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i]+41,Phi[2][i]+1)<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;

        ind+=1;
	}
     } 
     htmlFile << "</table>" << std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<a href=\"#Top\">to top</a><br>\n";

    
     htmlFile << "<h3> 2.C.List of Gain unstable channels and its value in % (for other criterias - rate)</h3>"<< std::endl;
     htmlFile << "<table>"<< std::endl;         
     htmlFile << "<tr>";
     htmlFile << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Am</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>"   << std::endl; 
//     htmlFile << "<td class=\"s9\" align=\"center\">GS(%)</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>"   << std::endl;
     htmlFile << "</tr>"   << std::endl;     
   
     for (int i=1;i<=NWarn;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
	   raw_class2="<td class=\"s11\" align=\"center\">";
	   raw_class3="<td class=\"s13\" align=\"center\">";
	   
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
	   raw_class2="<td class=\"s12\" align=\"center\">";
	   raw_class3="<td class=\"s14\" align=\"center\">";
        }
        const CellDB db;
        CellDB ce;
	if ((ce.size()>=1)&&(Sub[1][i]==sub)) {
           if (Sub[1][i]==1) {
	      ce = db.find("subdet", "HB").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
	      if (ce.size()==0) {cout<<"Error: No such HB, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HB, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	      }
	   if (Sub[1][i]==2) {
	      ce = db.find("subdet", "HE").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
	      if (ce.size()==0) {cout<<"Error: No such HE, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HE, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}	   
	      }
	   if (Sub[1][i]==3) {
	      ce = db.find("subdet", "HO").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
	      if (ce.size()==0) {cout<<"Error: No such HO, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HO, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}	   
	      }	   
	   if (Sub[1][i]==4) {
	      ce = db.find("subdet", "HF").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
	      if (ce.size()==0) {cout<<"Error: No such HF, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HF, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}	   
	      }	
           htmlFile << "<td class=\"s4\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[1][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[1][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[1][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[1][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
           htmlFile << raw_class1<< Map_Ampl[2][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[3][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[4][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[5][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[6][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[11][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//           htmlFile << raw_class2<< Map_Ampl[12][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;	 
//	   htmlFile << raw_class2<< Map_Ampl[13][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[14][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[15][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[16][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class3<< Map_Ampl[21][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[31][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[32][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i]+41,Phi[1][i]+1)<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;
	   htmlFile << "</tr>" << std::endl;
           ind+=1;
	}
     } 
     htmlFile << "</table>" << std::endl; 
     htmlFile << "<br>"<< std::endl;
    
     
//     htmlFile << "<h3> 2.D.List of channels with Bad Pedestals (rate > 0.1) and its rates (for GS - %)</h3>"<< std::endl;
     htmlFile << "<a name=\"BadPedestals\"></a>\n";
     htmlFile << "<h3> 2.C.List of channels with Bad Pedestals (rate > 0.1) and its rates </h3>"<< std::endl;
     htmlFile << "<table>"<< std::endl;         
     htmlFile << "<tr>";
     htmlFile << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Am</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>"   << std::endl;
//     htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>"   << std::endl; 
//     htmlFile << "<td class=\"s9\" align=\"center\">GS(%)</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>"   << std::endl;
     htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>"   << std::endl;
     htmlFile << "</tr>"   << std::endl;     
   
     for (int i=1;i<=NPed;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
	   raw_class2="<td class=\"s11\" align=\"center\">";
	   raw_class3="<td class=\"s13\" align=\"center\">";
	   
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
	   raw_class2="<td class=\"s12\" align=\"center\">";
	   raw_class3="<td class=\"s14\" align=\"center\">";
        }
        const CellDB db;
        CellDB ce;
	if ((ce.size()>=1)&&(Sub[3][i]==sub)) {
	
	// AZ 19.03.2018
// AZ 19           if (Sub[3][i]==1) {
// AZ 19	      ce = db.find("subdet", "HB").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HB, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HB, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;}
// AZ 19	      }
// AZ 19	   if (Sub[3][i]==2) {
// AZ 19	      ce = db.find("subdet", "HE").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HE, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HE, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;}	   
// AZ 19	      }
// AZ 19	   if (Sub[3][i]==3) {
// AZ 19	      ce = db.find("subdet", "HO").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HO, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HO, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;}	   
// AZ 19	      }	   
// AZ 19	   if (Sub[3][i]==4) {
// AZ 19	      ce = db.find("subdet", "HF").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
// AZ 19	      if (ce.size()==0) {cout<<"Error: No such HF, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;continue;}
// AZ 19	      else if (ce.size()>1) {cout<<"Warning: More than one line correspond to such HF, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;}	   
// AZ 19	      }	
	   
           htmlFile << "<td class=\"s4\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[3][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[3][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[3][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[1][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
           htmlFile << raw_class1<< Map_Ampl[2][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[3][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[4][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[5][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class1<< Map_Ampl[6][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[11][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//         htmlFile << raw_class2<< Map_Ampl[12][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;	 
//	   htmlFile << raw_class2<< Map_Ampl[13][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[14][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[15][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class2<< Map_Ampl[16][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
//	   htmlFile << raw_class3<< Map_Ampl[21][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[31][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << raw_class<< Map_Ampl[32][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i]+41,Phi[3][i]+1)<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;
	   htmlFile << "</tr>" << std::endl;
           ind+=1;
	}
     } 
     htmlFile << "</table><br>" << std::endl;
     htmlFile << "<a href=\"#Top\">to top</a><br>\n";
*/
    ///////////////////////////////////////////////////////////////   AZ 19.03.2018

    htmlFile << "</body> " << std::endl;
    htmlFile << "</html> " << std::endl;
    htmlFile.close();
  }

  //======================================================================
  // Creating description html file:
  ofstream htmlFile;

  //======================================================================
  /*
     htmlFile.open("HELP.html");  
     htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">"<< std::endl;
     htmlFile << "<head>"<< std::endl;
     htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>"<< std::endl;
     htmlFile << "<title> Remote Monitoring Tool </title>"<< std::endl;
     htmlFile << "<style type=\"text/css\">"<< std::endl;
     htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"<< std::endl;
     htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }"<< std::endl;
     htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; text-align: center;}"<< std::endl;
     htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }"<< std::endl;
     htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }"<< std::endl;
     htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }"<< std::endl;
     htmlFile << "</style>"<< std::endl;
     htmlFile << "<body>"<< std::endl;
     htmlFile << "<h1>  Description of Remote Monitoring Tool criteria for bad channel selection</h1>"<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - C means CAPID Errors assuming we inspect CAPID non-rotation,error & validation bits, and for this criterion - no need to apply any cuts to select bcs.</h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - A means full amplitude, collected over all time slices </h3> "<< std::endl;
     htmlFile << "<h3> - R means ratio criterion where we define as a bad, the channels, for which the signal portion in 4 middle TSs(plus one, minus two around TS with maximal amplitude) is out of some range of reasonable values </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - W means width of shape distribution. Width is defined as square root from dispersion. </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - TN means mean time position of adc signal. </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - TX means TS number of maximum signal </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;     
     htmlFile << "<h3> - m means megatile channels. For example Am means Amplitude criteria for megatile channels </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - c means calibration channels. For example Ac means Amplitude criteria for calibration channels </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "<h3> - Pm means Pedestals. </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;  
     htmlFile << "<h3> - pWm  means pedestal Width. </h3> "<< std::endl;
     htmlFile << "<br>"<< std::endl;
     htmlFile << "</body> " << std::endl;
     htmlFile << "</html> " << std::endl; 
     htmlFile.close();
*/
  //======================================================================

  //======================================================================
  // Creating main html file:
  htmlFile.open("MAP.html");
  htmlFile << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
  htmlFile << "<head>" << std::endl;
  htmlFile << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
  htmlFile << "<title> Remote Monitoring Tool </title>" << std::endl;
  htmlFile << "<style type=\"text/css\">" << std::endl;
  htmlFile << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
           << std::endl;
  htmlFile << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
  htmlFile << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
              "text-align: center;}"
           << std::endl;
  htmlFile << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
  htmlFile << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
  htmlFile << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
  htmlFile << "   td.s5 { font-family: arial, arial ce, helvetica; background-color: #FF00FF; }" << std::endl;
  htmlFile << "   td.s6 { font-family: arial, arial ce, helvetica; background-color: #9ACD32; }" << std::endl;
  htmlFile << "   td.s7 { font-family: arial, arial ce, helvetica; background-color: #32CD32; }" << std::endl;
  htmlFile << "</style>" << std::endl;
  htmlFile << "<body>" << std::endl;

  htmlFile << "<h1> Remote Monitoring Tool, RUN = " << runnumber << ". </h1>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h2> 1. Analysis results for subdetectors </h2>" << std::endl;
  htmlFile << "<table width=\"400\">" << std::endl;
  htmlFile << "<tr>" << std::endl;

  // AZ 12.03.2019
  /*
     htmlFile << "  <td><a href=\"HB.html\">HB</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HE.html\">HE</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HO.html\">HO</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"HF.html\">HF</a></td>"<< std::endl;    
*/

  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
           << runnumber << "/HB.html\">HB</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
           << runnumber << "/HE.html\">HE</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
           << runnumber << "/HO.html\">HO</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"
           << runnumber << "/HF.html\">HF</a></td>" << std::endl;

  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;
  //AZ2023:
  /*
  htmlFile << "<h2> 2. HCAL status over all criteria and subdetectors </h2>" << std::endl;
  htmlFile << "<h3> 2.A. Channels in detector space </h3>" << std::endl;
  htmlFile << "<h4> Legend for channel status: green - good, others - may be a problems, white - not applicable or out "
              "of range </h4>"
           << std::endl;
  htmlFile << " <img src=\"MAP.png\" />" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h3> 2.B. List of Bad channels </h3>" << std::endl;

  //htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
  //   htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/HELP.html\"> "
              "Description of criteria for bad channel selection</a></td>"
           << std::endl;

  htmlFile << "<table>" << std::endl;
  htmlFile << "<tr>";
  htmlFile << "<td class=\"s4\" align=\"center\">#</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">RM</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>" << std::endl;
  htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>" << std::endl;
  htmlFile << "<td class=\"s5\" align=\"center\">RMT-criteria</td>" << std::endl;
  htmlFile << "</tr>" << std::endl;
*/
  //AZ2023  ind = 0;
  // AZ 19.03.2018
  /*     
     for (int i=1;i<=NBad;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
        }
        const CellDB db;
        const CellDB ce = db.find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
	//           if (ce.size()==0) {cout<<"Error: No such Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;continue;}
//	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}
	
	if (ce.size()>=1) {
	   htmlFile << "<tr>"<< std::endl;
           htmlFile << "<td class=\"s1\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[2][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[2][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[2][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
           htmlFile << raw_class1<< Comment[2][i]<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;

        ind+=1;
	}
     } /// end loop
*/
  //AZ2023  htmlFile << "</table>" << std::endl;
  //AZ2023  htmlFile << "<br>" << std::endl;
  /*     
     htmlFile << "<h3> 2.C.List of Gain unstable channels </h3>"<< std::endl;
     //htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
     //   htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

     htmlFile << "<table>"<< std::endl;     
     htmlFile << "<tr>";
     htmlFile << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Failed criteria</td>"   << std::endl;
     htmlFile << "</tr>"   << std::endl;     
   
     for (int i=1;i<=NWarn;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
        }
        const CellDB db;
        const CellDB ce = db.find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
	//	    if (ce.size()==0) {cout<<"Error: No such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;continue;}
//	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	
	if (ce.size()>=1) {
	   htmlFile << "<tr>"<< std::endl;
           htmlFile << "<td class=\"s1\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[1][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[1][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[1][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
           htmlFile << raw_class1<< Comment[1][i]<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;

           ind+=1;
	}
     } 
    
   
     htmlFile << "</table>" << std::endl;
     htmlFile << "<br>"<< std::endl;
     
     
     htmlFile << "<h3> 2.D.List of channels with bad Pedestals </h3>"<< std::endl;
     // htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
     //   htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/GLOBAL_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
  htmlFile << "  <td><a href=\"https://cms-conddb.cern.ch/eosweb/hcal/HcalRemoteMonitoring/GlobalRMT/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

     htmlFile << "<table>"<< std::endl;     
     htmlFile << "<tr>";
     htmlFile << "<td class=\"s4\" align=\"center\">#</td>"    << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ETA</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PHI</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DEPTH</td>"<< std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RBX</td>"  << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">PIXEL</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">RM_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">FIBER_CH</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">QIE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">ADC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">CRATE</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">DCC</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">SPIGOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FIBER</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_SLOT</td>"   << std::endl;
     htmlFile << "<td class=\"s1\" align=\"center\">HTR_FPGA</td>"   << std::endl;
     htmlFile << "<td class=\"s5\" align=\"center\">Failed criteria</td>"   << std::endl;
     htmlFile << "</tr>"   << std::endl;     
   
     for (int i=1;i<=NPed;i++) {
        if((ind%2)==1){
           raw_class="<td class=\"s2\" align=\"center\">";
	   raw_class1="<td class=\"s6\" align=\"center\">";
        }else{
           raw_class="<td class=\"s3\" align=\"center\">";
	   raw_class1="<td class=\"s7\" align=\"center\">";
        }
        const CellDB db;
        const CellDB ce = db.find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
	//	    if (ce.size()==0) {cout<<"Error: No such Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;continue;}
//	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}
	
	if (ce.size()>=1) {
	   htmlFile << "<tr>"<< std::endl;
           htmlFile << "<td class=\"s1\" align=\"center\">" << ind+1 <<"</td>"<< std::endl;
           htmlFile << raw_class<< Eta[3][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Phi[3][i]<<"</td>"<< std::endl;
           htmlFile << raw_class<< Depth[3][i] <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RBX <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RM <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Pixel <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].RMfiber <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberCh <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].QIE <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].ADC<<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].VMECardID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].dccID <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].Spigot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].FiberIndex <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrSlot <<"</td>"<< std::endl;
           htmlFile << raw_class<< ce[0].HtrTB <<"</td>"<< std::endl;
           htmlFile << raw_class1<< Comment[3][i]<<"</td>"<< std::endl;
	   htmlFile << "</tr>" << std::endl;

           ind+=1;
	}
     } 
    
   
     htmlFile << "</table>" << std::endl;
*/
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();
  //======================================================================

  //======================================================================
  // Close and delete all possible things:
  hfile->Close();
  //  hfile->Delete();
  //  Exit Root
  gSystem->Exit(0);
  //======================================================================
}
