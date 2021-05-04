// How to run:
//root -b -q -l 'RemoteMonitoringMAP.C+("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214513.root","/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214512.root")'
//root -b -q -l 'RemoteMonitoringMAP.C+(" /afs/cern.ch/work/d/dtlisov/private/Monitoring/histos/LED_211659.root","/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/RDMweb/histos/LED_214512.root")'
// ./RemoteMonitoringMAP.cc.exe root_file1  root_file_ref PEDESTAL
//./RemoteMonitoringMAP.cc.exe root_file1  root_file_ref LASER

//MapRateAmpl2HB

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

using namespace std;
// h_ADC_HBdepth1
// ok change   Hist_ADC_HB_DS
int main(int argc, char *argv[]) {
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(1);
  // ok change
  if (argc < 3)
    return 1;
  char fname[300];
  char refname[300];
  char runtypeC[300];
  sprintf(fname, "%s", argv[1]);
  sprintf(refname, "%s", argv[2]);
  sprintf(runtypeC, "%s", argv[3]);
  std::cout << fname << " " << refname << " " << runtypeC << std::endl;
  // ok change

  //======================================================================
  // Connect the input files, parameters and get the 2-d histogram in memory
  //    TFile *hfile= new TFile("GlobalHist.root", "READ");
  string promt = (string)fname;
  string runtype = (string)runtypeC;
  string runnumber = "";
  for (unsigned int i = promt.size() - 11; i < promt.size() - 5; i++)
    runnumber += fname[i];
  string refrunnumber = "";
  promt = (string)refname;
  for (unsigned int i = promt.size() - 11; i < promt.size() - 5; i++)
    refrunnumber += refname[i];

  TFile *hfile = new TFile(fname, "READ");
  TFile *hreffile = new TFile(refname, "READ");
  double MIN_M[7][5];
  double MAX_M[7][5];
  double MIN_C[7][5];
  double MAX_C[7][5];
  double porog[5];
  double Pedest[2][5];

  if (runtype == "LED") {
    //CUTS:    [test][subdetector]                                       ADC amplitude  Am      Width  for Wm             Ratio cut for Rm             TS mean for TNm           TS max  for TXm
    double MIN_M_LED[7][5] = {{0., 0., 0., 0., 0.},
                              {0., 0., 0., 0., 0.},
                              {0, 100., 70., 40., 40.},
                              {0, 0.7, 0.7, 0.1, 0.1},
                              {0, 0.7, 0.6, 0.40, 0.45},
                              {0, 2.5, 1.0, 1.0, 1.0},
                              {0, 1.5, 1.5, 0.5, 0.5}};
    double MAX_M_LED[7][5] = {{0., 0., 0., 0., 0.},
                              {0., 0., 0., 0., 0.},
                              {0, 3000, 500000, 3000, 150000},
                              {0, 2.5, 2.5, 2.8, 2.6},
                              {0, 0.94, 0.95, 1.04, 1.02},
                              {0, 5.5, 5.2, 4.8, 4.2},
                              {0, 6.5, 6.5, 8.5, 8.5}};
    double MIN_C_LED[7][5] = {{0., 0., 0., 0., 0.},
                              {0., 0., 0., 0., 0.},
                              {0, 1000., 1000., 1000., 100.},
                              {0, 1.3, 1.3, 0.7, 0.3},
                              {0, 0.76, 0.76, 0.85, 0.5},
                              {0, 2.4, 2.4, 1.5, 3.5},
                              {0, 1.5, 1.5, 1.5, 3.5}};
    double MAX_C_LED[7][5] = {{0., 0., 0., 0., 0.},
                              {0., 0., 0., 0., 0.},
                              {0, 1E20, 1E20, 1E20, 1E20},
                              {0, 1.9, 1.9, 1.65, 1.5},
                              {0, 0.94, 0.94, 0.99, 0.8},
                              {0, 3.7, 3.7, 2.7, 4.5},
                              {0, 2.5, 2.5, 2.5, 4.5}};
    double porog_LED[5] = {0., 2., 2., 2., 2.};  // Cut for GS test in pro cents
    double Pedest_LED[2][5] = {{0., 0.1, 0.6, 0.1, 0.8},
                               {0., 0.1, 0.1, 0.1, 0.4}};  //Cuts for Pedestal  and pedestal  Width
    for (int i = 0; i <= 6; i++)
      for (int j = 0; j <= 4; j++) {
        MIN_M[i][j] = MIN_M_LED[i][j];
        MAX_M[i][j] = MAX_M_LED[i][j];
        MIN_C[i][j] = MIN_C_LED[i][j];
        MAX_C[i][j] = MAX_C_LED[i][j];
      }
    for (int i = 0; i <= 4; i++) {
      porog[i] = porog_LED[i];
      Pedest[0][i] = Pedest_LED[0][i];
      Pedest[1][i] = Pedest_LED[1][i];
    }
  }

  if (runtype == "LASER") {
    //CUTS:    [test][subdetector]                                  ADC amplitude  Am      Width  for Wm             Ratio cut for Rm             TS mean for TNm           TS max  for TXm
    double MIN_M_LASER[7][5] = {{0., 0., 0., 0., 0.},
                                {0., 0., 0., 0., 0.},
                                {0, 40., 40., 100., 40.},
                                {0, 0.3, 0.9, 0.2, 0.2},
                                {0, 0.5, 0.55, 0.55, 0.60},
                                {0, 5.0, 2.5, 1.1, 5.5},
                                {0, 1.5, 1.5, 1.5, 1.5}};
    double MAX_M_LASER[7][5] = {{0., 0., 0., 0., 0.},
                                {0., 0., 0., 0., 0.},
                                {0, 3500, 350000, 3500, 150000},
                                {0, 2.5, 3.6, 2.6, 2.1},
                                {0, 1.00, 1.00, 1.04, 1.02},
                                {0, 7.5, 6.5, 4.4, 8.5},
                                {0, 8.5, 8.5, 6.5, 8.5}};
    double MIN_C_LASER[7][5] = {{0., 0., 0., 0., 0.},
                                {0., 0., 0., 0., 0.},
                                {0, 1000., 1000., 1000., 100.},
                                {0, 1.3, 1.3, 0.7, 0.3},
                                {0, 0.76, 0.76, 0.85, 0.5},
                                {0, 2.4, 2.4, 1.5, 3.5},
                                {0, 1.5, 1.5, 1.5, 3.5}};
    double MAX_C_LASER[7][5] = {{0., 0., 0., 0., 0.},
                                {0., 0., 0., 0., 0.},
                                {0, 1E20, 1E20, 1E20, 1E20},
                                {0, 1.9, 1.9, 1.65, 1.5},
                                {0, 0.94, 0.94, 1.0, 0.8},
                                {0, 3.7, 3.7, 2.7, 4.5},
                                {0, 2.5, 2.5, 2.5, 4.5}};
    double porog_LASER[5] = {0., 2., 2., 2., 2.};  // Cut for GS test in pro cents
    //     double porog_LASER[5] = {0., 200., 200., 100., 100.}; // Cut for GS test in pro cents
    double Pedest_LASER[2][5] = {{0., 0.1, 0.6, 0.1, 0.8},
                                 {0., 0.1, 0.1, 0.1, 0.4}};  //Cuts for Pedestal  and pedestal  Width
    for (int i = 0; i <= 6; i++)
      for (int j = 0; j <= 4; j++) {
        MIN_M[i][j] = MIN_M_LASER[i][j];
        MAX_M[i][j] = MAX_M_LASER[i][j];
        MIN_C[i][j] = MIN_C_LASER[i][j];
        MAX_C[i][j] = MAX_C_LASER[i][j];
      }
    for (int i = 0; i <= 4; i++) {
      porog[i] = porog_LASER[i];
      Pedest[0][i] = Pedest_LASER[0][i];
      Pedest[1][i] = Pedest_LASER[1][i];
    }
  }
  if (runtype == "PEDESTAL") {
    //CUTS:    [test][subdetector]                                  ADC amplitude  Am      Width  for Wm             Ratio cut for Rm             TS mean for TNm           TS max  for TXm
    double MIN_M_PEDESTAL[7][5] = {{0., 0., 0., 0., 0.},
                                   {0., 0., 0., 0., 0.},
                                   {0, 10., 10., 200., 5.},
                                   {0, 2.7, 2.7, 2.7, 0.2},
                                   {0, 0.31, 0.31, 0.05, 0.15},
                                   {0, 4.5, 4.5, 4.5, 2.0},
                                   {0, 0.5, 0.5, 0.5, 0.5}};
    double MAX_M_PEDESTAL[7][5] = {{0., 0., 0., 0., 0.},
                                   {0., 0., 0., 0., 0.},
                                   {0, 2500., 250000., 2500., 150000.},
                                   {0, 3.0, 3.0, 5.0, 3.0},
                                   {0, 0.95, 0.95, 1.00, 0.98},
                                   {0, 4.6, 4.6, 4.6, 7.0},
                                   {0, 9.5, 9.5, 8.5, 8.5}};
    double MIN_C_PEDESTAL[7][5] = {{0., 0., 0., 0., 0.},
                                   {0., 0., 0., 0., 0.},
                                   {0, 1000., 1000., 1000., 100.},
                                   {0, 1.3, 1.3, 0.7, 0.3},
                                   {0, 0.76, 0.76, 0.85, 0.5},
                                   {0, 2.4, 2.4, 1.5, 3.5},
                                   {0, 1.5, 1.5, 1.5, 3.5}};
    double MAX_C_PEDESTAL[7][5] = {{0., 0., 0., 0., 0.},
                                   {0., 0., 0., 0., 0.},
                                   {0, 1E20, 1E20, 1E20, 1E20},
                                   {0, 1.9, 1.9, 1.65, 1.5},
                                   {0, 0.94, 0.94, 1.0, 0.8},
                                   {0, 3.7, 3.7, 2.7, 4.5},
                                   {0, 2.5, 2.5, 2.5, 4.5}};
    double porog_PEDESTAL[5] = {0., 2., 2., 2., 2.};  // Cut for GS test in pro cents
    //     double porog_PEDESTAL[5] = {0., 200., 200., 100., 100.}; // Cut for GS test in pro cents
    double Pedest_PEDESTAL[2][5] = {{0., 0.1, 0.6, 0.1, 0.8},
                                    {0., 0.1, 0.1, 0.1, 0.4}};  //Cuts for Pedestal  and pedestal  Width
    for (int i = 0; i <= 6; i++)
      for (int j = 0; j <= 4; j++) {
        MIN_M[i][j] = MIN_M_PEDESTAL[i][j];
        MAX_M[i][j] = MAX_M_PEDESTAL[i][j];
        MIN_C[i][j] = MIN_C_PEDESTAL[i][j];
        MAX_C[i][j] = MAX_C_PEDESTAL[i][j];
      }
    for (int i = 0; i <= 4; i++) {
      porog[i] = porog_PEDESTAL[i];
      Pedest[0][i] = Pedest_PEDESTAL[0][i];
      Pedest[1][i] = Pedest_PEDESTAL[1][i];
    }
  }

  //======================================================================

  cout << endl;
  cout << MIN_M[2][1] << endl;

  //======================================================================
  // Prepare histograms and plot them to .png files

  //TCanvas *cHB = new TCanvas("cHB","cHB",1000,500);
  TCanvas *cHB = new TCanvas("cHB", "cHB", 1000, 1000);
  //TCanvas *cHE = new TCanvas("cHE","cHE",1500,500);
  TCanvas *cHE = new TCanvas("cHE", "cHE", 1500, 1500);
  //TCanvas *cONE = new TCanvas("cONE","cONE",500,500);
  TCanvas *cONE = new TCanvas("cONE", "cONE", 1500, 500);
  TCanvas *cPED = new TCanvas("cPED", "cPED", 1000, 500);
  //TCanvas *cHF = new TCanvas("cHF","cHF",1000,1000);
  TCanvas *cHF = new TCanvas("cHF", "cHF", 1000, 1000);

  // Iteration Method for Calibration Group:

  TCanvas *c1x0 = new TCanvas("c1x0", "c1x0", 300, 10, 800, 700);

  TCanvas *c1x1 = new TCanvas("c1x1", "c1x1", 100, 10, 600, 700);

  TCanvas *c3x5 = new TCanvas("c3x5", "c3x5", 1000, 1500);
  //

  char *str = (char *)alloca(10000);

  // before upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3; HO depth4; HF depth1,2
  // 5 depthes:  0(empty),   1,2,3,4

  // upgrade 2017:
  // depth: HB depth1,2; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 8 depthes:  0(empty),   1,2,3,4,5,6,7

  // upgrade 2021:
  // depth: HB depth1,2,3,4; HE depth1,2,3,4,5,6,7; HO depth4; HF depth1,2,3,4
  // 10 depthes:  0(empty),   1,2,3,4,5,6,7,8,9

  //  Int_t ALLDEPTH = 5;
  //  Int_t ALLDEPTH = 8;
  Int_t ALLDEPTH = 10;

  int k_min[5] = {0, 1, 1, 4, 1};  // minimum depth for each subdet
  //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
  //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
  int k_max[5] = {0, 4, 7, 4, 4};  // maximum depth for each subdet

  TH2F *Map_Ampl[33][5][ALLDEPTH];       // 2D histogramm for test,subdet,depth
  TH2F *Map_SUB[5][ALLDEPTH];            // 2d histogramm for subdet, depth
  TH1F *HistAmplDepth[22][5][ALLDEPTH];  // 1d histogramm for test,subdet, depth
  TH1F *HistAmpl[22][5];                 // 1d histogramm for test,subdet

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

  Map_SUB[1][1] = (TH2F *)hfile->Get("h_mapDepth1_HB");
  Map_SUB[1][2] = (TH2F *)hfile->Get("h_mapDepth2_HB");
  Map_SUB[2][1] = (TH2F *)hfile->Get("h_mapDepth1_HE");
  Map_SUB[2][2] = (TH2F *)hfile->Get("h_mapDepth2_HE");
  Map_SUB[2][3] = (TH2F *)hfile->Get("h_mapDepth3_HE");
  Map_SUB[3][4] = (TH2F *)hfile->Get("h_mapDepth4_HO");
  Map_SUB[4][1] = (TH2F *)hfile->Get("h_mapDepth1_HF");
  Map_SUB[4][2] = (TH2F *)hfile->Get("h_mapDepth2_HF");

  Map_SUB[1][3] = (TH2F *)hfile->Get("h_mapDepth3_HB");
  Map_SUB[1][4] = (TH2F *)hfile->Get("h_mapDepth4_HB");
  Map_SUB[2][4] = (TH2F *)hfile->Get("h_mapDepth4_HE");
  Map_SUB[2][5] = (TH2F *)hfile->Get("h_mapDepth5_HE");
  Map_SUB[2][6] = (TH2F *)hfile->Get("h_mapDepth6_HE");
  Map_SUB[2][7] = (TH2F *)hfile->Get("h_mapDepth7_HE");
  Map_SUB[4][3] = (TH2F *)hfile->Get("h_mapDepth3_HF");
  Map_SUB[4][4] = (TH2F *)hfile->Get("h_mapDepth4_HF");

  //+++++++++++++++++++++++++++++
  //Test 0 Entries
  //+++++++++++++++++++++++++++++

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //       if (sub==1) cHB->Divide(2,1);
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
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
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

  Map_Ampl[1][1][1] = (TH2F *)hfile->Get("h_mapDepth1Error_HB");
  Map_Ampl[1][1][2] = (TH2F *)hfile->Get("h_mapDepth2Error_HB");
  Map_Ampl[1][2][1] = (TH2F *)hfile->Get("h_mapDepth1Error_HE");
  Map_Ampl[1][2][2] = (TH2F *)hfile->Get("h_mapDepth2Error_HE");
  Map_Ampl[1][2][3] = (TH2F *)hfile->Get("h_mapDepth3Error_HE");
  Map_Ampl[1][3][4] = (TH2F *)hfile->Get("h_mapDepth4Error_HO");
  Map_Ampl[1][4][1] = (TH2F *)hfile->Get("h_mapDepth1Error_HF");
  Map_Ampl[1][4][2] = (TH2F *)hfile->Get("h_mapDepth2Error_HF");

  Map_Ampl[1][1][3] = (TH2F *)hfile->Get("h_mapDepth3Error_HB");
  Map_Ampl[1][1][4] = (TH2F *)hfile->Get("h_mapDepth4Error_HB");
  Map_Ampl[1][2][4] = (TH2F *)hfile->Get("h_mapDepth4Error_HE");
  Map_Ampl[1][2][5] = (TH2F *)hfile->Get("h_mapDepth5Error_HE");
  Map_Ampl[1][2][6] = (TH2F *)hfile->Get("h_mapDepth6Error_HE");
  Map_Ampl[1][2][7] = (TH2F *)hfile->Get("h_mapDepth7Error_HE");
  Map_Ampl[1][4][3] = (TH2F *)hfile->Get("h_mapDepth3Error_HF");
  Map_Ampl[1][4][4] = (TH2F *)hfile->Get("h_mapDepth4Error_HF");

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
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
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

  Map_Ampl[2][1][1] = (TH2F *)hfile->Get("h_mapDepth1ADCAmpl225_HB");
  Map_Ampl[2][1][2] = (TH2F *)hfile->Get("h_mapDepth2ADCAmpl225_HB");
  Map_Ampl[2][2][1] = (TH2F *)hfile->Get("h_mapDepth1ADCAmpl225_HE");
  Map_Ampl[2][2][2] = (TH2F *)hfile->Get("h_mapDepth2ADCAmpl225_HE");
  Map_Ampl[2][2][3] = (TH2F *)hfile->Get("h_mapDepth3ADCAmpl225_HE");
  Map_Ampl[2][3][4] = (TH2F *)hfile->Get("h_mapDepth4ADCAmpl225_HO");
  Map_Ampl[2][4][1] = (TH2F *)hfile->Get("h_mapDepth1ADCAmpl225_HF");
  Map_Ampl[2][4][2] = (TH2F *)hfile->Get("h_mapDepth2ADCAmpl225_HF");

  Map_Ampl[2][1][3] = (TH2F *)hfile->Get("h_mapDepth3ADCAmpl225_HB");
  Map_Ampl[2][1][4] = (TH2F *)hfile->Get("h_mapDepth4ADCAmpl225_HB");
  Map_Ampl[2][2][4] = (TH2F *)hfile->Get("h_mapDepth4ADCAmpl225_HE");
  Map_Ampl[2][2][5] = (TH2F *)hfile->Get("h_mapDepth5ADCAmpl225_HE");
  Map_Ampl[2][2][6] = (TH2F *)hfile->Get("h_mapDepth6ADCAmpl225_HE");
  Map_Ampl[2][2][7] = (TH2F *)hfile->Get("h_mapDepth7ADCAmpl225_HE");
  Map_Ampl[2][4][3] = (TH2F *)hfile->Get("h_mapDepth3ADCAmpl225_HF");
  Map_Ampl[2][4][4] = (TH2F *)hfile->Get("h_mapDepth4ADCAmpl225_HF");

  HistAmpl[2][1] = (TH1F *)hfile->Get("h_ADCAmpl_HB");
  HistAmpl[2][2] = (TH1F *)hfile->Get("h_ADCAmpl_HE");
  HistAmpl[2][3] = (TH1F *)hfile->Get("h_ADCAmpl_HO");
  HistAmpl[2][4] = (TH1F *)hfile->Get("h_ADCAmpl_HF");

  //+++++++++++++++++++++++++++++
  //Test 3 (Wm) Rate of RMS
  //+++++++++++++++++++++++++++++

  Map_Ampl[3][1][1] = (TH2F *)hfile->Get("h_mapDepth1Amplitude225_HB");
  Map_Ampl[3][1][2] = (TH2F *)hfile->Get("h_mapDepth2Amplitude225_HB");
  Map_Ampl[3][2][1] = (TH2F *)hfile->Get("h_mapDepth1Amplitude225_HE");
  Map_Ampl[3][2][2] = (TH2F *)hfile->Get("h_mapDepth2Amplitude225_HE");
  Map_Ampl[3][2][3] = (TH2F *)hfile->Get("h_mapDepth3Amplitude225_HE");
  Map_Ampl[3][3][4] = (TH2F *)hfile->Get("h_mapDepth4Amplitude225_HO");
  Map_Ampl[3][4][1] = (TH2F *)hfile->Get("h_mapDepth1Amplitude225_HF");
  Map_Ampl[3][4][2] = (TH2F *)hfile->Get("h_mapDepth2Amplitude225_HF");

  Map_Ampl[3][1][3] = (TH2F *)hfile->Get("h_mapDepth3Amplitude225_HB");
  Map_Ampl[3][1][4] = (TH2F *)hfile->Get("h_mapDepth4Amplitude225_HB");
  Map_Ampl[3][2][4] = (TH2F *)hfile->Get("h_mapDepth4Amplitude225_HE");
  Map_Ampl[3][2][5] = (TH2F *)hfile->Get("h_mapDepth5Amplitude225_HE");
  Map_Ampl[3][2][6] = (TH2F *)hfile->Get("h_mapDepth6Amplitude225_HE");
  Map_Ampl[3][2][7] = (TH2F *)hfile->Get("h_mapDepth7Amplitude225_HE");
  Map_Ampl[3][4][3] = (TH2F *)hfile->Get("h_mapDepth3Amplitude225_HF");
  Map_Ampl[3][4][4] = (TH2F *)hfile->Get("h_mapDepth4Amplitude225_HF");

  HistAmpl[3][1] = (TH1F *)hfile->Get("h_Amplitude_HB");
  HistAmpl[3][2] = (TH1F *)hfile->Get("h_Amplitude_HE");
  HistAmpl[3][3] = (TH1F *)hfile->Get("h_Amplitude_HO");
  HistAmpl[3][4] = (TH1F *)hfile->Get("h_Amplitude_HF");

  //+++++++++++++++++++++++++++++
  //Test 4 (Rm) Rate of ratio 4 near max TS/ All TS
  //+++++++++++++++++++++++++++++

  Map_Ampl[4][1][1] = (TH2F *)hfile->Get("h_mapDepth1Ampl047_HB");
  Map_Ampl[4][1][2] = (TH2F *)hfile->Get("h_mapDepth2Ampl047_HB");
  Map_Ampl[4][2][1] = (TH2F *)hfile->Get("h_mapDepth1Ampl047_HE");
  Map_Ampl[4][2][2] = (TH2F *)hfile->Get("h_mapDepth2Ampl047_HE");
  Map_Ampl[4][2][3] = (TH2F *)hfile->Get("h_mapDepth3Ampl047_HE");
  Map_Ampl[4][3][4] = (TH2F *)hfile->Get("h_mapDepth4Ampl047_HO");
  Map_Ampl[4][4][1] = (TH2F *)hfile->Get("h_mapDepth1Ampl047_HF");
  Map_Ampl[4][4][2] = (TH2F *)hfile->Get("h_mapDepth2Ampl047_HF");

  Map_Ampl[4][1][3] = (TH2F *)hfile->Get("h_mapDepth3Ampl047_HB");
  Map_Ampl[4][1][4] = (TH2F *)hfile->Get("h_mapDepth4Ampl047_HB");
  Map_Ampl[4][2][4] = (TH2F *)hfile->Get("h_mapDepth4Ampl047_HE");
  Map_Ampl[4][2][5] = (TH2F *)hfile->Get("h_mapDepth5Ampl047_HE");
  Map_Ampl[4][2][6] = (TH2F *)hfile->Get("h_mapDepth6Ampl047_HE");
  Map_Ampl[4][2][7] = (TH2F *)hfile->Get("h_mapDepth7Ampl047_HE");
  Map_Ampl[4][4][3] = (TH2F *)hfile->Get("h_mapDepth3Ampl047_HF");
  Map_Ampl[4][4][4] = (TH2F *)hfile->Get("h_mapDepth4Ampl047_HF");

  HistAmpl[4][1] = (TH1F *)hfile->Get("h_Ampl_HB");
  HistAmpl[4][2] = (TH1F *)hfile->Get("h_Ampl_HE");
  HistAmpl[4][3] = (TH1F *)hfile->Get("h_Ampl_HO");
  HistAmpl[4][4] = (TH1F *)hfile->Get("h_Ampl_HF");

  //+++++++++++++++++++++++++++++
  //Test 5 (TNm) Mean position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[5][1][1] = (TH2F *)hfile->Get("h_mapDepth1TSmeanA225_HB");
  Map_Ampl[5][1][2] = (TH2F *)hfile->Get("h_mapDepth2TSmeanA225_HB");
  Map_Ampl[5][2][1] = (TH2F *)hfile->Get("h_mapDepth1TSmeanA225_HE");
  Map_Ampl[5][2][2] = (TH2F *)hfile->Get("h_mapDepth2TSmeanA225_HE");
  Map_Ampl[5][2][3] = (TH2F *)hfile->Get("h_mapDepth3TSmeanA225_HE");
  Map_Ampl[5][3][4] = (TH2F *)hfile->Get("h_mapDepth4TSmeanA225_HO");
  Map_Ampl[5][4][1] = (TH2F *)hfile->Get("h_mapDepth1TSmeanA225_HF");
  Map_Ampl[5][4][2] = (TH2F *)hfile->Get("h_mapDepth2TSmeanA225_HF");

  Map_Ampl[5][1][3] = (TH2F *)hfile->Get("h_mapDepth3TSmeanA225_HB");
  Map_Ampl[5][1][4] = (TH2F *)hfile->Get("h_mapDepth4TSmeanA225_HB");
  Map_Ampl[5][2][4] = (TH2F *)hfile->Get("h_mapDepth4TSmeanA225_HE");
  Map_Ampl[5][2][5] = (TH2F *)hfile->Get("h_mapDepth5TSmeanA225_HE");
  Map_Ampl[5][2][6] = (TH2F *)hfile->Get("h_mapDepth6TSmeanA225_HE");
  Map_Ampl[5][2][7] = (TH2F *)hfile->Get("h_mapDepth7TSmeanA225_HE");
  Map_Ampl[5][4][3] = (TH2F *)hfile->Get("h_mapDepth3TSmeanA225_HF");
  Map_Ampl[5][4][4] = (TH2F *)hfile->Get("h_mapDepth4TSmeanA225_HF");

  HistAmpl[5][1] = (TH1F *)hfile->Get("h_TSmeanA_HB");
  HistAmpl[5][2] = (TH1F *)hfile->Get("h_TSmeanA_HE");
  HistAmpl[5][3] = (TH1F *)hfile->Get("h_TSmeanA_HO");
  HistAmpl[5][4] = (TH1F *)hfile->Get("h_TSmeanA_HF");

  //+++++++++++++++++++++++++++++
  //Test 6 (TXm) Maximum position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[6][1][1] = (TH2F *)hfile->Get("h_mapDepth1TSmaxA225_HB");
  Map_Ampl[6][1][2] = (TH2F *)hfile->Get("h_mapDepth2TSmaxA225_HB");
  Map_Ampl[6][2][1] = (TH2F *)hfile->Get("h_mapDepth1TSmaxA225_HE");
  Map_Ampl[6][2][2] = (TH2F *)hfile->Get("h_mapDepth2TSmaxA225_HE");
  Map_Ampl[6][2][3] = (TH2F *)hfile->Get("h_mapDepth3TSmaxA225_HE");
  Map_Ampl[6][3][4] = (TH2F *)hfile->Get("h_mapDepth4TSmaxA225_HO");
  Map_Ampl[6][4][1] = (TH2F *)hfile->Get("h_mapDepth1TSmaxA225_HF");
  Map_Ampl[6][4][2] = (TH2F *)hfile->Get("h_mapDepth2TSmaxA225_HF");

  Map_Ampl[6][1][3] = (TH2F *)hfile->Get("h_mapDepth3TSmaxA225_HB");
  Map_Ampl[6][1][4] = (TH2F *)hfile->Get("h_mapDepth4TSmaxA225_HB");
  Map_Ampl[6][2][4] = (TH2F *)hfile->Get("h_mapDepth4TSmaxA225_HE");
  Map_Ampl[6][2][5] = (TH2F *)hfile->Get("h_mapDepth5TSmaxA225_HE");
  Map_Ampl[6][2][6] = (TH2F *)hfile->Get("h_mapDepth6TSmaxA225_HE");
  Map_Ampl[6][2][7] = (TH2F *)hfile->Get("h_mapDepth7TSmaxA225_HE");
  Map_Ampl[6][4][3] = (TH2F *)hfile->Get("h_mapDepth3TSmaxA225_HF");
  Map_Ampl[6][4][4] = (TH2F *)hfile->Get("h_mapDepth4TSmaxA225_HF");

  HistAmpl[6][1] = (TH1F *)hfile->Get("h_TSmaxA_HB");
  HistAmpl[6][2] = (TH1F *)hfile->Get("h_TSmaxA_HE");
  HistAmpl[6][3] = (TH1F *)hfile->Get("h_TSmaxA_HO");
  HistAmpl[6][4] = (TH1F *)hfile->Get("h_TSmaxA_HF");

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
      //          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
      //        int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
      //          int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
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
      cONE->Divide(3, 1);
      if (test == 2 && sub == 2) {
        cONE->cd(2);
        TH1F *kjkjkhj2 = (TH1F *)hfile->Get("h_AmplitudeHEtest1");
        kjkjkhj2->Draw("");
        kjkjkhj2->SetTitle("HE, All Depth: shunt1");
        cONE->cd(3);
        TH1F *kjkjkhj3 = (TH1F *)hfile->Get("h_AmplitudeHEtest6");
        kjkjkhj3->Draw("");
        kjkjkhj3->SetTitle("HE, All Depth: shunt6");
      }
      if (test == 2 && sub == 1) {
        cONE->cd(2);
        TH1F *kjkjkhb2 = (TH1F *)hfile->Get("h_AmplitudeHBtest1");
        kjkjkhb2->Draw("");
        kjkjkhb2->SetTitle("HB, All Depth: shunt1");
        cONE->cd(3);
        TH1F *kjkjkhb3 = (TH1F *)hfile->Get("h_AmplitudeHBtest6");
        kjkjkhb3->Draw("");
        kjkjkhb3->SetTitle("HB, All Depth: shunt6");
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
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);  //Tn
      if (test == 6)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);  //Tx
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

  TH2F *Map_Calib[5][5];  // 2d histogramm for subdet, depth

  Map_Calib[1][1] = (TH2F *)hfile->Get("h_map_HB");
  Map_Calib[1][2] = (TH2F *)hfile->Get("h_map_HB");
  Map_Calib[2][1] = (TH2F *)hfile->Get("h_map_HE");
  Map_Calib[2][2] = (TH2F *)hfile->Get("h_map_HE");
  Map_Calib[2][3] = (TH2F *)hfile->Get("h_map_HE");
  Map_Calib[3][4] = (TH2F *)hfile->Get("h_map_HO");
  Map_Calib[4][1] = (TH2F *)hfile->Get("h_map_HF");
  Map_Calib[4][2] = (TH2F *)hfile->Get("h_map_HF");

  //+++++++++++++++++++++++++++++
  //Test 0 Entries
  //+++++++++++++++++++++++++++++

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //       if (sub==1) cHB->Divide(2,1);
                                        //       if (sub==2) cHE->Divide(3,1);
    cONE->Divide(1, 1);
    //      if (sub==4) cHB->Divide(2,1);
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //       int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depth
    int k = 1;
    cONE->cd(k);
    //          if (sub==1) cHB->cd(k);
    //          if (sub==2) cHE->cd(k);
    if (sub == 3)
      k = 4;
    //	    if (sub==4) cHB->cd(k);
    gPad->SetGridy();
    gPad->SetGridx();
    gPad->SetLogz();
    if (sub == 1)
      sprintf(str, "HB");
    if (sub == 2)
      sprintf(str, "HE");
    if (sub == 3)
      sprintf(str, "HO");
    if (sub == 4)
      sprintf(str, "HF");
    Map_Calib[sub][k]->SetTitle(str);
    Map_Calib[sub][k]->SetXTitle("#eta \b");
    Map_Calib[sub][k]->SetYTitle("#phi \b");
    Map_Calib[sub][k]->SetZTitle("Number of events\b");
    if (sub == 3)
      Map_Calib[sub][k]->SetTitleOffset(0.8, "Z");
    Map_Calib[sub][k]->Draw("COLZ");
    Map_Calib[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
    //            Map_Calib[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
    //            if (sub==1) {cHB->Modified(); cHB->Update();}
    //            if (sub==2) {cHE->Modified(); cHE->Update();}
    cONE->Modified();
    cONE->Update();
    //            if (sub==4) {cHB->Modified(); cHB->Update();}
    //       }//end depth

    if (sub == 1) {
      cONE->Print("MapRateCalibEntryHB.png");
      cONE->Clear();
    }
    if (sub == 2) {
      cONE->Print("MapRateCalibEntryHE.png");
      cONE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateCalibEntryHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cONE->Print("MapRateCalibEntryHF.png");
      cONE->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++
  //Test 11 (Cc) Rate of Cap ID errors for calibration channels
  //+++++++++++++++++++++++++++++

  Map_Ampl[11][1][1] = (TH2F *)hfile->Get("h_mapCapCalib047_HB");
  Map_Ampl[11][1][2] = (TH2F *)hfile->Get("h_mapCapCalib047_HB");
  Map_Ampl[11][2][1] = (TH2F *)hfile->Get("h_mapCapCalib047_HE");
  Map_Ampl[11][2][2] = (TH2F *)hfile->Get("h_mapCapCalib047_HE");
  Map_Ampl[11][2][3] = (TH2F *)hfile->Get("h_mapCapCalib047_HE");
  Map_Ampl[11][3][4] = (TH2F *)hfile->Get("h_mapCapCalib047_HO");
  Map_Ampl[11][4][1] = (TH2F *)hfile->Get("h_mapCapCalib047_HF");
  Map_Ampl[11][4][2] = (TH2F *)hfile->Get("h_mapCapCalib047_HF");

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                        //       if (sub==1) cHB->Divide(2,1);
                                        //       if (sub==2) cHE->Divide(3,1);
    cONE->Divide(1, 1);
    //      if (sub==4) cHB->Divide(2,1);
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //       int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depth
    int k = 1;
    cONE->cd(k);
    //          if (sub==1) cHB->cd(k);
    //          if (sub==2) cHE->cd(k);
    if (sub == 3)
      k = 4;
    //	    if (sub==4) cHB->cd(k);
    Map_Ampl[11][sub][k]->Divide(Map_Ampl[11][sub][k], Map_Calib[sub][k], 1, 1, "B");
    gPad->SetGridy();
    gPad->SetGridx();
    gPad->SetLogz();
    if (sub == 1)
      sprintf(str, "HB");
    if (sub == 2)
      sprintf(str, "HE");
    if (sub == 3)
      sprintf(str, "HO");
    if (sub == 4)
      sprintf(str, "HF");
    Map_Ampl[11][sub][k]->SetTitle(str);
    Map_Ampl[11][sub][k]->SetXTitle("#eta \b");
    Map_Ampl[11][sub][k]->SetYTitle("#phi \b");
    Map_Ampl[11][sub][k]->SetZTitle("Rate \b");
    Map_Ampl[11][sub][k]->SetTitleOffset(0.75, "Z");
    Map_Ampl[11][sub][k]->Draw("COLZ");
    Map_Ampl[11][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
    Map_Ampl[11][sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
    //            if (sub==1) {cHB->Modified(); cHB->Update();}
    //            if (sub==2) {cHE->Modified(); cHE->Update();}
    cONE->Modified();
    cONE->Update();
    //            if (sub==4) {cHB->Modified(); cHB->Update();}
    //       }//end depth

    if (sub == 1) {
      cONE->Print("MapRateCapCalibHB.png");
      cONE->Clear();
    }
    if (sub == 2) {
      cONE->Print("MapRateCapCalibHE.png");
      cONE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateCapCalibHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cONE->Print("MapRateCapCalibHF.png");
      cONE->Clear();
    }
  }  // end sub

  //+++++++++++++++++++++++++++++
  //Test 12 (Ac) ADC amplitude for calibration chanels
  //+++++++++++++++++++++++++++++

  Map_Ampl[12][1][1] = (TH2F *)hfile->Get("h_mapADCCalib047_HB");
  Map_Ampl[12][1][2] = (TH2F *)hfile->Get("h_mapADCCalib047_HB");
  Map_Ampl[12][2][1] = (TH2F *)hfile->Get("h_mapADCCalib047_HE");
  Map_Ampl[12][2][2] = (TH2F *)hfile->Get("h_mapADCCalib047_HE");
  Map_Ampl[12][2][3] = (TH2F *)hfile->Get("h_mapADCCalib047_HE");
  Map_Ampl[12][3][4] = (TH2F *)hfile->Get("h_mapADCCalib047_HO");
  Map_Ampl[12][4][1] = (TH2F *)hfile->Get("h_mapADCCalib047_HF");
  Map_Ampl[12][4][2] = (TH2F *)hfile->Get("h_mapADCCalib047_HF");

  HistAmpl[12][1] = (TH1F *)hfile->Get("h_ADCCalib_HB");
  HistAmpl[12][2] = (TH1F *)hfile->Get("h_ADCCalib_HE");
  HistAmpl[12][3] = (TH1F *)hfile->Get("h_ADCCalib_HO");
  HistAmpl[12][4] = (TH1F *)hfile->Get("h_ADCCalib_HF");

  //+++++++++++++++++++++++++++++
  //Test 13 (Wc) Rate of RMS
  //+++++++++++++++++++++++++++++

  Map_Ampl[13][1][1] = (TH2F *)hfile->Get("h_mapWidthCalib047_HB");
  Map_Ampl[13][1][2] = (TH2F *)hfile->Get("h_mapWidthCalib047_HB");
  Map_Ampl[13][2][1] = (TH2F *)hfile->Get("h_mapWidthCalib047_HE");
  Map_Ampl[13][2][2] = (TH2F *)hfile->Get("h_mapWidthCalib047_HE");
  Map_Ampl[13][2][3] = (TH2F *)hfile->Get("h_mapWidthCalib047_HE");
  Map_Ampl[13][3][4] = (TH2F *)hfile->Get("h_mapWidthCalib047_HO");
  Map_Ampl[13][4][1] = (TH2F *)hfile->Get("h_mapWidthCalib047_HF");
  Map_Ampl[13][4][2] = (TH2F *)hfile->Get("h_mapWidthCalib047_HF");

  HistAmpl[13][1] = (TH1F *)hfile->Get("h_WidthCalib_HB");
  HistAmpl[13][2] = (TH1F *)hfile->Get("h_WidthCalib_HE");
  HistAmpl[13][3] = (TH1F *)hfile->Get("h_WidthCalib_HO");
  HistAmpl[13][4] = (TH1F *)hfile->Get("h_WidthCalib_HF");

  //+++++++++++++++++++++++++++++
  //Test 14 (Rc) Rate of ratio 4 near max TS/ All TS
  //+++++++++++++++++++++++++++++

  Map_Ampl[14][1][1] = (TH2F *)hfile->Get("h_mapRatioCalib047_HB");
  Map_Ampl[14][1][2] = (TH2F *)hfile->Get("h_mapRatioCalib047_HB");
  Map_Ampl[14][2][1] = (TH2F *)hfile->Get("h_mapRatioCalib047_HE");
  Map_Ampl[14][2][2] = (TH2F *)hfile->Get("h_mapRatioCalib047_HE");
  Map_Ampl[14][2][3] = (TH2F *)hfile->Get("h_mapRatioCalib047_HE");
  Map_Ampl[14][3][4] = (TH2F *)hfile->Get("h_mapRatioCalib047_HO");
  Map_Ampl[14][4][1] = (TH2F *)hfile->Get("h_mapRatioCalib047_HF");
  Map_Ampl[14][4][2] = (TH2F *)hfile->Get("h_mapRatioCalib047_HF");

  HistAmpl[14][1] = (TH1F *)hfile->Get("h_RatioCalib_HB");
  HistAmpl[14][2] = (TH1F *)hfile->Get("h_RatioCalib_HE");
  HistAmpl[14][3] = (TH1F *)hfile->Get("h_RatioCalib_HO");
  HistAmpl[14][4] = (TH1F *)hfile->Get("h_RatioCalib_HF");

  //+++++++++++++++++++++++++++++
  //Test 15 (TNc) Mean position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[15][1][1] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HB");
  Map_Ampl[15][1][2] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HB");
  Map_Ampl[15][2][1] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HE");
  Map_Ampl[15][2][2] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HE");
  Map_Ampl[15][2][3] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HE");
  Map_Ampl[15][3][4] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HO");
  Map_Ampl[15][4][1] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HF");
  Map_Ampl[15][4][2] = (TH2F *)hfile->Get("h_mapTSmeanCalib047_HF");

  HistAmpl[15][1] = (TH1F *)hfile->Get("h_TSmeanCalib_HB");
  HistAmpl[15][2] = (TH1F *)hfile->Get("h_TSmeanCalib_HE");
  HistAmpl[15][3] = (TH1F *)hfile->Get("h_TSmeanCalib_HO");
  HistAmpl[15][4] = (TH1F *)hfile->Get("h_TSmeanCalib_HF");

  //+++++++++++++++++++++++++++++
  //Test 16 (TXc) Maximum position in 1-8 TS range
  //+++++++++++++++++++++++++++++

  Map_Ampl[16][1][1] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HB");
  Map_Ampl[16][1][2] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HB");
  Map_Ampl[16][2][1] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HE");
  Map_Ampl[16][2][2] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HE");
  Map_Ampl[16][2][3] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HE");
  Map_Ampl[16][3][4] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HO");
  Map_Ampl[16][4][1] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HF");
  Map_Ampl[16][4][2] = (TH2F *)hfile->Get("h_mapTSmaxCalib047_HF");

  HistAmpl[16][1] = (TH1F *)hfile->Get("h_TSmaxCalib_HB");
  HistAmpl[16][2] = (TH1F *)hfile->Get("h_TSmaxCalib_HE");
  HistAmpl[16][3] = (TH1F *)hfile->Get("h_TSmaxCalib_HO");
  HistAmpl[16][4] = (TH1F *)hfile->Get("h_TSmaxCalib_HF");

  for (int test = 12; test <= 16; test++) {  //Test: 2-Am, 3-Wm, 4-Rm, 5-TNm, 6-TXm,
    for (int sub = 1; sub <= 4; sub++) {     //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
      if (sub == 1)
        cONE->Divide(1, 1);  //cHB->Divide(2,1);
      if (sub == 2)
        cONE->Divide(1, 1);  //cHE->Divide(3,1);
      if (sub == 3)
        cONE->Divide(1, 1);
      if (sub == 4)
        cONE->Divide(1, 1);  //cHB->Divide(2,1);
                             //          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
                             //          int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
                             //          for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depth
      int k = 1;
      if (sub == 1) {
        k = 1;
        cONE->cd(k);
      }  //cHB->cd(k); }
      if (sub == 2) {
        k = 1;
        cONE->cd(k);
      }  //cHE->cd(k); }
      if (sub == 3) {
        k = 4;
        cONE->cd(k - 3);
      }
      if (sub == 4) {
        k = 1;
        cONE->cd(k);
      }  //cHB->cd(k); }
      Map_Ampl[test][sub][k]->Divide(Map_Ampl[test][sub][k], Map_Calib[sub][k], 1, 1, "B");
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogz();
      if (sub == 1)
        sprintf(str, "HB");
      if (sub == 2)
        sprintf(str, "HE");
      if (sub == 3)
        sprintf(str, "HO");
      if (sub == 4)
        sprintf(str, "HF");
      Map_Ampl[test][sub][k]->SetTitle(str);
      Map_Ampl[test][sub][k]->SetXTitle("#eta \b");
      Map_Ampl[test][sub][k]->SetYTitle("#phi \b");
      Map_Ampl[test][sub][k]->SetZTitle("Rate \b");
      Map_Ampl[test][sub][k]->SetTitleOffset(0.8, "Z");
      Map_Ampl[test][sub][k]->Draw("COLZ");
      Map_Ampl[test][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      Map_Ampl[test][sub][k]->GetZaxis()->SetRangeUser(0.00001, 1.);
      //              if (sub==1) {cHB->Modified(); cHB->Update();}
      //              if (sub==2) {cHE->Modified(); cHE->Update();}
      cONE->Modified();
      cONE->Update();
      //              if (sub==4) {cHB->Modified(); cHB->Update();}
      //          }//end depth
      if (test == 12) {
        if (sub == 1) {
          cONE->Print("MapRateAmplCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("MapRateAmplCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateAmplCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("MapRateAmplCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 13) {
        if (sub == 1) {
          cONE->Print("MapRateRMSCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("MapRateRMSCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateRMSCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("MapRateRMSCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 14) {
        if (sub == 1) {
          cONE->Print("MapRate43TStoAllTSCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("MapRate43TStoAllTSCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRate43TStoAllTSCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("MapRate43TStoAllTSCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 15) {
        if (sub == 1) {
          cONE->Print("MapRateMeanPosCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("MapRateMeanPosCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateMeanPosCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("MapRateMeanPosCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 16) {
        if (sub == 1) {
          cONE->Print("MapRateMaxPosCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("MapRateMaxPosCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("MapRateMaxPosCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("MapRateMaxPosCalibHF.png");
          cONE->Clear();
        }
      }

      cONE->Divide(1, 1);
      cONE->cd(1);
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogy();
      if (sub == 1)
        HistAmpl[test][sub]->SetTitle("HB, All Depth");
      if (sub == 2)
        HistAmpl[test][sub]->SetTitle("HE, All Depth");
      if (sub == 3)
        HistAmpl[test][sub]->SetTitle("HO, All Depth");
      if (sub == 4)
        HistAmpl[test][sub]->SetTitle("HF, All Depth");
      if (test == 12)
        HistAmpl[test][sub]->SetXTitle("ADC Amlitude in each event & cell \b");
      if (test == 13)
        HistAmpl[test][sub]->SetXTitle("Amplitude RMS in each event & cell \b");
      if (test == 14)
        HistAmpl[test][sub]->SetXTitle("Ratio in each event & cell \b");
      if (test == 15)
        HistAmpl[test][sub]->SetXTitle("Mean TS position in each event & cell \b");
      if (test == 16)
        HistAmpl[test][sub]->SetXTitle("Max TS position in each event & cell \b");
      HistAmpl[test][sub]->SetYTitle("Number of cell-events \b");
      HistAmpl[test][sub]->SetLineColor(4);
      HistAmpl[test][sub]->SetLineWidth(2);
      HistAmpl[test][sub]->SetTitleOffset(1.4, "Y");
      HistAmpl[test][sub]->Draw("");
      //        HistAmpl[test][sub]->GetYaxis()->SetRangeUser(1., 100.);
      if (test == 12) {
        gPad->SetLogx();
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(1., 10000.);
      }
      if (test == 13)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 5.);
      if (test == 14)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 1.);
      if (test == 15)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);
      if (test == 16)
        HistAmpl[test][sub]->GetXaxis()->SetRangeUser(0., 9.);
      cONE->Modified();
      cONE->Update();
      double min_x[] = {MIN_C[test - 10][sub], MIN_C[test - 10][sub]};
      double min_y[] = {0., 100000000.};
      TGraph *MIN = new TGraph(2, min_x, min_y);
      MIN->SetLineStyle(2);
      MIN->SetLineColor(2);
      MIN->SetLineWidth(2 + 100 * 100);
      MIN->SetFillStyle(3005);
      MIN->SetFillColor(2);
      MIN->Draw("L");
      double max_x[] = {MAX_C[test - 10][sub], MAX_C[test - 10][sub]};
      double max_y[] = {0., 100000000.};
      TGraph *MAX = new TGraph(2, max_x, max_y);
      MAX->SetLineStyle(2);
      MAX->SetLineColor(2);
      MAX->SetLineWidth(-2 - 100 * 100);
      MAX->SetFillStyle(3004);
      MAX->SetFillColor(2);
      MAX->Draw("L");
      if (test == 12) {
        if (sub == 1) {
          cONE->Print("HistAmplCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistAmplCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistAmplCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistAmplCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 13) {
        if (sub == 1) {
          cONE->Print("HistRMSCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistRMSCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistRMSCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistRMSCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 14) {
        if (sub == 1) {
          cONE->Print("Hist43TStoAllTSCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("Hist43TStoAllTSCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("Hist43TStoAllTSCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("Hist43TStoAllTSCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 15) {
        if (sub == 1) {
          cONE->Print("HistMeanPosCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistMeanPosCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistMeanPosCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistMeanPosCalibHF.png");
          cONE->Clear();
        }
      }
      if (test == 16) {
        if (sub == 1) {
          cONE->Print("HistMaxPosCalibHB.png");
          cONE->Clear();
        }
        if (sub == 2) {
          cONE->Print("HistMaxPosCalibHE.png");
          cONE->Clear();
        }
        if (sub == 3) {
          cONE->Print("HistMaxPosCalibHO.png");
          cONE->Clear();
        }
        if (sub == 4) {
          cONE->Print("HistMaxPosCalibHF.png");
          cONE->Clear();
        }
      }
    }  // end sub
  }    //end test

  //+++++++++++++++++++++++++++++
  //Test 21 (GS) Amplitude drift
  //+++++++++++++++++++++++++++++

  Map_Ampl[21][1][1] = (TH2F *)hfile->Get("h_mapDepth1AmplE34_HB");
  Map_Ampl[21][1][2] = (TH2F *)hfile->Get("h_mapDepth2AmplE34_HB");
  Map_Ampl[21][1][3] = (TH2F *)hfile->Get("h_mapDepth3AmplE34_HB");
  Map_Ampl[21][1][4] = (TH2F *)hfile->Get("h_mapDepth4AmplE34_HB");
  Map_Ampl[21][2][1] = (TH2F *)hfile->Get("h_mapDepth1AmplE34_HE");
  Map_Ampl[21][2][2] = (TH2F *)hfile->Get("h_mapDepth2AmplE34_HE");
  Map_Ampl[21][2][3] = (TH2F *)hfile->Get("h_mapDepth3AmplE34_HE");
  Map_Ampl[21][2][4] = (TH2F *)hfile->Get("h_mapDepth4AmplE34_HE");
  Map_Ampl[21][2][5] = (TH2F *)hfile->Get("h_mapDepth5AmplE34_HE");
  Map_Ampl[21][2][6] = (TH2F *)hfile->Get("h_mapDepth6AmplE34_HE");
  Map_Ampl[21][2][7] = (TH2F *)hfile->Get("h_mapDepth7AmplE34_HE");
  Map_Ampl[21][3][4] = (TH2F *)hfile->Get("h_mapDepth4AmplE34_HO");
  Map_Ampl[21][4][1] = (TH2F *)hfile->Get("h_mapDepth1AmplE34_HF");
  Map_Ampl[21][4][2] = (TH2F *)hfile->Get("h_mapDepth2AmplE34_HF");
  Map_Ampl[21][4][3] = (TH2F *)hfile->Get("h_mapDepth3AmplE34_HF");
  Map_Ampl[21][4][4] = (TH2F *)hfile->Get("h_mapDepth4AmplE34_HF");

  TH2F *Map_RefAmpl[5][ALLDEPTH];  // 2D histogramm for subdet, depth
  TH2F *Map_RefSUB[5][ALLDEPTH];   // 2d histogramm for subdet, depth

  Map_RefAmpl[1][1] = (TH2F *)hreffile->Get("h_mapDepth1AmplE34_HB");
  Map_RefAmpl[1][2] = (TH2F *)hreffile->Get("h_mapDepth2AmplE34_HB");
  Map_RefAmpl[1][3] = (TH2F *)hreffile->Get("h_mapDepth3AmplE34_HB");
  Map_RefAmpl[1][4] = (TH2F *)hreffile->Get("h_mapDepth4AmplE34_HB");
  Map_RefAmpl[2][1] = (TH2F *)hreffile->Get("h_mapDepth1AmplE34_HE");
  Map_RefAmpl[2][2] = (TH2F *)hreffile->Get("h_mapDepth2AmplE34_HE");
  Map_RefAmpl[2][3] = (TH2F *)hreffile->Get("h_mapDepth3AmplE34_HE");
  Map_RefAmpl[2][4] = (TH2F *)hreffile->Get("h_mapDepth4AmplE34_HE");
  Map_RefAmpl[2][5] = (TH2F *)hreffile->Get("h_mapDepth5AmplE34_HE");
  Map_RefAmpl[2][6] = (TH2F *)hreffile->Get("h_mapDepth6AmplE34_HE");
  Map_RefAmpl[2][7] = (TH2F *)hreffile->Get("h_mapDepth7AmplE34_HE");
  Map_RefAmpl[3][4] = (TH2F *)hreffile->Get("h_mapDepth4AmplE34_HO");
  Map_RefAmpl[4][1] = (TH2F *)hreffile->Get("h_mapDepth1AmplE34_HF");
  Map_RefAmpl[4][2] = (TH2F *)hreffile->Get("h_mapDepth2AmplE34_HF");
  Map_RefAmpl[4][3] = (TH2F *)hreffile->Get("h_mapDepth3AmplE34_HF");
  Map_RefAmpl[4][4] = (TH2F *)hreffile->Get("h_mapDepth4AmplE34_HF");

  Map_RefSUB[1][1] = (TH2F *)hreffile->Get("h_mapDepth1_HB");
  Map_RefSUB[1][2] = (TH2F *)hreffile->Get("h_mapDepth2_HB");
  Map_RefSUB[1][3] = (TH2F *)hreffile->Get("h_mapDepth3_HB");
  Map_RefSUB[1][4] = (TH2F *)hreffile->Get("h_mapDepth4_HB");
  Map_RefSUB[2][1] = (TH2F *)hreffile->Get("h_mapDepth1_HE");
  Map_RefSUB[2][2] = (TH2F *)hreffile->Get("h_mapDepth2_HE");
  Map_RefSUB[2][3] = (TH2F *)hreffile->Get("h_mapDepth3_HE");
  Map_RefSUB[2][4] = (TH2F *)hreffile->Get("h_mapDepth4_HE");
  Map_RefSUB[2][5] = (TH2F *)hreffile->Get("h_mapDepth5_HE");
  Map_RefSUB[2][6] = (TH2F *)hreffile->Get("h_mapDepth6_HE");
  Map_RefSUB[2][7] = (TH2F *)hreffile->Get("h_mapDepth7_HE");
  Map_RefSUB[3][4] = (TH2F *)hreffile->Get("h_mapDepth4_HO");
  Map_RefSUB[4][1] = (TH2F *)hreffile->Get("h_mapDepth1_HF");
  Map_RefSUB[4][2] = (TH2F *)hreffile->Get("h_mapDepth2_HF");
  Map_RefSUB[4][3] = (TH2F *)hreffile->Get("h_mapDepth3_HF");
  Map_RefSUB[4][4] = (TH2F *)hreffile->Get("h_mapDepth4_HF");

  HistAmplDepth[21][1][1] = new TH1F("diffAmpl_Depth1_HB", "", 100, -10., 10.);
  HistAmplDepth[21][1][2] = new TH1F("diffAmpl_Depth2_HB", "", 100, -10., 10.);
  HistAmplDepth[21][1][3] = new TH1F("diffAmpl_Depth3_HB", "", 100, -10., 10.);
  HistAmplDepth[21][1][4] = new TH1F("diffAmpl_Depth4_HB", "", 100, -10., 10.);
  HistAmplDepth[21][2][1] = new TH1F("diffAmpl_Depth1_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][2] = new TH1F("diffAmpl_Depth2_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][3] = new TH1F("diffAmpl_Depth3_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][4] = new TH1F("diffAmpl_Depth4_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][5] = new TH1F("diffAmpl_Depth5_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][6] = new TH1F("diffAmpl_Depth6_HE", "", 100, -10., 10.);
  HistAmplDepth[21][2][7] = new TH1F("diffAmpl_Depth7_HE", "", 100, -10., 10.);
  HistAmplDepth[21][3][4] = new TH1F("diffAmpl_Depth4_HO", "", 100, -10., 10.);
  HistAmplDepth[21][4][1] = new TH1F("diffAmpl_Depth1_HF", "", 100, -10., 10.);
  HistAmplDepth[21][4][2] = new TH1F("diffAmpl_Depth2_HF", "", 100, -10., 10.);
  HistAmplDepth[21][4][3] = new TH1F("diffAmpl_Depth3_HF", "", 100, -10., 10.);
  HistAmplDepth[21][4][4] = new TH1F("diffAmpl_Depth4_HF", "", 100, -10., 10.);

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
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      Map_Ampl[21][sub][k]->Divide(Map_Ampl[21][sub][k], Map_SUB[sub][k], 1, 1, "B");
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
      Map_Ampl[21][sub][k]->SetTitle(str);
      Map_Ampl[21][sub][k]->SetXTitle("#eta \b");
      Map_Ampl[21][sub][k]->SetYTitle("#phi \b");
      Map_Ampl[21][sub][k]->SetZTitle("Response \b");
      Map_Ampl[21][sub][k]->SetTitleOffset(1.1, "Z");
      Map_Ampl[21][sub][k]->Draw("COLZ");
      Map_Ampl[21][sub][k]->GetYaxis()->SetRangeUser(0, 71.);
      //            Map_Ampl[21][sub][k]->GetZaxis()->SetRangeUser(1., 10.);
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
      cHB->Print("MapRateAmpl1HB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MapRateAmpl1HE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateAmpl1HO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MapRateAmpl1HF.png");
      cHF->Clear();
    }

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

    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      Map_RefAmpl[sub][k]->Divide(Map_RefAmpl[sub][k], Map_RefSUB[sub][k], 1, 1, "B");
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
      Map_RefAmpl[sub][k]->SetTitle(str);
      Map_RefAmpl[sub][k]->SetXTitle("#eta \b");
      Map_RefAmpl[sub][k]->SetYTitle("#phi \b");
      Map_RefAmpl[sub][k]->SetZTitle("Response\b");
      Map_RefAmpl[sub][k]->SetTitleOffset(1.1, "Z");
      Map_RefAmpl[sub][k]->Draw("COLZ");
      Map_RefAmpl[sub][k]->GetYaxis()->SetRangeUser(0, 71.);
      //            Map_RefAmpl[21][sub][k]->GetZaxis()->SetRangeUser(1., 10.);
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
      cHB->Print("MapRateAmpl2HB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MapRateAmpl2HE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateAmpl2HO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MapRateAmpl2HF.png");
      cHF->Clear();
    }

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

    for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
      if (sub == 1)
        cHB->cd(k);
      if (sub == 2)
        cHE->cd(k);
      if (sub == 3)
        cONE->cd(k - 3);
      if (sub == 4)
        cHF->cd(k);
      TH2F *TTT = new TH2F("Map", "Map", 82, -41, 40, 72, 0, 71);
      for (int x = 1; x <= Map_Ampl[21][sub][k]->GetXaxis()->GetNbins(); x++) {
        for (int y = 1; y <= Map_Ampl[21][sub][k]->GetYaxis()->GetNbins(); y++) {
          if (Map_Ampl[21][sub][k]->GetBinContent(x, y) != 0 && Map_RefAmpl[sub][k]->GetBinContent(x, y) != 0) {
            double ccc1 = Map_Ampl[21][sub][k]->GetBinContent(x, y) - Map_RefAmpl[sub][k]->GetBinContent(x, y);
            ccc1 = 100. * ccc1 / Map_Ampl[21][sub][k]->GetBinContent(x, y);  // in %
            HistAmplDepth[21][sub][k]->Fill(ccc1);
            Map_Ampl[21][sub][k]->SetBinContent(x, y, fabs(ccc1));
            if (fabs(ccc1) > porog[sub])
              TTT->SetBinContent(x, y, fabs(ccc1));
            else
              TTT->SetBinContent(x, y, 0);
          }
        }  //end y
      }    //esnd x
      gPad->SetGridy();
      gPad->SetGridx();
      //            gPad->SetLogz();
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      TTT->SetTitle(str);
      TTT->SetXTitle("#eta \b");
      TTT->SetYTitle("#phi \b");
      TTT->SetZTitle("Relative difference, % \b");
      TTT->SetTitleOffset(0.9, "Z");
      TTT->Draw("COLZ");
      TTT->GetYaxis()->SetRangeUser(0, 71.);
      TTT->GetZaxis()->SetRangeUser(0, 10.);
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
      cHB->Print("MapRateAmplDriftHB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("MapRateAmplDriftHE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("MapRateAmplDriftHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("MapRateAmplDriftHF.png");
      cHF->Clear();
    }
    ////////////////////////////////////////////////////////////////////////////////////

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
      gPad->SetLogy();
      if (sub == 1)
        sprintf(str, "HB, Depth%d \b", k);
      if (sub == 2)
        sprintf(str, "HE, Depth%d \b", k);
      if (sub == 3)
        sprintf(str, "HO, Depth%d \b", k);
      if (sub == 4)
        sprintf(str, "HF, Depth%d \b", k);
      HistAmplDepth[21][sub][k]->SetTitle(str);
      HistAmplDepth[21][sub][k]->SetYTitle("Number of cell-events \b");
      HistAmplDepth[21][sub][k]->SetXTitle("Per cent \b");
      HistAmplDepth[21][sub][k]->SetLineColor(4);
      HistAmplDepth[21][sub][k]->SetLineWidth(2);
      HistAmplDepth[21][sub][k]->SetTitleOffset(1.4, "Y");
      HistAmplDepth[21][sub][k]->Draw();
      //          HistAmplDepth[21][sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      HistAmplDepth[21][sub][k]->GetXaxis()->SetRangeUser(-10., 10.);
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
      double min_x[] = {-1 * porog[sub], -1 * porog[sub]};
      double min_y[] = {0., 100000000.};
      TGraph *MIN = new TGraph(2, min_x, min_y);
      MIN->SetLineStyle(2);
      MIN->SetLineColor(2);
      MIN->SetLineWidth(2 + 100 * 100);
      MIN->SetFillStyle(3005);
      MIN->SetFillColor(2);
      MIN->Draw("L");
      double max_x[] = {porog[sub], porog[sub]};
      double max_y[] = {0., 100000000.};
      TGraph *MAX = new TGraph(2, max_x, max_y);
      MAX->SetLineStyle(2);
      MAX->SetLineColor(2);
      MAX->SetLineWidth(-2 - 100 * 100);
      MAX->SetFillStyle(3004);
      MAX->SetFillColor(2);
      MAX->Draw("L");
    }  // end depth
    if (sub == 1) {
      cHB->Print("HistAmplDriftDepthHB.png");
      cHB->Clear();
    }
    if (sub == 2) {
      cHE->Print("HistAmplDriftDepthHE.png");
      cHE->Clear();
    }
    if (sub == 3) {
      cONE->Print("HistAmplDriftDepthHO.png");
      cONE->Clear();
    }
    if (sub == 4) {
      cHF->Print("HistAmplDriftDepthHF.png");
      cHF->Clear();
    }
  }  //end sub

  //+++++++++++++++++++++++++++++++++++
  //Test 31, 32 Pedestal, pedestalWidths
  //++++++++++++++++++++++++++++++++++++

  Map_Ampl[31][1][1] = (TH2F *)hfile->Get("h_mapDepth1pedestal_HB");
  Map_Ampl[31][1][2] = (TH2F *)hfile->Get("h_mapDepth2pedestal_HB");
  Map_Ampl[31][1][3] = (TH2F *)hfile->Get("h_mapDepth3pedestal_HB");
  Map_Ampl[31][1][4] = (TH2F *)hfile->Get("h_mapDepth4pedestal_HB");
  Map_Ampl[31][2][1] = (TH2F *)hfile->Get("h_mapDepth1pedestal_HE");
  Map_Ampl[31][2][2] = (TH2F *)hfile->Get("h_mapDepth2pedestal_HE");
  Map_Ampl[31][2][3] = (TH2F *)hfile->Get("h_mapDepth3pedestal_HE");
  Map_Ampl[31][2][4] = (TH2F *)hfile->Get("h_mapDepth4pedestal_HE");
  Map_Ampl[31][2][5] = (TH2F *)hfile->Get("h_mapDepth5pedestal_HE");
  Map_Ampl[31][2][6] = (TH2F *)hfile->Get("h_mapDepth6pedestal_HE");
  Map_Ampl[31][2][7] = (TH2F *)hfile->Get("h_mapDepth7pedestal_HE");
  Map_Ampl[31][3][4] = (TH2F *)hfile->Get("h_mapDepth4pedestal_HO");
  Map_Ampl[31][4][1] = (TH2F *)hfile->Get("h_mapDepth1pedestal_HF");
  Map_Ampl[31][4][2] = (TH2F *)hfile->Get("h_mapDepth2pedestal_HF");
  Map_Ampl[31][4][3] = (TH2F *)hfile->Get("h_mapDepth3pedestal_HF");
  Map_Ampl[31][4][4] = (TH2F *)hfile->Get("h_mapDepth4pedestal_HF");

  Map_Ampl[32][1][1] = (TH2F *)hfile->Get("h_mapDepth1pedestalw_HB");
  Map_Ampl[32][1][2] = (TH2F *)hfile->Get("h_mapDepth2pedestalw_HB");
  Map_Ampl[32][1][3] = (TH2F *)hfile->Get("h_mapDepth3pedestalw_HB");
  Map_Ampl[32][1][4] = (TH2F *)hfile->Get("h_mapDepth4pedestalw_HB");
  Map_Ampl[32][2][1] = (TH2F *)hfile->Get("h_mapDepth1pedestalw_HE");
  Map_Ampl[32][2][2] = (TH2F *)hfile->Get("h_mapDepth2pedestalw_HE");
  Map_Ampl[32][2][3] = (TH2F *)hfile->Get("h_mapDepth3pedestalw_HE");
  Map_Ampl[32][2][4] = (TH2F *)hfile->Get("h_mapDepth4pedestalw_HE");
  Map_Ampl[32][2][5] = (TH2F *)hfile->Get("h_mapDepth5pedestalw_HE");
  Map_Ampl[32][2][6] = (TH2F *)hfile->Get("h_mapDepth6pedestalw_HE");
  Map_Ampl[32][2][7] = (TH2F *)hfile->Get("h_mapDepth7pedestalw_HE");
  Map_Ampl[32][3][4] = (TH2F *)hfile->Get("h_mapDepth4pedestalw_HO");
  Map_Ampl[32][4][1] = (TH2F *)hfile->Get("h_mapDepth1pedestalw_HF");
  Map_Ampl[32][4][2] = (TH2F *)hfile->Get("h_mapDepth2pedestalw_HF");
  Map_Ampl[32][4][3] = (TH2F *)hfile->Get("h_mapDepth3pedestalw_HF");
  Map_Ampl[32][4][4] = (TH2F *)hfile->Get("h_mapDepth4pedestalw_HF");

  HistPed[1][1][0] = (TH1F *)hfile->Get("h_pedestal0_HB");
  HistPed[1][1][1] = (TH1F *)hfile->Get("h_pedestal1_HB");
  HistPed[1][1][2] = (TH1F *)hfile->Get("h_pedestal2_HB");
  HistPed[1][1][3] = (TH1F *)hfile->Get("h_pedestal3_HB");
  HistPed[2][1][0] = (TH1F *)hfile->Get("h_pedestalw0_HB");
  HistPed[2][1][1] = (TH1F *)hfile->Get("h_pedestalw1_HB");
  HistPed[2][1][2] = (TH1F *)hfile->Get("h_pedestalw2_HB");
  HistPed[2][1][3] = (TH1F *)hfile->Get("h_pedestalw3_HB");

  HistPed[1][2][0] = (TH1F *)hfile->Get("h_pedestal0_HE");
  HistPed[1][2][1] = (TH1F *)hfile->Get("h_pedestal1_HE");
  HistPed[1][2][2] = (TH1F *)hfile->Get("h_pedestal2_HE");
  HistPed[1][2][3] = (TH1F *)hfile->Get("h_pedestal3_HE");
  HistPed[2][2][0] = (TH1F *)hfile->Get("h_pedestalw0_HE");
  HistPed[2][2][1] = (TH1F *)hfile->Get("h_pedestalw1_HE");
  HistPed[2][2][2] = (TH1F *)hfile->Get("h_pedestalw2_HE");
  HistPed[2][2][3] = (TH1F *)hfile->Get("h_pedestalw3_HE");

  HistPed[1][3][0] = (TH1F *)hfile->Get("h_pedestal0_HO");
  HistPed[1][3][1] = (TH1F *)hfile->Get("h_pedestal1_HO");
  HistPed[1][3][2] = (TH1F *)hfile->Get("h_pedestal2_HO");
  HistPed[1][3][3] = (TH1F *)hfile->Get("h_pedestal3_HO");
  HistPed[2][3][0] = (TH1F *)hfile->Get("h_pedestalw0_HO");
  HistPed[2][3][1] = (TH1F *)hfile->Get("h_pedestalw1_HO");
  HistPed[2][3][2] = (TH1F *)hfile->Get("h_pedestalw2_HO");
  HistPed[2][3][3] = (TH1F *)hfile->Get("h_pedestalw3_HO");

  HistPed[1][4][0] = (TH1F *)hfile->Get("h_pedestal0_HF");
  HistPed[1][4][1] = (TH1F *)hfile->Get("h_pedestal1_HF");
  HistPed[1][4][2] = (TH1F *)hfile->Get("h_pedestal2_HF");
  HistPed[1][4][3] = (TH1F *)hfile->Get("h_pedestal3_HF");
  HistPed[2][4][0] = (TH1F *)hfile->Get("h_pedestalw0_HF");
  HistPed[2][4][1] = (TH1F *)hfile->Get("h_pedestalw1_HF");
  HistPed[2][4][2] = (TH1F *)hfile->Get("h_pedestalw2_HF");
  HistPed[2][4][3] = (TH1F *)hfile->Get("h_pedestalw3_HF");

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
      //          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
      //        int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
      //          int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
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
  Map_Ped[1][1] = (TH2F *)hfile->Get("h2_pedvsampl_HB");
  Map_Ped[1][2] = (TH2F *)hfile->Get("h2_pedvsampl_HE");
  Map_Ped[1][3] = (TH2F *)hfile->Get("h2_pedvsampl_HO");
  Map_Ped[1][4] = (TH2F *)hfile->Get("h2_pedvsampl_HF");
  Map_Ped[2][1] = (TH2F *)hfile->Get("h2_pedwvsampl_HB");
  Map_Ped[2][2] = (TH2F *)hfile->Get("h2_pedwvsampl_HE");
  Map_Ped[2][3] = (TH2F *)hfile->Get("h2_pedwvsampl_HO");
  Map_Ped[2][4] = (TH2F *)hfile->Get("h2_pedwvsampl_HF");
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
  hist_GoodTSshape[1] = (TH1F *)hfile->Get("h_shape_good_channels_HB");
  hist_GoodTSshape[2] = (TH1F *)hfile->Get("h_shape_good_channels_HE");
  hist_GoodTSshape[3] = (TH1F *)hfile->Get("h_shape_good_channels_HO");
  hist_GoodTSshape[4] = (TH1F *)hfile->Get("h_shape_good_channels_HF");

  hist_GoodTSshape0[1] = (TH1F *)hfile->Get("h_shape0_good_channels_HB");
  hist_GoodTSshape0[2] = (TH1F *)hfile->Get("h_shape0_good_channels_HE");
  hist_GoodTSshape0[3] = (TH1F *)hfile->Get("h_shape0_good_channels_HO");
  hist_GoodTSshape0[4] = (TH1F *)hfile->Get("h_shape0_good_channels_HF");

  hist_BadTSshape[1] = (TH1F *)hfile->Get("h_shape_bad_channels_HB");
  hist_BadTSshape[2] = (TH1F *)hfile->Get("h_shape_bad_channels_HE");
  hist_BadTSshape[3] = (TH1F *)hfile->Get("h_shape_bad_channels_HO");
  hist_BadTSshape[4] = (TH1F *)hfile->Get("h_shape_bad_channels_HF");

  hist_BadTSshape0[1] = (TH1F *)hfile->Get("h_shape0_bad_channels_HB");
  hist_BadTSshape0[2] = (TH1F *)hfile->Get("h_shape0_bad_channels_HE");
  hist_BadTSshape0[3] = (TH1F *)hfile->Get("h_shape0_bad_channels_HO");
  hist_BadTSshape0[4] = (TH1F *)hfile->Get("h_shape0_bad_channels_HF");

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

  /*
 //+++++++++++++++++++++++++++++++++++  
//Test 42 ADC in TS distributions   
//++++++++++++++++++++++++++++++++++++
 
std::cout<<" We are here to print ADC "<<std::endl; 

  hist_ADC_All[1] = (TH1F*)hfile->Get("h_ADC_HB");
  hist_ADC_All[2] = (TH1F*)hfile->Get("h_ADC_HE");
  hist_ADC_All[3] = (TH1F*)hfile->Get("h_ADC_HO");
  hist_ADC_All[4] = (TH1F*)hfile->Get("h_ADC_HF");


  hist_ADC_DS[1][1] = (TH1F*)hfile->Get("h_ADC_HBdepth1");
  hist_ADC_DS[1][2] = (TH1F*)hfile->Get("h_ADC_HBdepth2");
  hist_ADC_DS[2][1] = (TH1F*)hfile->Get("h_ADC_HEdepth1");
  hist_ADC_DS[2][2] = (TH1F*)hfile->Get("h_ADC_HEdepth2");
  hist_ADC_DS[2][3] = (TH1F*)hfile->Get("h_ADC_HEdepth3");
  hist_ADC_DS[3][4] = (TH1F*)hfile->Get("h_ADC_HOdepth4");
  hist_ADC_DS[4][1] = (TH1F*)hfile->Get("h_ADC_HFdepth1");
  hist_ADC_DS[4][2] = (TH1F*)hfile->Get("h_ADC_HFdepth2");


  cONE->Clear();   
  cONE->Divide(1,1);
  cONE->cd(1);

  for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF         
      gPad->SetGridy();
      gPad->SetGridx();
      gPad->SetLogy();
      hist_ADC_All[sub]->SetXTitle("ADC counts \b");
      hist_ADC_All[sub]->SetYTitle("Number of channels-events \b");
      if (sub==1) sprintf(str,"ADC counts Distribution HB\b");
      if (sub==2) sprintf(str,"ADC counts Distribution HE\b");
      if (sub==3) sprintf(str,"ADC counts Distribution HO\b");
      if (sub==4) sprintf(str,"ADC counts Distribution HF\b");
      hist_ADC_All[sub]->SetTitle(str);
      hist_ADC_All[sub]->Draw("");
      // hist_ADC_All[sub]->GetYaxis()->SetRangeUser(0, 72.);
      hist_ADC_All[sub]->GetXaxis()->SetRangeUser(0.000, 1000.);
      cONE->Modified(); cONE->Update();
      if (sub==1) {cONE->Print("Hist_ADC_HB_All.png"); cONE->Clear();} 
      if (sub==2) {cONE->Print("Hist_ADC_HE_All.png"); cONE->Clear();}
      if (sub==3) {cONE->Print("Hist_ADC_HO_All.png"); cONE->Clear();}
      if (sub==4) {cONE->Print("Hist_ADC_HF_All.png"); cONE->Clear();} 
  }// end sub      
  

  for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
          if (sub==1) cHB->Divide(2,1);
          if (sub==2) cHE->Divide(3,1);
          if (sub==3) cONE->Divide(1,1);
          if (sub==4) cHB->Divide(2,1);
          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
          int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet	
          for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depths 
              if (sub==1) cHB->cd(k); 
              if (sub==2) cHE->cd(k);
	      if (sub==3) cONE->cd(k-3);
	      if (sub==4) cHB->cd(k); 
      	      gPad->SetGridy();
              gPad->SetGridx();
              gPad->SetLogy();
              hist_ADC_DS[sub][k]->SetXTitle("ADC counts \b");
              hist_ADC_DS[sub][k]->SetYTitle("Number of channels-events \b");
              if (sub==1) sprintf(str,"HB, Depth%d \b", k);
              if (sub==2) sprintf(str,"HE, Depth%d \b", k);
              if (sub==3) sprintf(str,"HO, Depth%d \b", k);
              if (sub==4) sprintf(str,"HF, Depth%d \b", k); 
              hist_ADC_DS[sub][k]->SetTitle(str);
              hist_ADC_DS[sub][k]->Draw("");
           // hist_ADC_DS[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
              hist_ADC_DS[sub][k]->GetXaxis()->SetRangeUser(0.000, 1000.);
              if (sub==1) {cHB->Modified(); cHB->Update();} 
              if (sub==2) {cHE->Modified(); cHE->Update();}
              if (sub==3) {cONE->Modified();cONE->Update();}
              if (sub==4) {cHB->Modified(); cHB->Update();} 
          }//end depth    
	  if (sub==1) {cHB->Print("Hist_ADC_HB_DS.png"); cHB->Clear();} 
          if (sub==2) {cHE->Print("Hist_ADC_HE_DS.png"); cHE->Clear();}
          if (sub==3) {cONE->Print("Hist_ADC_HO_DS.png"); cONE->Clear();}
          if (sub==4) {cHB->Print("Hist_ADC_HF_DS.png"); cHB->Clear();} 
  }// end sub


 //+++++++++++++++++++++++++++++++++++  
//Test 43 Sum ADC in TS distributions   
//++++++++++++++++++++++++++++++++++++

  hist_SumADC[1][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HB");
  hist_SumADC[1][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HB");
  hist_SumADC[2][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HE");
  hist_SumADC[2][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HE");
  hist_SumADC[2][3] = (TH1F*)hfile->Get("h_sumamplitude_depth3_HE");
  hist_SumADC[3][4] = (TH1F*)hfile->Get("h_sumamplitude_depth4_HO");
  hist_SumADC[4][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HF");
  hist_SumADC[4][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HF");

  hist_SumADC0[1][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HB0");
  hist_SumADC0[1][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HB0");
  hist_SumADC0[2][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HE0");
  hist_SumADC0[2][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HE0");
  hist_SumADC0[2][3] = (TH1F*)hfile->Get("h_sumamplitude_depth3_HE0");
  hist_SumADC0[3][4] = (TH1F*)hfile->Get("h_sumamplitude_depth4_HO0");
  hist_SumADC0[4][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HF0");
  hist_SumADC0[4][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HF0");

  hist_SumADC1[1][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HB1");
  hist_SumADC1[1][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HB1");
  hist_SumADC1[2][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HE1");
  hist_SumADC1[2][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HE1");
  hist_SumADC1[2][3] = (TH1F*)hfile->Get("h_sumamplitude_depth3_HE1");
  hist_SumADC1[3][4] = (TH1F*)hfile->Get("h_sumamplitude_depth4_HO1");
  hist_SumADC1[4][1] = (TH1F*)hfile->Get("h_sumamplitude_depth1_HF1");
  hist_SumADC1[4][2] = (TH1F*)hfile->Get("h_sumamplitude_depth2_HF1");

  for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
          if (sub==1) cHB->Divide(2,1);
          if (sub==2) cHE->Divide(3,1);
          if (sub==3) cONE->Divide(1,1);
          if (sub==4) cHB->Divide(2,1);
          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
          int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet	
          for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depths 
              if (sub==1) cHB->cd(k); 
              if (sub==2) cHE->cd(k);
	      if (sub==3) cONE->cd(k-3);
	      if (sub==4) cHB->cd(k); 
      	      gPad->SetGridy();
              gPad->SetGridx();
              gPad->SetLogy();
              hist_SumADC[sub][k]->SetXTitle("ADC sum \b");
              hist_SumADC[sub][k]->SetYTitle("Number of channels-events \b");
              if (sub==1) sprintf(str,"HB, Depth%d \b", k);
              if (sub==2) sprintf(str,"HE, Depth%d \b", k);
              if (sub==3) sprintf(str,"HO, Depth%d \b", k);
              if (sub==4) sprintf(str,"HF, Depth%d \b", k); 
              hist_SumADC[sub][k]->SetTitle(str);
              hist_SumADC[sub][k]->Draw("");
           // hist_SumADC[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
           // hist_SumADC[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
              if (sub==1) {cHB->Modified(); cHB->Update();} 
              if (sub==2) {cHE->Modified(); cHE->Update();}
              if (sub==3) {cONE->Modified();cONE->Update();}
              if (sub==4) {cHB->Modified(); cHB->Update();} 
          }//end depth    
	  if (sub==1) {cHB->Print("Hist_SumADC_HB.png"); cHB->Clear();} 
          if (sub==2) {cHE->Print("Hist_SumADC_HE.png"); cHE->Clear();}
          if (sub==3) {cONE->Print("Hist_SumADC_HO.png"); cONE->Clear();}
          if (sub==4) {cHB->Print("Hist_SumADC_HF.png"); cHB->Clear();} 
  }// end sub


  for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
          if (sub==1) cHB->Divide(2,1);
          if (sub==2) cHE->Divide(3,1);
          if (sub==3) cONE->Divide(1,1);
          if (sub==4) cHB->Divide(2,1);
          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
          int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet	
          for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depths 
              if (sub==1) cHB->cd(k); 
              if (sub==2) cHE->cd(k);
	      if (sub==3) cONE->cd(k-3);
	      if (sub==4) cHB->cd(k); 
      	      gPad->SetGridy();
              gPad->SetGridx();
              gPad->SetLogy();
              hist_SumADC0[sub][k]->SetXTitle("ADC sum \b");
              hist_SumADC0[sub][k]->SetYTitle("Number of channels-events \b");
              if (sub==1) sprintf(str,"HB, Depth%d \b", k);
              if (sub==2) sprintf(str,"HE, Depth%d \b", k);
              if (sub==3) sprintf(str,"HO, Depth%d \b", k);
              if (sub==4) sprintf(str,"HF, Depth%d \b", k); 
              hist_SumADC0[sub][k]->SetTitle(str);
              hist_SumADC0[sub][k]->Draw("");
           // hist_SumADC0[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
           // hist_SumADC0[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
              if (sub==1) {cHB->Modified(); cHB->Update();} 
              if (sub==2) {cHE->Modified(); cHE->Update();}
              if (sub==3) {cONE->Modified();cONE->Update();}
              if (sub==4) {cHB->Modified(); cHB->Update();} 
          }//end depth    
	  if (sub==1) {cHB->Print("Hist_SumADC_HB0.png"); cHB->Clear();} 
          if (sub==2) {cHE->Print("Hist_SumADC_HE0.png"); cHE->Clear();}
          if (sub==3) {cONE->Print("Hist_SumADC_HO0.png"); cONE->Clear();}
          if (sub==4) {cHB->Print("Hist_SumADC_HF0.png"); cHB->Clear();} 
  }// end sub

  for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
          if (sub==1) cHB->Divide(2,1);
          if (sub==2) cHE->Divide(3,1);
          if (sub==3) cONE->Divide(1,1);
          if (sub==4) cHB->Divide(2,1);
          int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
          int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet	
          for (int k=k_min[sub];k<=k_max[sub];k++) {  //Depths 
              if (sub==1) cHB->cd(k); 
              if (sub==2) cHE->cd(k);
	      if (sub==3) cONE->cd(k-3);
	      if (sub==4) cHB->cd(k); 
      	      gPad->SetGridy();
              gPad->SetGridx();
              gPad->SetLogy();
              hist_SumADC1[sub][k]->SetXTitle("ADC sum \b");
              hist_SumADC1[sub][k]->SetYTitle("Number of channels-events \b");
              if (sub==1) sprintf(str,"HB, Depth%d \b", k);
              if (sub==2) sprintf(str,"HE, Depth%d \b", k);
              if (sub==3) sprintf(str,"HO, Depth%d \b", k);
              if (sub==4) sprintf(str,"HF, Depth%d \b", k); 
              hist_SumADC1[sub][k]->SetTitle(str);
              hist_SumADC1[sub][k]->Draw("");
           // hist_SumADC1[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
           // hist_SumADC1[sub][k]->GetZaxis()->SetRangeUser(0.0001, 1.);
              if (sub==1) {cHB->Modified(); cHB->Update();} 
              if (sub==2) {cHE->Modified(); cHE->Update();}
              if (sub==3) {cONE->Modified();cONE->Update();}
              if (sub==4) {cHB->Modified(); cHB->Update();} 
          }//end depth    
	  if (sub==1) {cHB->Print("Hist_SumADC_HB1.png"); cHB->Clear();} 
          if (sub==2) {cHE->Print("Hist_SumADC_HE1.png"); cHE->Clear();}
          if (sub==3) {cONE->Print("Hist_SumADC_HO1.png"); cONE->Clear();}
          if (sub==4) {cHB->Print("Hist_SumADC_HF1.png"); cHB->Clear();} 
  }// end sub

*/

  ////////////////////         Iteration Method for Calibration Group:
  ////////////////////         Iteration Method for Calibration Group:
  ////////////////////         Iteration Method for Calibration Group:
  ////////////////////         Iteration Method for Calibration Group:
  ////////////////////         Iteration Method for Calibration Group:

  const int nsub = 4;
  const int ndepth = 7;
  const int neta = 82;
  const int nphi = 72;
  int njeta = neta;
  int njphi = nphi;
  int njdepth = ndepth;
  double alexhe[ndepth][njeta][njphi];
  double variance[ndepth][njeta][njphi];

  const int jposbeg = 57;
  const int jposend = 71;

  const int jnegbeg = 12;
  const int jnegend = 26;

  cout << "Iteration Method for Calibration Group    nsub=  " << nsub << "  ndepth =  " << ndepth << " neta =  " << neta
       << " nphi =  " << nphi << endl;
  cout << "Iteration Method for Calibration Group      njdepth =  " << njdepth << " njeta =  " << njeta
       << " njphi =  " << njphi << endl;
  cout << "Iteration Method for Calibration Group        jposbeg =  " << jposbeg << " jposend =  " << jposend
       << " jnegbeg =  " << jnegbeg << " jnegend =  " << jnegend << endl;

  /*
    depth=5                    all
    i=1-82 in initial histo
    j=i-1 in massives
    
    j= 13   23 (11)           12    25 (!!!)
    j-41 in histo
    -28  -18 (11)          -29    -16
    
    j= 59   69 (11)           57    70 (!!!)
    j-41 in histo
    18   28 (11)           16    29
  */

  //                                   RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR:

  TH2F *maprphinorm1HE1 = (TH2F *)hfile->Get("h_maprphinorm_HE1");
  TH2F *maprphinorm0HE1 = (TH2F *)hfile->Get("h_maprphinorm0_HE1");
  TH2F *maprphinormHE1 = (TH2F *)maprphinorm1HE1->Clone("maprphinormHE1");
  maprphinormHE1->Divide(maprphinorm1HE1, maprphinorm0HE1, 1, 1, "B");
  TH2F *maprphinorm1HE2 = (TH2F *)hfile->Get("h_maprphinorm_HE2");
  TH2F *maprphinorm0HE2 = (TH2F *)hfile->Get("h_maprphinorm0_HE2");
  TH2F *maprphinormHE2 = (TH2F *)maprphinorm1HE2->Clone("maprphinormHE2");
  maprphinormHE2->Divide(maprphinorm1HE2, maprphinorm0HE2, 1, 1, "B");
  TH2F *maprphinorm1HE3 = (TH2F *)hfile->Get("h_maprphinorm_HE3");
  TH2F *maprphinorm0HE3 = (TH2F *)hfile->Get("h_maprphinorm0_HE3");
  TH2F *maprphinormHE3 = (TH2F *)maprphinorm1HE3->Clone("maprphinormHE3");
  maprphinormHE3->Divide(maprphinorm1HE3, maprphinorm0HE3, 1, 1, "B");
  TH2F *maprphinorm1HE4 = (TH2F *)hfile->Get("h_maprphinorm_HE4");
  TH2F *maprphinorm0HE4 = (TH2F *)hfile->Get("h_maprphinorm0_HE4");
  TH2F *maprphinormHE4 = (TH2F *)maprphinorm1HE4->Clone("maprphinormHE4");
  maprphinormHE4->Divide(maprphinorm1HE4, maprphinorm0HE4, 1, 1, "B");
  TH2F *maprphinorm1HE5 = (TH2F *)hfile->Get("h_maprphinorm_HE5");
  TH2F *maprphinorm0HE5 = (TH2F *)hfile->Get("h_maprphinorm0_HE5");
  TH2F *maprphinormHE5 = (TH2F *)maprphinorm1HE5->Clone("maprphinormHE5");
  maprphinormHE5->Divide(maprphinorm1HE5, maprphinorm0HE5, 1, 1, "B");
  TH2F *maprphinorm1HE6 = (TH2F *)hfile->Get("h_maprphinorm_HE6");
  TH2F *maprphinorm0HE6 = (TH2F *)hfile->Get("h_maprphinorm0_HE6");
  TH2F *maprphinormHE6 = (TH2F *)maprphinorm1HE6->Clone("maprphinormHE6");
  maprphinormHE6->Divide(maprphinorm1HE6, maprphinorm0HE6, 1, 1, "B");
  TH2F *maprphinorm1HE7 = (TH2F *)hfile->Get("h_maprphinorm_HE7");
  TH2F *maprphinorm0HE7 = (TH2F *)hfile->Get("h_maprphinorm0_HE7");
  TH2F *maprphinormHE7 = (TH2F *)maprphinorm1HE7->Clone("maprphinormHE7");
  maprphinormHE7->Divide(maprphinorm1HE7, maprphinorm0HE7, 1, 1, "B");
  //=========================== put R into massive alexhe
  for (int jeta = 0; jeta < njeta; jeta++) {
    for (int jphi = 0; jphi < njphi; jphi++) {
      alexhe[0][jeta][jphi] = maprphinormHE1->GetBinContent(jeta + 1, jphi + 1);
      alexhe[1][jeta][jphi] = maprphinormHE2->GetBinContent(jeta + 1, jphi + 1);
      alexhe[2][jeta][jphi] = maprphinormHE3->GetBinContent(jeta + 1, jphi + 1);
      alexhe[3][jeta][jphi] = maprphinormHE4->GetBinContent(jeta + 1, jphi + 1);
      alexhe[4][jeta][jphi] = maprphinormHE5->GetBinContent(jeta + 1, jphi + 1);
      alexhe[5][jeta][jphi] = maprphinormHE6->GetBinContent(jeta + 1, jphi + 1);
      alexhe[6][jeta][jphi] = maprphinormHE7->GetBinContent(jeta + 1, jphi + 1);
    }
  }

  //------------------------  2D-eta/phi-plot: R, averaged over depthes
  //======================================================================
  //======================================================================
  cout << "      R2D-eta/phi-plot: R, averaged over depthes *****" << endl;
  c1x0->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F *Gefz42D = new TH2F("Gefz42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F *Gefz42D0 = new TH2F("Gefz42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F *Gefz42DF = (TH2F *)Gefz42D0->Clone("Gefz42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = alexhe[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0 ) kkk=k2plot+1; //-41 +41 !=0
        if (ccc1 != 0.) {
          Gefz42D->Fill(kkk, jphi, ccc1);
          Gefz42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  Gefz42DF->Divide(Gefz42D, Gefz42D0, 1, 1, "B");  // average A
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gefz42DF->SetMarkerStyle(20);
  Gefz42DF->SetMarkerSize(0.4);
  Gefz42DF->GetZaxis()->SetLabelSize(0.08);
  Gefz42DF->SetXTitle("<R>_depth       #eta  \b");
  Gefz42DF->SetYTitle("      #phi \b");
  Gefz42DF->SetZTitle("<R>_depth \b");
  Gefz42DF->SetMarkerColor(2);
  Gefz42DF->SetLineColor(0);  //      Gefz42DF->SetMaximum(1.000);  //      Gefz42DF->SetMinimum(1.0);
  Gefz42DF->Draw("COLZ");

  /////////////////
  c1x0->Update();
  c1x0->Print("Rhist2IterationMethodHE.png");
  c1x0->Clear();
  // clean-up
  if (Gefz42D)
    delete Gefz42D;
  if (Gefz42D0)
    delete Gefz42D0;
  if (Gefz42DF)
    delete Gefz42DF;

  //====================================================================== 1D plot: R vs phi , averaged over depthes & eta
  //======================================================================
  cout << "      1D plot: R vs phi , averaged over depthes & eta *****" << endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F *Gefz41D = new TH1F("Gefz41D", "", nphi, 0., 72.);
  TH1F *Gefz41D0 = new TH1F("Gefz41D0", "", nphi, 0., 72.);
  TH1F *Gefz41DF = (TH1F *)Gefz41D0->Clone("Gefz41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = alexhe[i][jeta][jphi];
        if (ccc1 != 0.) {
          Gefz41D->Fill(jphi, ccc1);
          Gefz41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     Gefz41D->Sumw2();Gefz41D0->Sumw2();

  Gefz41DF->Divide(Gefz41D, Gefz41D0, 1, 1, "B");  // R averaged over depthes & eta
  Gefz41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {Gefz41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gefz41DF->SetMarkerStyle(20);
  Gefz41DF->SetMarkerSize(1.4);
  Gefz41DF->GetZaxis()->SetLabelSize(0.08);
  Gefz41DF->SetXTitle("#phi  \b");
  Gefz41DF->SetYTitle("  <R> \b");
  Gefz41DF->SetZTitle("<R>_PHI  - AllDepthes \b");
  Gefz41DF->SetMarkerColor(4);
  Gefz41DF->SetLineColor(4);
  Gefz41DF->SetMinimum(0.8);  //      Gefz41DF->SetMaximum(1.000);
  Gefz41DF->Draw("Error");

  /////////////////
  c1x1->Update();
  c1x1->Print("Rhist1IterationMethodHE.png");
  c1x1->Clear();
  // clean-up
  if (Gefz41D)
    delete Gefz41D;
  if (Gefz41D0)
    delete Gefz41D0;
  if (Gefz41DF)
    delete Gefz41DF;

  //========================================================================================== 4
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=1
  cout << "      1D plot: R vs phi , different eta,  depth=1 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection1 = 1;
  TH1F *h2CeffHEpositivedirection1 = new TH1F("h2CeffHEpositivedirection1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F *HEpositivedirection1 = (TH1F *)h2CeffHEpositivedirection1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "444        kcountHEpositivedirection1   =     " << kcountHEpositivedirection1
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection1);
          HEpositivedirection1->SetMarkerStyle(20);
          HEpositivedirection1->SetMarkerSize(0.4);
          HEpositivedirection1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection1->SetXTitle("HEpositivedirection1 \b");
          HEpositivedirection1->SetMarkerColor(2);
          HEpositivedirection1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection1 == 1)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirection1 == 2)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirection1 == 3)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirection1 == 4)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirection1 == 5)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirection1 == 6)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirection1 == 7)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirection1 == 8)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirection1 == 9)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirection1 == 10)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirection1 == 11)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 28; depth = 1 \b");
          if (kcountHEpositivedirection1 == 12)
            HEpositivedirection1->SetXTitle("R for HE+ jeta = 29; depth = 1 \b");
          HEpositivedirection1->Draw("Error");
          kcountHEpositivedirection1++;
          if (kcountHEpositivedirection1 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection1)
    delete h2CeffHEpositivedirection1;

  //========================================================================================== 5
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=2
  cout << "      1D plot: R vs phi , different eta,  depth=2 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection2 = 1;
  TH1F *h2CeffHEpositivedirection2 = new TH1F("h2CeffHEpositivedirection2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F *HEpositivedirection2 = (TH1F *)h2CeffHEpositivedirection2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "555        kcountHEpositivedirection2   =     " << kcountHEpositivedirection2
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection2);
          HEpositivedirection2->SetMarkerStyle(20);
          HEpositivedirection2->SetMarkerSize(0.4);
          HEpositivedirection2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection2->SetXTitle("HEpositivedirection2 \b");
          HEpositivedirection2->SetMarkerColor(2);
          HEpositivedirection2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection2 == 1)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirection2 == 2)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirection2 == 3)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirection2 == 4)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirection2 == 5)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirection2 == 6)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirection2 == 7)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirection2 == 8)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirection2 == 9)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirection2 == 10)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirection2 == 11)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirection2 == 12)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 28; depth = 2 \b");
          if (kcountHEpositivedirection2 == 13)
            HEpositivedirection2->SetXTitle("R for HE+ jeta = 29; depth = 2 \b");
          HEpositivedirection2->Draw("Error");
          kcountHEpositivedirection2++;
          if (kcountHEpositivedirection2 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection2)
    delete h2CeffHEpositivedirection2;
  //========================================================================================== 6
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=3
  cout << "      1D plot: R vs phi , different eta,  depth=3 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection3 = 1;
  TH1F *h2CeffHEpositivedirection3 = new TH1F("h2CeffHEpositivedirection3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F *HEpositivedirection3 = (TH1F *)h2CeffHEpositivedirection3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "666        kcountHEpositivedirection3   =     " << kcountHEpositivedirection3
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection3);
          HEpositivedirection3->SetMarkerStyle(20);
          HEpositivedirection3->SetMarkerSize(0.4);
          HEpositivedirection3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection3->SetXTitle("HEpositivedirection3 \b");
          HEpositivedirection3->SetMarkerColor(2);
          HEpositivedirection3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection3 == 1)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirection3 == 2)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirection3 == 3)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirection3 == 4)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirection3 == 5)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirection3 == 6)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirection3 == 7)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirection3 == 8)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirection3 == 9)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirection3 == 10)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirection3 == 11)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirection3 == 12)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 28; depth = 3 \b");
          if (kcountHEpositivedirection3 == 13)
            HEpositivedirection3->SetXTitle("R for HE+ jeta = 29; depth = 3 \b");
          HEpositivedirection3->Draw("Error");
          kcountHEpositivedirection3++;
          if (kcountHEpositivedirection3 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection3)
    delete h2CeffHEpositivedirection3;
  //========================================================================================== 7
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=4
  cout << "      1D plot: R vs phi , different eta,  depth=4 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection4 = 1;
  TH1F *h2CeffHEpositivedirection4 = new TH1F("h2CeffHEpositivedirection4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F *HEpositivedirection4 = (TH1F *)h2CeffHEpositivedirection4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "777        kcountHEpositivedirection4   =     " << kcountHEpositivedirection4
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection4);
          HEpositivedirection4->SetMarkerStyle(20);
          HEpositivedirection4->SetMarkerSize(0.4);
          HEpositivedirection4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection4->SetXTitle("HEpositivedirection4 \b");
          HEpositivedirection4->SetMarkerColor(2);
          HEpositivedirection4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection4 == 1)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 16; depth = 4 \b");
          if (kcountHEpositivedirection4 == 2)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirection4 == 3)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirection4 == 4)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirection4 == 5)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirection4 == 6)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirection4 == 7)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirection4 == 8)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirection4 == 9)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirection4 == 10)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirection4 == 11)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 27; depth = 4 \b");
          if (kcountHEpositivedirection4 == 12)
            HEpositivedirection4->SetXTitle("R for HE+ jeta = 28; depth = 4 \b");
          HEpositivedirection4->Draw("Error");
          kcountHEpositivedirection4++;
          if (kcountHEpositivedirection4 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection4)
    delete h2CeffHEpositivedirection4;
  //========================================================================================== 8
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=5
  cout << "      1D plot: R vs phi , different eta,  depth=5 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection5 = 1;
  TH1F *h2CeffHEpositivedirection5 = new TH1F("h2CeffHEpositivedirection5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F *HEpositivedirection5 = (TH1F *)h2CeffHEpositivedirection5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          //	       cout<<"888  initial      kcountHEpositivedirection5   =     "<<kcountHEpositivedirection5  <<"   jeta-41=     "<< jeta-41 <<"   jphi=     "<< jphi <<"   alexhe[i][jeta][jphi]=     "<< alexhe[i][jeta][jphi] <<"  depth=     "<< i <<endl;

          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "888        kcountHEpositivedirection5   =     " << kcountHEpositivedirection5
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection5);
          HEpositivedirection5->SetMarkerStyle(20);
          HEpositivedirection5->SetMarkerSize(0.4);
          HEpositivedirection5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection5->SetXTitle("HEpositivedirection5 \b");
          HEpositivedirection5->SetMarkerColor(2);
          HEpositivedirection5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection5 == 1)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirection5 == 2)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirection5 == 3)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirection5 == 4)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirection5 == 5)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirection5 == 6)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirection5 == 7)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirection5 == 8)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirection5 == 9)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirection5 == 10)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 27; depth = 5 \b");
          if (kcountHEpositivedirection5 == 11)
            HEpositivedirection5->SetXTitle("R for HE+ jeta = 28; depth = 5 \b");
          HEpositivedirection5->Draw("Error");
          kcountHEpositivedirection5++;
          if (kcountHEpositivedirection5 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection5)
    delete h2CeffHEpositivedirection5;
  //========================================================================================== 9
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=6
  cout << "      1D plot: R vs phi , different eta,  depth=6 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection6 = 1;
  TH1F *h2CeffHEpositivedirection6 = new TH1F("h2CeffHEpositivedirection6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F *HEpositivedirection6 = (TH1F *)h2CeffHEpositivedirection6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "999        kcountHEpositivedirection6   =     " << kcountHEpositivedirection6
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection6);
          HEpositivedirection6->SetMarkerStyle(20);
          HEpositivedirection6->SetMarkerSize(0.4);
          HEpositivedirection6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection6->SetXTitle("HEpositivedirection6 \b");
          HEpositivedirection6->SetMarkerColor(2);
          HEpositivedirection6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection6 == 1)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirection6 == 2)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirection6 == 3)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirection6 == 4)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirection6 == 5)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirection6 == 6)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirection6 == 7)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirection6 == 8)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirection6 == 9)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 27; depth = 6 \b");
          if (kcountHEpositivedirection6 == 10)
            HEpositivedirection6->SetXTitle("R for HE+ jeta = 28; depth = 6 \b");
          HEpositivedirection6->Draw("Error");
          kcountHEpositivedirection6++;
          if (kcountHEpositivedirection6 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection6)
    delete h2CeffHEpositivedirection6;
  //========================================================================================== 10
  //======================================================================
  //======================================================================1D plot: R vs phi , different eta,  depth=7
  cout << "      1D plot: R vs phi , different eta,  depth=7 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirection7 = 1;
  TH1F *h2CeffHEpositivedirection7 = new TH1F("h2CeffHEpositivedirection7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F *HEpositivedirection7 = (TH1F *)h2CeffHEpositivedirection7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = alexhe[i][jeta][jphi];
          if (ccc1 != 0.) {
            HEpositivedirection7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirection7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1010       kcountHEpositivedirection7   =     " << kcountHEpositivedirection7
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirection7);
          HEpositivedirection7->SetMarkerStyle(20);
          HEpositivedirection7->SetMarkerSize(0.4);
          HEpositivedirection7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirection7->SetXTitle("HEpositivedirection7 \b");
          HEpositivedirection7->SetMarkerColor(2);
          HEpositivedirection7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirection7 == 1)
            HEpositivedirection7->SetXTitle("R for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirection7 == 2)
            HEpositivedirection7->SetXTitle("R for HE+ jeta = 27; depth = 7 \b");
          if (kcountHEpositivedirection7 == 3)
            HEpositivedirection7->SetXTitle("R for HE+ jeta = 28; depth = 7 \b");
          HEpositivedirection7->Draw("Error");
          kcountHEpositivedirection7++;
          if (kcountHEpositivedirection7 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Rhist1IterationMethodDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirection7)
    delete h2CeffHEpositivedirection7;

  //                                   DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD:

  cout << "    Start Vaiance: preparation  *****" << endl;
  TH2F *mapdphinorm1HE1 = (TH2F *)hfile->Get("h_maprphinorm2_HE1");
  TH2F *mapdphinorm0HE1 = (TH2F *)hfile->Get("h_maprphinorm0_HE1");
  TH2F *mapdphinormHE1 = (TH2F *)mapdphinorm1HE1->Clone("mapdphinormHE1");
  mapdphinormHE1->Divide(mapdphinorm1HE1, mapdphinorm0HE1, 1, 1, "B");
  TH2F *mapdphinorm1HE2 = (TH2F *)hfile->Get("h_maprphinorm2_HE2");
  TH2F *mapdphinorm0HE2 = (TH2F *)hfile->Get("h_maprphinorm0_HE2");
  TH2F *mapdphinormHE2 = (TH2F *)mapdphinorm1HE2->Clone("mapdphinormHE2");
  mapdphinormHE2->Divide(mapdphinorm1HE2, mapdphinorm0HE2, 1, 1, "B");
  TH2F *mapdphinorm1HE3 = (TH2F *)hfile->Get("h_maprphinorm2_HE3");
  TH2F *mapdphinorm0HE3 = (TH2F *)hfile->Get("h_maprphinorm0_HE3");
  TH2F *mapdphinormHE3 = (TH2F *)mapdphinorm1HE3->Clone("mapdphinormHE3");
  mapdphinormHE3->Divide(mapdphinorm1HE3, mapdphinorm0HE3, 1, 1, "B");
  TH2F *mapdphinorm1HE4 = (TH2F *)hfile->Get("h_maprphinorm2_HE4");
  TH2F *mapdphinorm0HE4 = (TH2F *)hfile->Get("h_maprphinorm0_HE4");
  TH2F *mapdphinormHE4 = (TH2F *)mapdphinorm1HE4->Clone("mapdphinormHE4");
  mapdphinormHE4->Divide(mapdphinorm1HE4, mapdphinorm0HE4, 1, 1, "B");
  TH2F *mapdphinorm1HE5 = (TH2F *)hfile->Get("h_maprphinorm2_HE5");
  TH2F *mapdphinorm0HE5 = (TH2F *)hfile->Get("h_maprphinorm0_HE5");
  TH2F *mapdphinormHE5 = (TH2F *)mapdphinorm1HE5->Clone("mapdphinormHE5");
  mapdphinormHE5->Divide(mapdphinorm1HE5, mapdphinorm0HE5, 1, 1, "B");
  TH2F *mapdphinorm1HE6 = (TH2F *)hfile->Get("h_maprphinorm2_HE6");
  TH2F *mapdphinorm0HE6 = (TH2F *)hfile->Get("h_maprphinorm0_HE6");
  TH2F *mapdphinormHE6 = (TH2F *)mapdphinorm1HE6->Clone("mapdphinormHE6");
  mapdphinormHE6->Divide(mapdphinorm1HE6, mapdphinorm0HE6, 1, 1, "B");
  TH2F *mapdphinorm1HE7 = (TH2F *)hfile->Get("h_maprphinorm2_HE7");
  TH2F *mapdphinorm0HE7 = (TH2F *)hfile->Get("h_maprphinorm0_HE7");
  TH2F *mapdphinormHE7 = (TH2F *)mapdphinorm1HE7->Clone("mapdphinormHE7");
  mapdphinormHE7->Divide(mapdphinorm1HE7, mapdphinorm0HE7, 1, 1, "B");
  cout << "      Vaiance: preparation DONE *****" << endl;
  //====================================================================== put Vaiance=Dispersia = Sig**2=<R**2> - (<R>)**2
  //                                                                                           = sum(R*R)/N - (sum(R)/N)**2
  //                                                                                           =  mapdphinormHE1,2,3..7   -  alexhe*alexhe
  for (int jeta = 0; jeta < njeta; jeta++) {
    for (int jphi = 0; jphi < njphi; jphi++) {
      variance[0][jeta][jphi] =
          fabs(mapdphinormHE1->GetBinContent(jeta + 1, jphi + 1) - alexhe[0][jeta][jphi] * alexhe[0][jeta][jphi]);
      if (variance[0][jeta][jphi] < 0.00003)
        variance[0][jeta][jphi] = 0.000067;
      variance[1][jeta][jphi] =
          fabs(mapdphinormHE2->GetBinContent(jeta + 1, jphi + 1) - alexhe[1][jeta][jphi] * alexhe[1][jeta][jphi]);
      if (variance[1][jeta][jphi] < 0.00003)
        variance[1][jeta][jphi] = 0.000067;
      variance[2][jeta][jphi] =
          fabs(mapdphinormHE3->GetBinContent(jeta + 1, jphi + 1) - alexhe[2][jeta][jphi] * alexhe[2][jeta][jphi]);
      if (variance[2][jeta][jphi] < 0.00003)
        variance[2][jeta][jphi] = 0.000067;
      variance[3][jeta][jphi] =
          fabs(mapdphinormHE4->GetBinContent(jeta + 1, jphi + 1) - alexhe[3][jeta][jphi] * alexhe[3][jeta][jphi]);
      if (variance[3][jeta][jphi] < 0.00003)
        variance[3][jeta][jphi] = 0.000067;
      variance[4][jeta][jphi] =
          fabs(mapdphinormHE5->GetBinContent(jeta + 1, jphi + 1) - alexhe[4][jeta][jphi] * alexhe[4][jeta][jphi]);
      if (variance[4][jeta][jphi] < 0.00003)
        variance[4][jeta][jphi] = 0.000067;
      variance[5][jeta][jphi] =
          fabs(mapdphinormHE6->GetBinContent(jeta + 1, jphi + 1) - alexhe[5][jeta][jphi] * alexhe[5][jeta][jphi]);
      if (variance[5][jeta][jphi] < 0.00003)
        variance[5][jeta][jphi] = 0.000067;
      variance[6][jeta][jphi] =
          fabs(mapdphinormHE7->GetBinContent(jeta + 1, jphi + 1) - alexhe[6][jeta][jphi] * alexhe[6][jeta][jphi]);
      if (variance[6][jeta][jphi] < 0.00003)
        variance[6][jeta][jphi] = 0.000067;
    }
  }
  cout << "      Vaiance: DONE*****" << endl;

  //------------------------  2D-eta/phi-plot: D, averaged over depthes
  //======================================================================
  //======================================================================
  cout << "      R2D-eta/phi-plot: D, averaged over depthes *****" << endl;
  c1x1->Clear();
  /////////////////
  c1x0->Divide(1, 1);
  c1x0->cd(1);
  TH2F *Defz42D = new TH2F("Defz42D", "", neta, -41., 41., nphi, 0., 72.);
  TH2F *Defz42D0 = new TH2F("Defz42D0", "", neta, -41., 41., nphi, 0., 72.);
  TH2F *Defz42DF = (TH2F *)Defz42D0->Clone("Defz42DF");
  for (int i = 0; i < ndepth; i++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int jphi = 0; jphi < nphi; jphi++) {
        double ccc1 = variance[i][jeta][jphi];
        int k2plot = jeta - 41;
        int kkk = k2plot;  //if(k2plot >0   kkk=k2plot+1; //-41 +41 !=0
        if (alexhe[i][jeta][jphi] > 0.) {
          Defz42D->Fill(kkk, jphi, ccc1);
          Defz42D0->Fill(kkk, jphi, 1.);
        }
      }
    }
  }
  Defz42DF->Divide(Defz42D, Defz42D0, 1, 1, "B");  // average A
  //    Defz1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Defz42DF->SetMarkerStyle(20);
  Defz42DF->SetMarkerSize(0.4);
  Defz42DF->GetZaxis()->SetLabelSize(0.08);
  Defz42DF->SetXTitle("<D>_depth       #eta  \b");
  Defz42DF->SetYTitle("      #phi \b");
  Defz42DF->SetZTitle("<D>_depth \b");
  Defz42DF->SetMarkerColor(2);
  Defz42DF->SetLineColor(0);  //      Defz42DF->SetMaximum(1.000);  //      Defz42DF->SetMinimum(1.0);
  Defz42DF->Draw("COLZ");
  /////////////////
  c1x0->Update();
  c1x0->Print("Dhist2IterationMethodHE.png");
  c1x0->Clear();
  // clean-up
  if (Defz42D)
    delete Defz42D;
  if (Defz42D0)
    delete Defz42D0;
  if (Defz42DF)
    delete Defz42DF;

  //====================================================================== 1D plot: D vs phi , averaged over depthes & eta
  //======================================================================
  cout << "      1D plot: D vs phi , averaged over depthes & eta *****" << endl;
  c1x1->Clear();
  /////////////////
  c1x1->Divide(1, 1);
  c1x1->cd(1);
  TH1F *Defz41D = new TH1F("Defz41D", "", nphi, 0., 72.);
  TH1F *Defz41D0 = new TH1F("Defz41D0", "", nphi, 0., 72.);
  TH1F *Defz41DF = (TH1F *)Defz41D0->Clone("Defz41DF");

  for (int jphi = 0; jphi < nphi; jphi++) {
    for (int jeta = 0; jeta < neta; jeta++) {
      for (int i = 0; i < ndepth; i++) {
        double ccc1 = variance[i][jeta][jphi];
        if (alexhe[i][jeta][jphi] > 0.) {
          Defz41D->Fill(jphi, ccc1);
          Defz41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     Defz41D->Sumw2();Defz41D0->Sumw2();

  Defz41DF->Divide(Defz41D, Defz41D0, 1, 1, "B");  // R averaged over depthes & eta
  Defz41D0->Sumw2();
  //    for (int jphi=1;jphi<73;jphi++) {Defz41DF->SetBinError(jphi,0.01);}
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Defz41DF->SetMarkerStyle(20);
  Defz41DF->SetMarkerSize(1.4);
  Defz41DF->GetZaxis()->SetLabelSize(0.08);
  Defz41DF->SetXTitle("#phi  \b");
  Defz41DF->SetYTitle("  <D> \b");
  Defz41DF->SetZTitle("<D>_PHI  - AllDepthes \b");
  Defz41DF->SetMarkerColor(4);
  Defz41DF->SetLineColor(4);
  Defz41DF->SetMinimum(0.8);
  Defz41DF->SetMinimum(-0.015);
  Defz41DF->Draw("Error");
  /////////////////
  c1x1->Update();
  c1x1->Print("Dhist1IterationMethodHE.png");
  c1x1->Clear();
  // clean-up
  if (Defz41D)
    delete Defz41D;
  if (Defz41D0)
    delete Defz41D0;
  if (Defz41DF)
    delete Defz41DF;

  //========================================================================================== 14
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=1
  cout << "      1D plot: D vs phi , different eta,  depth=1 *****" << endl;
  c3x5->Clear();
  /////////////////
  c3x5->Divide(3, 5);
  c3x5->cd(1);

  int kcountHEpositivedirectionD1 = 1;
  TH1F *h2CeffHEpositivedirectionD1 = new TH1F("h2CeffHEpositivedirectionD1", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=1
      for (int i = 0; i < 1; i++) {
        TH1F *HEpositivedirectionD1 = (TH1F *)h2CeffHEpositivedirectionD1->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD1->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD1->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1414       kcountHEpositivedirectionD1   =     " << kcountHEpositivedirectionD1
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD1);
          HEpositivedirectionD1->SetMarkerStyle(20);
          HEpositivedirectionD1->SetMarkerSize(0.4);
          HEpositivedirectionD1->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD1->SetXTitle("HEpositivedirectionD1 \b");
          HEpositivedirectionD1->SetMarkerColor(2);
          HEpositivedirectionD1->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD1 == 1)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 18; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 2)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 19; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 3)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 20; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 4)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 21; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 5)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 22; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 6)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 23; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 7)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 24; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 8)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 25; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 9)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 26; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 10)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 27; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 11)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 28; depth = 1 \b");
          if (kcountHEpositivedirectionD1 == 12)
            HEpositivedirectionD1->SetXTitle("D for HE+ jeta = 29; depth = 1 \b");
          HEpositivedirectionD1->Draw("Error");
          kcountHEpositivedirectionD1++;
          if (kcountHEpositivedirectionD1 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth1HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD1)
    delete h2CeffHEpositivedirectionD1;

  //========================================================================================== 15
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=2
  cout << "      1D plot: D vs phi , different eta,  depth=2 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD2 = 1;
  TH1F *h2CeffHEpositivedirectionD2 = new TH1F("h2CeffHEpositivedirectionD2", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=2
      for (int i = 1; i < 2; i++) {
        TH1F *HEpositivedirectionD2 = (TH1F *)h2CeffHEpositivedirectionD2->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD2->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD2->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1515       kcountHEpositivedirectionD2   =     " << kcountHEpositivedirectionD2
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD2);
          HEpositivedirectionD2->SetMarkerStyle(20);
          HEpositivedirectionD2->SetMarkerSize(0.4);
          HEpositivedirectionD2->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD2->SetXTitle("HEpositivedirectionD2 \b");
          HEpositivedirectionD2->SetMarkerColor(2);
          HEpositivedirectionD2->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD2 == 1)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 17; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 2)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 18; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 3)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 19; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 4)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 20; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 5)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 21; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 6)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 22; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 7)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 23; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 8)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 24; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 9)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 25; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 10)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 26; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 11)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 27; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 12)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 28; depth = 2 \b");
          if (kcountHEpositivedirectionD2 == 13)
            HEpositivedirectionD2->SetXTitle("D for HE+ jeta = 29; depth = 2 \b");
          HEpositivedirectionD2->Draw("Error");
          kcountHEpositivedirectionD2++;
          if (kcountHEpositivedirectionD2 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth2HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD2)
    delete h2CeffHEpositivedirectionD2;

  //========================================================================================== 16
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=3
  cout << "      1D plot: D vs phi , different eta,  depth=3 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD3 = 1;
  TH1F *h2CeffHEpositivedirectionD3 = new TH1F("h2CeffHEpositivedirectionD3", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=3
      for (int i = 2; i < 3; i++) {
        TH1F *HEpositivedirectionD3 = (TH1F *)h2CeffHEpositivedirectionD3->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD3->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD3->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1616       kcountHEpositivedirectionD3   =     " << kcountHEpositivedirectionD3
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD3);
          HEpositivedirectionD3->SetMarkerStyle(20);
          HEpositivedirectionD3->SetMarkerSize(0.4);
          HEpositivedirectionD3->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD3->SetXTitle("HEpositivedirectionD3 \b");
          HEpositivedirectionD3->SetMarkerColor(2);
          HEpositivedirectionD3->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD3 == 1)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 17; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 2)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 18; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 3)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 19; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 4)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 20; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 5)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 21; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 6)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 22; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 7)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 23; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 8)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 24; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 9)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 25; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 10)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 26; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 11)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 27; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 12)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 28; depth = 3 \b");
          if (kcountHEpositivedirectionD3 == 13)
            HEpositivedirectionD3->SetXTitle("D for HE+ jeta = 29; depth = 3 \b");
          HEpositivedirectionD3->Draw("Error");
          kcountHEpositivedirectionD3++;
          if (kcountHEpositivedirectionD3 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth3HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD3)
    delete h2CeffHEpositivedirectionD3;

  //========================================================================================== 17
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=4
  cout << "      1D plot: D vs phi , different eta,  depth=4 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD4 = 1;
  TH1F *h2CeffHEpositivedirectionD4 = new TH1F("h2CeffHEpositivedirectionD4", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=4
      for (int i = 3; i < 4; i++) {
        TH1F *HEpositivedirectionD4 = (TH1F *)h2CeffHEpositivedirectionD4->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD4->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD4->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1717       kcountHEpositivedirectionD4   =     " << kcountHEpositivedirectionD4
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD4);
          HEpositivedirectionD4->SetMarkerStyle(20);
          HEpositivedirectionD4->SetMarkerSize(0.4);
          HEpositivedirectionD4->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD4->SetXTitle("HEpositivedirectionD4 \b");
          HEpositivedirectionD4->SetMarkerColor(2);
          HEpositivedirectionD4->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD4 == 1)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 16; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 2)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 18; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 3)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 19; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 4)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 20; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 5)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 21; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 6)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 22; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 7)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 23; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 8)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 24; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 9)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 25; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 10)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 26; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 11)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 27; depth = 4 \b");
          if (kcountHEpositivedirectionD4 == 12)
            HEpositivedirectionD4->SetXTitle("D for HE+ jeta = 28; depth = 4 \b");
          HEpositivedirectionD4->Draw("Error");
          kcountHEpositivedirectionD4++;
          if (kcountHEpositivedirectionD4 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth4HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD4)
    delete h2CeffHEpositivedirectionD4;

  //========================================================================================== 18
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=5
  cout << "      1D plot: D vs phi , different eta,  depth=5 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD5 = 1;
  TH1F *h2CeffHEpositivedirectionD5 = new TH1F("h2CeffHEpositivedirectionD5", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=5
      for (int i = 4; i < 5; i++) {
        TH1F *HEpositivedirectionD5 = (TH1F *)h2CeffHEpositivedirectionD5->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD5->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD5->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1818       kcountHEpositivedirectionD5   =     " << kcountHEpositivedirectionD5
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD5);
          HEpositivedirectionD5->SetMarkerStyle(20);
          HEpositivedirectionD5->SetMarkerSize(0.4);
          HEpositivedirectionD5->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD5->SetXTitle("HEpositivedirectionD5 \b");
          HEpositivedirectionD5->SetMarkerColor(2);
          HEpositivedirectionD5->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD5 == 1)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 18; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 2)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 19; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 3)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 20; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 4)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 21; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 5)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 22; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 6)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 23; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 7)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 24; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 8)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 25; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 9)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 26; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 10)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 27; depth = 5 \b");
          if (kcountHEpositivedirectionD5 == 11)
            HEpositivedirectionD5->SetXTitle("D for HE+ jeta = 28; depth = 5 \b");
          HEpositivedirectionD5->Draw("Error");
          kcountHEpositivedirectionD5++;
          if (kcountHEpositivedirectionD5 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth5HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD5)
    delete h2CeffHEpositivedirectionD5;

  //========================================================================================== 19
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=6
  cout << "      1D plot: D vs phi , different eta,  depth=6 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD6 = 1;
  TH1F *h2CeffHEpositivedirectionD6 = new TH1F("h2CeffHEpositivedirectionD6", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=6
      for (int i = 5; i < 6; i++) {
        TH1F *HEpositivedirectionD6 = (TH1F *)h2CeffHEpositivedirectionD6->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD6->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD6->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest > 0.) {
          cout << "1919       kcountHEpositivedirectionD6   =     " << kcountHEpositivedirectionD6
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD6);
          HEpositivedirectionD6->SetMarkerStyle(20);
          HEpositivedirectionD6->SetMarkerSize(0.4);
          HEpositivedirectionD6->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD6->SetXTitle("HEpositivedirectionD6 \b");
          HEpositivedirectionD6->SetMarkerColor(2);
          HEpositivedirectionD6->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD6 == 1)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 19; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 2)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 20; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 3)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 21; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 4)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 22; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 5)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 23; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 6)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 24; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 7)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 25; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 8)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 26; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 9)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 27; depth = 6 \b");
          if (kcountHEpositivedirectionD6 == 10)
            HEpositivedirectionD6->SetXTitle("D for HE+ jeta = 28; depth = 6 \b");
          HEpositivedirectionD6->Draw("Error");
          kcountHEpositivedirectionD6++;
          if (kcountHEpositivedirectionD6 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth6HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD6)
    delete h2CeffHEpositivedirectionD6;

  //========================================================================================== 20
  //======================================================================
  //======================================================================1D plot: D vs phi , different eta,  depth=7
  cout << "      1D plot: D vs phi , different eta,  depth=7 *****" << endl;
  c3x5->Clear();
  c3x5->Divide(3, 5);
  c3x5->cd(1);
  int kcountHEpositivedirectionD7 = 1;
  TH1F *h2CeffHEpositivedirectionD7 = new TH1F("h2CeffHEpositivedirectionD7", "", nphi, 0., 72.);

  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirectionD:
    if (jeta - 41 > 0) {
      //	     for (int i=0;i<ndepth;i++) {
      // depth=7
      for (int i = 6; i < 7; i++) {
        TH1F *HEpositivedirectionD7 = (TH1F *)h2CeffHEpositivedirectionD7->Clone("twod1");

        float ccctest = 0;  // to avoid empty massive elements
        for (int jphi = 0; jphi < nphi; jphi++) {
          double ccc1 = variance[i][jeta][jphi];
          if (alexhe[i][jeta][jphi] > 0.) {
            HEpositivedirectionD7->Fill(jphi, ccc1);
            ccctest = 1.;  //HEpositivedirectionD7->SetBinError(i,0.01);
          }
        }  // for jphi
        if (ccctest != 0.) {
          cout << "2020       kcountHEpositivedirectionD7   =     " << kcountHEpositivedirectionD7
               << "   jeta-41=     " << jeta - 41 << endl;
          c3x5->cd(kcountHEpositivedirectionD7);
          HEpositivedirectionD7->SetMarkerStyle(20);
          HEpositivedirectionD7->SetMarkerSize(0.4);
          HEpositivedirectionD7->GetYaxis()->SetLabelSize(0.04);
          HEpositivedirectionD7->SetXTitle("HEpositivedirectionD7 \b");
          HEpositivedirectionD7->SetMarkerColor(2);
          HEpositivedirectionD7->SetLineColor(0);
          gPad->SetGridy();
          gPad->SetGridx();
          //	   gPad->SetLogy();
          if (kcountHEpositivedirectionD7 == 1)
            HEpositivedirectionD7->SetXTitle("D for HE+ jeta = 26; depth = 7 \b");
          if (kcountHEpositivedirectionD7 == 2)
            HEpositivedirectionD7->SetXTitle("D for HE+ jeta = 27; depth = 7 \b");
          if (kcountHEpositivedirectionD7 == 3)
            HEpositivedirectionD7->SetXTitle("D for HE+ jeta = 28; depth = 7 \b");
          HEpositivedirectionD7->Draw("Error");
          kcountHEpositivedirectionD7++;
          if (kcountHEpositivedirectionD7 > 15)
            break;  // 4x6 = 24
        }           //ccctest>0

      }  // for i
    }    //if(jeta-41 > 0)
  }      //for jeta
  /////////////////
  c3x5->Update();
  c3x5->Print("Dhist1IterationMethodDepth7HE.png");
  c3x5->Clear();
  // clean-up
  if (h2CeffHEpositivedirectionD7)
    delete h2CeffHEpositivedirectionD7;

  //=====================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================

  std::cout << " We are here to print 2017 MAPs " << std::endl;
  //======================================================================

  //======================================================================
  /// Prepare maps of good/bad channels:

  TH2F *Map_ALL = new TH2F("Map_All", "Map_all", 82, -41, 40, 72, 0, 71);
  int nx = Map_ALL->GetXaxis()->GetNbins();
  int ny = Map_ALL->GetYaxis()->GetNbins();
  int NBad = 0;
  int NWarn = 0;
  int NCalib = 0;
  int NPed = 0;
  //    int Eta[3][10000]={0};
  int Eta[4][10000] = {0};
  int Phi[4][10000] = {0};
  int Sub[4][10000] = {0};
  int Depth[4][10000] = {0};
  string Comment[4][10000] = {""};
  string Text[33] = {"", "Cm", "Am", "Wm", "Rm", "TNm", "TXm", "", "", "", "", "Cc", "Ac", "Wc", "Rc", "TNc", "TXc",
                     "", "",   "",   "",   "GS", "",    "",    "", "", "", "", "",   "",   "",   "Pm", "pWm"};
  int flag_W = 0;
  int flag_B = 0;
  int flag_P = 0;

  for (int i = 1; i <= nx; i++) {           //Eta
    for (int j = 1; j <= ny; j++) {         // Phi
      for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HO, 4-HF
                                            //	     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
        //	     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
        //	     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
        for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
          if (Map_SUB[sub][k]->GetBinContent(i, j) != 0) {
            Map_SUB[sub][k]->SetBinContent(i, j, 0.5);
            Map_ALL->SetBinContent(i, j, 0.5);
          }
        }
      }
    }
  }

  for (int i = 1; i <= nx; i++) {           //Eta
    for (int j = 1; j <= ny; j++) {         // Phi
      for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
                                            //	     int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
        //	     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
        //	     int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
        for (int k = k_min[sub]; k <= k_max[sub]; k++) {  //Depth
          flag_W = 0;
          flag_B = 0;
          flag_P = 0;
          for (int test = 1; test <= 6; test++) {  //Test: 1-Wm, 2-Rm, etc
                                                   //Bad
            if (Map_Ampl[test][sub][k]->GetBinContent(i, j) >
                0.1) {  //Rate 0.1 for displaying  on whole detector map and subdetector map
              Map_ALL->SetBinContent(i, j, 1.);
              Map_SUB[sub][k]->SetBinContent(i, j, 1.);
              if (flag_B == 0) {
                NBad += 1;
                Eta[2][NBad] = i - 41;
                Phi[2][NBad] = j - 1;
                Sub[2][NBad] = sub;
                Depth[2][NBad] = k;
                Comment[2][NBad] = Text[test];
              } else
                Comment[2][NBad] += ", " + Text[test];
              flag_B = 1;
              //		      cout<<"Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)<<endl;
            }
            //Warning
            /* 
                   if ((Map_Ampl[test][sub][k]->GetBinContent(i,j) != 0.)&&(Map_Ampl[test][sub][k]->GetBinContent(i,j) < 0.001) )  {
	              if (Map_SUB[sub][k]->GetBinContent(i,j)!=1.) Map_SUB[sub][k]->SetBinContent(i,j,0.75);
		      if (Map_ALL->GetBinContent(i,j)!=1.) Map_ALL->SetBinContent(i,j,0.75);
		      if (flag_W == 0) {
		         NWarn +=1; 
		         Eta[1][NWarn]=i-41;
		         Phi[1][NWarn]=j-1;
			 Sub[1][NWarn]=sub;
		         Depth[1][NWarn]=k;
		         Comment[1][NWarn]=Text[test]; 
		      } 
		      else Comment[1][NWarn]+=", "+Text[test];
		      flag_W = 1;		      		 
//		      cout<<"Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)<<endl;
                   }		   
*/

          }  //end test

          //Calib
          /*
		 for (int test=11;test<=16;test++) { //Test: 1-2.E, 2-2.F, etc
                   if (Map_Ampl[test][sub][k]->GetBinContent(i,j) != 0.)  {
//	              if (Map_SUB[sub][k]->GetBinContent(i,j)!=1.0) Map_SUB[sub][k]->SetBinContent(i,j,0.3);
//		      if (Map_ALL->GetBinContent(i,j)!=1.) Map_ALL->SetBinContent(i,j,0.3);
		      if (flag_W == 0) {
		         NWarn +=1; 
		         Eta[1][NWarn]=i-41;
		         Phi[1][NWarn]=j-1;
			 Sub[1][NWarn]=sub;
		         Depth[1][NWarn]=k;
		         Comment[1][NWarn]=Text[test]; 
		      } 
		      else Comment[1][NWarn]+=", "+Text[test];
		      flag_W = 1;		      		 
//		      cout<<"Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)<<endl;
                  } 
		} //end test
*/
          //Gain stabil
          for (int test = 21; test <= 21; test++) {
            if (abs(Map_Ampl[test][sub][k]->GetBinContent(i, j)) > porog[sub]) {
              if (Map_SUB[sub][k]->GetBinContent(i, j) != 1.0)
                Map_SUB[sub][k]->SetBinContent(i, j, 0.75);
              if (Map_ALL->GetBinContent(i, j) != 1.) {
                Map_ALL->SetBinContent(i, j, 0.75);
                if (flag_W == 0) {
                  NWarn += 1;
                  Eta[1][NWarn] = i - 41;
                  Phi[1][NWarn] = j - 1;
                  Sub[1][NWarn] = sub;
                  Depth[1][NWarn] = k;
                  Comment[1][NWarn] = Text[test];
                } else
                  Comment[1][NWarn] += ", " + Text[test];
                flag_W = 1;

                cout << "Map_Ampl[" << test << "][" << sub << "][" << k << "]->GetBinContent(" << i << "," << j
                     << ")= " << Map_Ampl[test][sub][k]->GetBinContent(i, j) << endl;
              }
            }
          }  //end test

          //Pedestals
          for (int test = 31; test <= 32; test++) {
            if (Map_Ampl[test][sub][k]->GetBinContent(i, j) > 0.1) {
              //	              if (Map_SUB[sub][k]->GetBinContent(i,j)!=1.0) Map_SUB[sub][k]->SetBinContent(i,j,0.75);
              //		      if (Map_ALL->GetBinContent(i,j)!=1.)  Map_ALL->SetBinContent(i,j,0.75);
              if (flag_P == 0) {
                NPed += 1;
                Eta[3][NPed] = i - 41;
                Phi[3][NPed] = j - 1;
                Sub[3][NPed] = sub;
                Depth[3][NPed] = k;
                Comment[3][NPed] = Text[test];
              } else
                Comment[3][NPed] += ", " + Text[test];
              flag_P = 1;

              //		      cout<<"Map_Ampl["<<test<<"]["<<sub<<"]["<<k<<"]->GetBinContent("<<i<<","<<j<<")= "<<Map_Ampl[test][sub][k]->GetBinContent(i,j)<<endl;
            }
          }  //end test

        }                               //end Depth
      }                                 //end Sub
    }                                   //end Phi
  }                                     //end Eta
                                        // subdet maps
  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO

    std::cout << " 2017 MAPS_SUB " << sub << std::endl;
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
    //       int k_min[5]={0,1,1,4,1}; // minimum depth for each subdet
    //     int k_max[5]={0,2,3,4,2}; // maximum depth for each subdet
    //       int k_max[5]={0,2,7,4,4}; // maximum depth for each subdet
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
      //          gPad->SetLogz();
      //          gStyle->SetTitleOffset(0.5, "Y");
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
      Map_SUB[sub][k]->Draw("COL");
      Map_SUB[sub][k]->GetYaxis()->SetRangeUser(0, 72.);
      Map_SUB[sub][k]->GetZaxis()->SetRangeUser(0., 1.);
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
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //  TCanvas *cmain1 = new TCanvas("cmain1","MAP",200,10,1400,1800);
  TCanvas *cmain1 = new TCanvas("cmain1", "MAP", 250, 10, 1450, 1410);
  cmain1->Divide(2, 2);

  cmain1->cd(1);
  TH1F *JDBEYESJ0 = (TH1F *)hfile->Get("h_totalAmplitudeHBperEvent");
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

  cmain1->cd(2);
  TH1F *JDBEYESJ1 = (TH1F *)hfile->Get("h_totalAmplitudeHEperEvent");
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

  cmain1->cd(3);
  TH1F *JDBEYESJ2 = (TH1F *)hfile->Get("h_totalAmplitudeHFperEvent");
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

  cmain1->cd(4);
  TH1F *JDBEYESJ3 = (TH1F *)hfile->Get("h_totalAmplitudeHOperEvent");
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

  cmain1->Modified();
  cmain1->Update();
  cmain1->Print("EVENTDEPENDENCE.png");

  std::cout << " EVENTDEPENDENCE " << std::endl;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // ALL SubDet
  gStyle->SetOptTitle(0);
  TCanvas *cmain = new TCanvas("cmain", "MAP", 1000, 1000);
  //cmain->cd(1);
  gPad->SetGridy();
  gPad->SetGridx();
  //   gPad->SetLogz();
  Map_ALL->SetTitleOffset(1.3, "Y");
  Map_ALL->SetXTitle("#eta \b");
  Map_ALL->SetYTitle("#phi \b");
  Map_ALL->Draw("COL");
  Map_ALL->GetYaxis()->SetRangeUser(0, 72.);
  Map_ALL->GetZaxis()->SetRangeUser(0, 1.);
  cmain->Modified();
  cmain->Update();
  cmain->Print("MAP.png");

  std::cout << "********      MAP_ALL  done" << std::endl;
  std::cout << "********" << std::endl;

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating each test kind for each subdet html pages:" << std::endl;

  //======================================================================

  //======================================================================
  // Creating each test kind for each subdet html pages:
  std::string raw_class, raw_class1, raw_class2, raw_class3;
  int ind = 0;

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
    ofstream htmlFileT, htmlFileC, htmlFileD, htmlFileP, htmlFileS, htmlFileM;
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
      htmlFileM.open("HE_IterationMethod.html");
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
    htmlFileT << "<title> Remote Monitoring Tool </title>" << std::endl;
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
    htmlFileT << "<h4> Channel legend: color is rate of entries </h4>" << std::endl;
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

    htmlFileT << "</body> " << std::endl;
    htmlFileT << "</html> " << std::endl;
    htmlFileT.close();

    //Calibration channels
    htmlFileC << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileC << "<head>" << std::endl;
    htmlFileC << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileC << "<title> Raw Data Analyser </title>" << std::endl;
    htmlFileC << "<style type=\"text/css\">" << std::endl;
    htmlFileC << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileC << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileC << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileC << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileC << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileC << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileC << "</style>" << std::endl;
    htmlFileC << "<body>" << std::endl;

    if (sub == 1)
      htmlFileC << "<h1> Criteria for calibration channels for HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileC << "<h1> Criteria for calibration channels for HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileC << "<h1> Criteria for calibration channels for HO, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileC << "<h1> Criteria for calibration channels for HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // Test Entries

    htmlFileC << "<h2> 0. Entries for each channel.</h3>" << std::endl;
    htmlFileC << "<h3> 0.A. Entries in each channel for each depth.</h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: color is rate of entries </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateCalibEntryHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateCalibEntryHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateCalibEntryHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateCalibEntryHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // Test Cc
    htmlFileC << "<h2> 1. Cc criterion: CapID errors for each channel.</h3>" << std::endl;
    htmlFileC << "<h3> 1.A. Rate of CapId failures in each channel for each depth.</h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateCapCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateCapCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateCapCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateCapCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // Ac
    htmlFileC << "<h2> 2. Ac criterion: ADC amplitude collected over all TSs(Full Amplitude) for each channel. </h3>"
              << std::endl;
    htmlFileC << "<h3> 2.A. Full ADC amplitude distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileC << "<h4> Legend: Bins less " << MIN_C[2][sub] << " correpond to bad ADC amplitude </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"HistAmplCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"HistAmplCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"HistAmplCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"HistAmplCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;
    htmlFileC << "<h3> 2.B. Rate of bad ADC amplitude (<" << MIN_C[2][sub] << ") in each channel for each depth. </h3>"
              << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colours - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateAmplCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateAmplCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateAmplCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateAmplCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // Test Wc
    htmlFileC << "<h2> 3. Wc criterion: RMS (width) of ADC amplutude for each channel.</h3>" << std::endl;
    htmlFileC << "<h3> 3.A. W distribution over all events, channel and depth.</h3>" << std::endl;
    htmlFileC << "<h4> Legend: Bins less " << MIN_C[3][sub] << " and more " << MAX_C[3][sub]
              << " correpond to bad RMS </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"HistRMSCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"HistRMSCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"HistRMSCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"HistRMSCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;
    htmlFileC << "<h3> 3.B. Rate of bad W (<" << MIN_C[3][sub] << ",>" << MAX_C[3][sub]
              << ") in each channel for each depth.</h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateRMSCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateRMSCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateRMSCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateRMSCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // Rc
    htmlFileC << "<h2> 4. Rc criterion: Ratio ADC value sum over five near maximum (-2, -1, max, +1, +2) TS to ADC "
                 "value sum over all TS for each channel. </h3>"
              << std::endl;
    htmlFileC << "<h3> 4.A. Ratio distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileC << "<h4> Legend: Bins less " << MIN_C[4][sub] << " and more " << MAX_C[4][sub]
              << " correpond to bad ratio </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"Hist43TStoAllTSCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"Hist43TStoAllTSCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"Hist43TStoAllTSCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"Hist43TStoAllTSCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;
    htmlFileC << "<h3> 4.B. Rate of bad Ratio (<" << MIN_C[4][sub] << ", >" << MAX_C[4][sub]
              << ") in each channel for each depth.</h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRate43TStoAllTSCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRate43TStoAllTSCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRate43TStoAllTSCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRate43TStoAllTSCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // TNc
    htmlFileC << "<h2> 5. TNc criterion: Mean TS position for each channel.</h3>" << std::endl;
    htmlFileC << "<h3> 5.A. TN position distribution over all events, channels and depths.</h3>" << std::endl;
    htmlFileC << "<h4> Legend: Bins less " << MIN_C[5][sub] << " and more " << MAX_C[5][sub]
              << " correpond to bad position </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"HistMeanPosCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"HistMeanPosCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"HistMeanPosCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"HistMeanPosCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;
    htmlFileC << "<h3> 5.B. Rate of bad TN position  (<" << MIN_C[5][sub] << ", >" << MAX_C[5][sub]
              << ") in each channel for each depth. </h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateMeanPosCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateMeanPosCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateMeanPosCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateMeanPosCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    // TXm
    htmlFileC << "<h2> 6.TXc criterion: Maximum TS position for each channel.</h3>" << std::endl;
    htmlFileC << "<h3> 6.A. TX position distribution over all events, channel and depth.</h3>" << std::endl;
    htmlFileC << "<h4> Legend: Bins less " << MIN_C[6][sub] << " and more " << MAX_C[6][sub]
              << " correpond to bad position </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"HistMaxPosCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"HistMaxPosCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"HistMaxPosCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"HistMaxPosCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;
    htmlFileC << "<h3> 6.B. Rate of bad TX position  (<" << MIN_C[6][sub] << ", >" << MAX_C[6][sub]
              << ") in each channel for each depth. </h3>" << std::endl;
    htmlFileC << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileC << " <img src=\"MapRateMaxPosCalibHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileC << " <img src=\"MapRateMaxPosCalibHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileC << " <img src=\"MapRateMaxPosCalibHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileC << " <img src=\"MapRateMaxPosCalibHF.png\" />" << std::endl;
    htmlFileC << "<br>" << std::endl;

    htmlFileC << "</body> " << std::endl;
    htmlFileC << "</html> " << std::endl;
    htmlFileC.close();

    //Response drift
    htmlFileD << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileD << "<head>" << std::endl;
    htmlFileD << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileD << "<title> Remore Monitoring Tool </title>" << std::endl;
    htmlFileD << "<style type=\"text/css\">" << std::endl;
    htmlFileD << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileD << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileD << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileD << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileD << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileD << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileD << "</style>" << std::endl;
    htmlFileD << "<body>" << std::endl;

    if (sub == 1)
      htmlFileD << "<h1> Response drift for HB: Current RUN = " << runnumber << ", Reference RUN = " << refrunnumber
                << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileD << "<h1> Response drift for HE: Current RUN = " << runnumber << ", Reference RUN = " << refrunnumber
                << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileD << "<h1> Response drift for HO: Current RUN = " << runnumber << ", Reference RUN = " << refrunnumber
                << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileD << "<h1> Response drift for HF: Current RUN = " << runnumber << ", Reference RUN = " << refrunnumber
                << " </h1>" << std::endl;
    htmlFileD << "<br>" << std::endl;

    // test GS
    htmlFileD << "<h2> 1. Gain Stability (GS) </h3>" << std::endl;
    htmlFileD << "<h3> 1.A. Averaged channel response, collected over all TS, for Current run in each channel for each "
                 "depth.</h3>"
              << std::endl;
    htmlFileD << "<h4> Channel legend: colour means cooresponding value of mean response. </h4>" << std::endl;
    if (sub == 1)
      htmlFileD << " <img src=\"MapRateAmpl1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileD << " <img src=\"MapRateAmpl1HE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileD << " <img src=\"MapRateAmpl1HO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileD << " <img src=\"MapRateAmpl1HF.png\" />" << std::endl;
    htmlFileD << "<br>" << std::endl;
    htmlFileD << "<h3> 1.B. Averaged channel response, collected over all TS, for Reference run in each channel for "
                 "each depth.</h3>"
              << std::endl;
    htmlFileD << "<h4> Channel legend: colour means cooresponding value of mean response. </h4>" << std::endl;
    if (sub == 1)
      htmlFileD << " <img src=\"MapRateAmpl2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileD << " <img src=\"MapRateAmpl2HE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileD << " <img src=\"MapRateAmpl2HO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileD << " <img src=\"MapRateAmpl2HF.png\" />" << std::endl;
    htmlFileD << "<br>" << std::endl;
    htmlFileD << "<h3> 1.C. Relative difference between Current and Reference run distribution over all events, "
                 "channels for each depth.</h3>"
              << std::endl;
    htmlFileD << "<h4>  Legend: Bins less -" << porog[sub] << "% and more +" << porog[sub]
              << "% correpond to bad relative difference position </h4>" << std::endl;
    if (sub == 1)
      htmlFileD << " <img src=\"HistAmplDriftDepthHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileD << " <img src=\"HistAmplDriftDepthHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileD << " <img src=\"HistAmplDriftDepthHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileD << " <img src=\"HistAmplDriftDepthHF.png\" />" << std::endl;
    htmlFileD << "<br>" << std::endl;
    htmlFileD << "<h3> 1.D. Rate of bad relative difference  (<-" << porog[sub] << ", >+" << porog[sub]
              << ") in each channel for each depth.</h3>" << std::endl;
    htmlFileD << "<h4> Channel legend: white - good, other colour - bad. </h4>" << std::endl;
    if (sub == 1)
      htmlFileD << " <img src=\"MapRateAmplDriftHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileD << " <img src=\"MapRateAmplDriftHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileD << " <img src=\"MapRateAmplDriftHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileD << " <img src=\"MapRateAmplDriftHF.png\" />" << std::endl;
    htmlFileD << "<br>" << std::endl;

    htmlFileD << "</body> " << std::endl;
    htmlFileD << "</html> " << std::endl;
    htmlFileD.close();

    // Pedestals
    htmlFileP << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileP << "<head>" << std::endl;
    htmlFileP << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileP << "<title> Remote Monitoring Tool </title>" << std::endl;
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
    htmlFileP.close();

    // TSs Shapes:

    htmlFileS << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileS << "<head>" << std::endl;
    htmlFileS << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileS << "<title> Remote Monitoring Tool </title>" << std::endl;
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
    htmlFileS << "<br>" << std::endl;

    htmlFileS << "<h2> 1.Mean ADC Shape.</h3>" << std::endl;
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
                 "criteria: CapId, A, W, P, Pw </h3>"
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
    /*
   htmlFileS << "<h2> 2. ADC in Time Slice </h3>"<< std::endl;
     htmlFileS << "<h3> 2.A. ADC counts  histogrammed over all channels, depth and events.</h3>"<< std::endl; 
//     htmlFileS << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl; 
     if (sub==1) htmlFileS << " <img src=\"Hist_ADC_HB_All.png\" />" << std::endl;
     if (sub==2) htmlFileS << " <img src=\"Hist_ADC_HE_All.png\" />" << std::endl;    
     if (sub==3) htmlFileS << " <img src=\"Hist_ADC_HO_All.png\" />" << std::endl;
     if (sub==4) htmlFileS << " <img src=\"Hist_ADC_HF_All.png\" />" << std::endl;
     htmlFileS << "<br>"<< std::endl;
     htmlFileS << "<h3> 2.B. ADC counts  histogrammed over all channels and events for each depth separately. </h3>"<< std::endl; 
//     htmlFileS << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
     if (sub==1) htmlFileS << " <img src=\"Hist_ADC_HB_DS.png\" />" << std::endl;
     if (sub==2) htmlFileS << " <img src=\"Hist_ADC_HE_DS.png\" />" << std::endl;
     if (sub==3) htmlFileS << " <img src=\"Hist_ADC_HO_DS.png\" />" << std::endl;
     if (sub==4) htmlFileS << " <img src=\"Hist_ADC_HF_DS.png\" />" << std::endl;

     htmlFileS << "<h2> 3. ADC Sum in Time Slice </h3>"<< std::endl;
     htmlFileS << "<h3> 3.A. ADC Sum over all channels histogrammed over all events for each depth separately. </h3>"<< std::endl; 
//     htmlFileS << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
     if (sub==1) htmlFileS << " <img src=\"Hist_SumADC_HB.png\" />" << std::endl;
     if (sub==2) htmlFileS << " <img src=\"Hist_SumADC_HE.png\" />" << std::endl;
     if (sub==3) htmlFileS << " <img src=\"Hist_SumADC_HO.png\" />" << std::endl;
     if (sub==4) htmlFileS << " <img src=\"Hist_SumADC_HF.png\" />" << std::endl;

     htmlFileS << "<h3> 3.B. ADC Sum over all channels histogrammed over all events for each depth separately. </h3>"<< std::endl; 
//     htmlFileS << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
     if (sub==1) htmlFileS << " <img src=\"Hist_SumADC_HB0.png\" />" << std::endl;
     if (sub==2) htmlFileS << " <img src=\"Hist_SumADC_HE0.png\" />" << std::endl;
     if (sub==3) htmlFileS << " <img src=\"Hist_SumADC_HO0.png\" />" << std::endl;
     if (sub==4) htmlFileS << " <img src=\"Hist_SumADC_HF0.png\" />" << std::endl;

     htmlFileS << "<h3> 3.C. ADC Sum over all channels histogrammed over all events for each depth separately. </h3>"<< std::endl; 
//     htmlFileS << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
     if (sub==1) htmlFileS << " <img src=\"Hist_SumADC_HB1.png\" />" << std::endl;
     if (sub==2) htmlFileS << " <img src=\"Hist_SumADC_HE1.png\" />" << std::endl;
     if (sub==3) htmlFileS << " <img src=\"Hist_SumADC_HO1.png\" />" << std::endl;
     if (sub==4) htmlFileS << " <img src=\"Hist_SumADC_HF1.png\" />" << std::endl;
*/
    htmlFileS.close();

    ///////////////////////////////////////////

    // Iteration Method for Calibration Group:

    htmlFileM << "</html><html xmlns=\"http://www.w3.org/1999/xhtml\">" << std::endl;
    htmlFileM << "<head>" << std::endl;
    htmlFileM << "<meta http-equiv=\"Content-Type\" content=\"text/html\"/>" << std::endl;
    htmlFileM << "<title> Remote Monitoring Tool </title>" << std::endl;
    htmlFileM << "<style type=\"text/css\">" << std::endl;
    htmlFileM << " body,td{ background-color: #FFFFCC; font-family: arial, arial ce, helvetica; font-size: 12px; }"
              << std::endl;
    htmlFileM << "   td.s0 { font-family: arial, arial ce, helvetica; }" << std::endl;
    htmlFileM << "   td.s1 { font-family: arial, arial ce, helvetica; font-weight: bold; background-color: #FFC169; "
                 "text-align: center;}"
              << std::endl;
    htmlFileM << "   td.s2 { font-family: arial, arial ce, helvetica; background-color: #eeeeee; }" << std::endl;
    htmlFileM << "   td.s3 { font-family: arial, arial ce, helvetica; background-color: #d0d0d0; }" << std::endl;
    htmlFileM << "   td.s4 { font-family: arial, arial ce, helvetica; background-color: #FFC169; }" << std::endl;
    htmlFileM << "</style>" << std::endl;
    htmlFileM << "<body>" << std::endl;

    if (sub == 1)
      htmlFileM << "<h1> Iteration Method for Calibration Group, HB, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 2)
      htmlFileM << "<h1> Iteration Method for Calibration Group, HE, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 3)
      htmlFileM << "<h1> Iteration Method for Calibration Group, HO, RUN = " << runnumber << " </h1>" << std::endl;
    if (sub == 4)
      htmlFileM << "<h1> Iteration Method for Calibration Group, HF, RUN = " << runnumber << " </h1>" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h2> 1:  Positive direction, R = R_depth_ieta_iphi = E_depth_ieta_iphi/E_depth_ieta </h3>"
              << std::endl;

    htmlFileM << "<h3> 1.A. eta/phi-plot: R, averaged over depthes </h3>" << std::endl;
    //     htmlFileM << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist2IterationMethodHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist2IterationMethodHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Rhist2IterationMethodHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist2IterationMethodHF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1B: R vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist1IterationMethodHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Rhist1IterationMethodHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist1IterationMethodHF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    /////////////////////////////////////////////// different Depthes:

    htmlFileM << "<h3> 1C: R vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth1HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1D: R vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth2HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1E: R vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth3HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth3HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1F: R vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth4HE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth4HO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth4HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1G: R vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth5HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1H: R vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth6HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 1I: R vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Rhist1IterationMethodDepth7HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h2> 2: Positive direction,   D(variance)   </h3>" << std::endl;

    htmlFileM << "<h3> 2A: eta/phi-plot: D(variance), averaged over depthes </h3>" << std::endl;
    //     htmlFileM << "<h4> Legend: Bins less "<<Pedest[0][sub]<<" correpond to bad Pedestals </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist2IterationMethodHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist2IterationMethodHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Dhist2IterationMethodHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist2IterationMethodHF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2B: D(variance) vs phi , averaged over depthes & eta </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist1IterationMethodHB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodHE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Dhist1IterationMethodHO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist1IterationMethodHF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    /////////////////////////////////////////////// different Depthes:

    htmlFileM << "<h3> 2C: D(variance) vs phi , different eta, Depth1 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth1HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth1HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth1HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2.D. D(variance) vs phi , different eta, Depth2 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth2HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth2HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth2HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2E: D(variance) vs phi , different eta, Depth3 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth3HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth3HE.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth3HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2F: D(variance) vs phi , different eta, Depth4 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 1)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth4HB.png\" />" << std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth4HE.png\" />" << std::endl;
    if (sub == 3)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth4HO.png\" />" << std::endl;
    if (sub == 4)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth4HF.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2G: D(variance) vs phi , different eta, Depth5 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth5HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2H: D(variance) vs phi , different eta, Depth6 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth6HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM << "<h3> 2I: D(variance) vs phi , different eta, Depth7 </h3>" << std::endl;
    //     htmlFileM << "<h4> Channel legend: white - good, other colour - bad. </h4>"<< std::endl;
    if (sub == 2)
      htmlFileM << " <img src=\"Dhist1IterationMethodDepth7HE.png\" />" << std::endl;
    htmlFileM << "<br>" << std::endl;

    htmlFileM.close();

    ///////////////////////////////////////////
    //
    //
  }  // end sub  //for (int sub=1;sub<=4;sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating subdet  html pages: - rather long time needed, waiting please"
            << std::endl;
  //======================================================================
  // Creating subdet  html pages:

  for (int sub = 1; sub <= 4; sub++) {  //Subdetector: 1-HB, 2-HE, 3-HF, 4-HO
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
    if (sub == 1)
      htmlFile << "<h2> 1. Analysis results for HB</h2>" << std::endl;
    if (sub == 2)
      htmlFile << "<h2> 1. Analysis results for HE</h2>" << std::endl;
    if (sub == 3)
      htmlFile << "<h2> 1. Analysis results for HO</h2>" << std::endl;
    if (sub == 4)
      htmlFile << "<h2> 1. Analysis results for HF</h2>" << std::endl;
    htmlFile << "<table width=\"600\">" << std::endl;
    htmlFile << "<tr>" << std::endl;

    if (sub == 1) {
      htmlFile << "  <td><a href=\"HB_Tile.html\">Megatile Channels</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HB_Calib.html\">Calibration Channels</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HB_Drift.html\">Gain Stability</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HB_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HB_Shapes.html\">ADC Shapes</a></td>" << std::endl;

      /*
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB_Tile.html\">Megatile Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB_Calib.html\">Calibration Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB_Drift.html\">Gain Stability</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB_Pedestals.html\">Pedestals</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB_Shapes.html\">ADC Shapes</a></td>"<< std::endl;
*/
    }
    if (sub == 2) {
      htmlFile << "  <td><a href=\"HE_Tile.html\">Megatile Channels</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_Calib.html\">Calibration Channels</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_Drift.html\">Gain Stability</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_Shapes.html\">ADC Shapes</a></td>" << std::endl;
      htmlFile << "  <td><a href=\"HE_IterationMethod.html\">Iteration Method</a></td>" << std::endl;

      /*
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_Tile.html\">Megatile Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_Calib.html\">Calibration Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_Drift.html\">Gain Stability</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_Pedestals.html\">Pedestals</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_Shapes.html\">ADC Shapes</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE_IterationMethod.html\">Iteration Method</a></td>"<< std::endl;
*/
    }
    if (sub == 3) {
      /*
       htmlFile << "  <td><a href=\"HO_Tile.html\">Megatile Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HO_Calib.html\">Calibration Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HO_Drift.html\">Gain Stability</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HO_Pedestals.html\">Pedestals</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HO_Shapes.html\">ADC Shapes</a></td>"<< std::endl;
       */
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HO_Tile.html\">Megatile Channels</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HO_Calib.html\">Calibration Channels</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HO_Drift.html\">Gain Stability</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HO_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HO_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }
    if (sub == 4) {
      /*
       htmlFile << "  <td><a href=\"HF_Tile.html\">Megatile Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HF_Calib.html\">Calibration Channels</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HF_Drift.html\">Gain Stability</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HF_Pedestals.html\">Pedestals</a></td>"<< std::endl;
       htmlFile << "  <td><a href=\"HF_Shapes.html\">ADC Shapes</a></td>"<< std::endl;
       */
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HF_Tile.html\">Megatile Channels</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HF_Calib.html\">Calibration Channels</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HF_Drift.html\">Gain Stability</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HF_Pedestals.html\">Pedestals</a></td>" << std::endl;
      htmlFile << "  <td><a "
                  "href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/"
                  "HcalRemoteMonitoring/RMT/LED_"
               << runnumber << "/HF_Shapes.html\">ADC Shapes</a></td>" << std::endl;
    }

    htmlFile << "</tr>" << std::endl;
    htmlFile << "</table>" << std::endl;
    htmlFile << "<br>" << std::endl;
    if (sub == 1)
      htmlFile << "<h2> 2.Status HB over all criteria </h2>" << std::endl;
    if (sub == 2)
      htmlFile << "<h2> 2.Status HE over all criteria </h2>" << std::endl;
    if (sub == 3)
      htmlFile << "<h2> 2.Status HO over all criteria </h2>" << std::endl;
    if (sub == 4)
      htmlFile << "<h2> 2.Status HF over all criteria </h2>" << std::endl;
    htmlFile << "<h3> 2.A.Channel map for each Depth </h3>" << std::endl;
    htmlFile << "<h4> Channel legend: green - good, red - bad (rate of failures at least 0.1), yellow - at least 2% "
                "gain drift, white - not applicable or out of range </h4>"
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

    htmlFile << "<h3> 2.B.List of Bad channels (rate > 0.1) and its rates for each RMT criteria (for GS - %) </h3>"
             << std::endl;

    htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>" << std::endl;
    //   htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

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
    htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Am</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>" << std::endl;
    htmlFile << "<td class=\"s9\" align=\"center\">GS (%)</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>" << std::endl;

    htmlFile << "</tr>" << std::endl;

    for (int i = 1; i <= NBad; i++) {
      if ((ind % 2) == 1) {
        raw_class = "<td class=\"s2\" align=\"center\">";
        raw_class1 = "<td class=\"s6\" align=\"center\">";
        raw_class2 = "<td class=\"s11\" align=\"center\">";
        raw_class3 = "<td class=\"s13\" align=\"center\">";

      } else {
        raw_class = "<td class=\"s3\" align=\"center\">";
        raw_class1 = "<td class=\"s7\" align=\"center\">";
        raw_class2 = "<td class=\"s12\" align=\"center\">";
        raw_class3 = "<td class=\"s14\" align=\"center\">";
      }
      const CellDB db;
      CellDB ce;
      if ((ce.size() >= 1) && (Sub[2][i] == sub)) {
        if (Sub[2][i] == 1) {
          ce = db.find("subdet", "HB").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HB, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HB, Eta=" << Eta[2][i] << ", Phi=" << Phi[2][i]
                 << ", Depth=" << Depth[2][i] << " in database" << endl;
          }
        }
        if (Sub[2][i] == 2) {
          ce = db.find("subdet", "HE").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HE, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HE, Eta=" << Eta[2][i] << ", Phi=" << Phi[2][i]
                 << ", Depth=" << Depth[2][i] << " in database" << endl;
          }
        }
        if (Sub[2][i] == 3) {
          ce = db.find("subdet", "HO").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HO, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HO, Eta=" << Eta[2][i] << ", Phi=" << Phi[2][i]
                 << ", Depth=" << Depth[2][i] << " in database" << endl;
          }
        }
        if (Sub[2][i] == 4) {
          ce = db.find("subdet", "HF").find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HF, Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HF, Eta=" << Eta[2][i] << ", Phi=" << Phi[2][i]
                 << ", Depth=" << Depth[2][i] << " in database" << endl;
          }
        }
        htmlFile << "<tr>" << std::endl;
        htmlFile << "<td class=\"s4\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
        htmlFile << raw_class << Eta[2][i] << "</td>" << std::endl;
        htmlFile << raw_class << Phi[2][i] << "</td>" << std::endl;
        htmlFile << raw_class << Depth[2][i] << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[1][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[2][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[3][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[4][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[5][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[6][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class3 << Map_Ampl[21][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[31][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[32][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[11][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[12][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[13][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[14][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[15][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[16][Sub[2][i]][Depth[2][i]]->GetBinContent(Eta[2][i] + 41, Phi[2][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << "</tr>" << std::endl;

        ind += 1;
      }
    }
    htmlFile << "</table>" << std::endl;
    htmlFile << "<br>" << std::endl;

    htmlFile << "<h3> 2.C.List of Gain unstable channels and its value in % (for other criterias - rate)</h3>"
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
    htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Am</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>" << std::endl;
    htmlFile << "<td class=\"s9\" align=\"center\">GS(%)</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>" << std::endl;
    htmlFile << "</tr>" << std::endl;

    for (int i = 1; i <= NWarn; i++) {
      if ((ind % 2) == 1) {
        raw_class = "<td class=\"s2\" align=\"center\">";
        raw_class1 = "<td class=\"s6\" align=\"center\">";
        raw_class2 = "<td class=\"s11\" align=\"center\">";
        raw_class3 = "<td class=\"s13\" align=\"center\">";

      } else {
        raw_class = "<td class=\"s3\" align=\"center\">";
        raw_class1 = "<td class=\"s7\" align=\"center\">";
        raw_class2 = "<td class=\"s12\" align=\"center\">";
        raw_class3 = "<td class=\"s14\" align=\"center\">";
      }
      const CellDB db;
      CellDB ce;
      if ((ce.size() >= 1) && (Sub[1][i] == sub)) {
        if (Sub[1][i] == 1) {
          ce = db.find("subdet", "HB").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HB, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HB, Eta=" << Eta[1][i] << ", Phi=" << Phi[1][i]
                 << ", Depth=" << Depth[1][i] << " in database" << endl;
          }
        }
        if (Sub[1][i] == 2) {
          ce = db.find("subdet", "HE").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HE, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HE, Eta=" << Eta[1][i] << ", Phi=" << Phi[1][i]
                 << ", Depth=" << Depth[1][i] << " in database" << endl;
          }
        }
        if (Sub[1][i] == 3) {
          ce = db.find("subdet", "HO").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HO, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HO, Eta=" << Eta[1][i] << ", Phi=" << Phi[1][i]
                 << ", Depth=" << Depth[1][i] << " in database" << endl;
          }
        }
        if (Sub[1][i] == 4) {
          ce = db.find("subdet", "HF").find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HF, Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HF, Eta=" << Eta[1][i] << ", Phi=" << Phi[1][i]
                 << ", Depth=" << Depth[1][i] << " in database" << endl;
          }
        }
        htmlFile << "<td class=\"s4\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
        htmlFile << raw_class << Eta[1][i] << "</td>" << std::endl;
        htmlFile << raw_class << Phi[1][i] << "</td>" << std::endl;
        htmlFile << raw_class << Depth[1][i] << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[1][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[2][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[3][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[4][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[5][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[6][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class3 << Map_Ampl[21][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[31][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[32][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[11][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[12][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[13][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[14][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[15][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[16][Sub[1][i]][Depth[1][i]]->GetBinContent(Eta[1][i] + 41, Phi[1][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << "</tr>" << std::endl;
        htmlFile << "</tr>" << std::endl;
        ind += 1;
      }
    }
    htmlFile << "</table>" << std::endl;
    htmlFile << "<br>" << std::endl;

    htmlFile << "<h3> 2.D.List of channels with Bad Pedestals (rate > 0.1) and its rates (for GS - %)</h3>"
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
    htmlFile << "<td class=\"s5\" align=\"center\">Cm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Am</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Wm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">Rm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TNm</td>" << std::endl;
    htmlFile << "<td class=\"s5\" align=\"center\">TXm</td>" << std::endl;
    htmlFile << "<td class=\"s9\" align=\"center\">GS(%)</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">Pm</td>" << std::endl;
    htmlFile << "<td class=\"s4\" align=\"center\">pWm</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Cc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Ac</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Wc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">Rc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TNc</td>" << std::endl;
    htmlFile << "<td class=\"s8\" align=\"center\">TXc</td>" << std::endl;
    htmlFile << "</tr>" << std::endl;

    for (int i = 1; i <= NPed; i++) {
      if ((ind % 2) == 1) {
        raw_class = "<td class=\"s2\" align=\"center\">";
        raw_class1 = "<td class=\"s6\" align=\"center\">";
        raw_class2 = "<td class=\"s11\" align=\"center\">";
        raw_class3 = "<td class=\"s13\" align=\"center\">";

      } else {
        raw_class = "<td class=\"s3\" align=\"center\">";
        raw_class1 = "<td class=\"s7\" align=\"center\">";
        raw_class2 = "<td class=\"s12\" align=\"center\">";
        raw_class3 = "<td class=\"s14\" align=\"center\">";
      }
      const CellDB db;
      CellDB ce;
      if ((ce.size() >= 1) && (Sub[3][i] == sub)) {
        if (Sub[3][i] == 1) {
          ce = db.find("subdet", "HB").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HB, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HB, Eta=" << Eta[3][i] << ", Phi=" << Phi[3][i]
                 << ", Depth=" << Depth[3][i] << " in database" << endl;
          }
        }
        if (Sub[3][i] == 2) {
          ce = db.find("subdet", "HE").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HE, Eta="<< Eta[3][i]  <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HE, Eta=" << Eta[3][i] << ", Phi=" << Phi[3][i]
                 << ", Depth=" << Depth[3][i] << " in database" << endl;
          }
        }
        if (Sub[3][i] == 3) {
          ce = db.find("subdet", "HO").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HO, Eta="<< Eta[3][i] <<", Phi="<<  Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HO, Eta=" << Eta[3][i] << ", Phi=" << Phi[3][i]
                 << ", Depth=" << Depth[3][i] << " in database" << endl;
          }
        }
        if (Sub[3][i] == 4) {
          ce = db.find("subdet", "HF").find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
          if (ce.size() == 0) {
            //		cout<<"Error: No such HF, Eta="<< Eta[3][i] <<", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;
            continue;
          } else if (ce.size() > 1) {
            cout << "Warning: More than one line correspond to such HF, Eta=" << Eta[3][i] << ", Phi=" << Phi[3][i]
                 << ", Depth=" << Depth[3][i] << " in database" << endl;
          }
        }
        htmlFile << "<td class=\"s4\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
        htmlFile << raw_class << Eta[3][i] << "</td>" << std::endl;
        htmlFile << raw_class << Phi[3][i] << "</td>" << std::endl;
        htmlFile << raw_class << Depth[3][i] << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
        htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[1][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[2][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[3][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[4][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[5][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class1 << Map_Ampl[6][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class3 << Map_Ampl[21][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[31][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class << Map_Ampl[32][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[11][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[12][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[13][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[14][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[15][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << raw_class2 << Map_Ampl[16][Sub[3][i]][Depth[3][i]]->GetBinContent(Eta[3][i] + 41, Phi[3][i] + 1)
                 << "</td>" << std::endl;
        htmlFile << "</tr>" << std::endl;
        htmlFile << "</tr>" << std::endl;
        ind += 1;
      }
    }
    htmlFile << "</table>" << std::endl;

    htmlFile << "</body> " << std::endl;
    htmlFile << "</html> " << std::endl;
    htmlFile.close();
  }

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating description HELP html file:" << std::endl;
  //======================================================================
  // Creating description html file:
  ofstream htmlFile;
  htmlFile.open("HELP.html");
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
  htmlFile << "</style>" << std::endl;
  htmlFile << "<body>" << std::endl;
  htmlFile << "<h1>  Description of Remote Monitoring Tool criteria for bad channel selection</h1>" << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - C means CAPID Errors assuming we inspect CAPID non-rotation,error & validation bits, and for "
              "this criterion - no need to apply any cuts to select bcs.</h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - A means full amplitude, collected over all time slices </h3> " << std::endl;
  htmlFile << "<h3> - R means ratio criterion where we define as a bad, the channels, for which the signal portion in "
              "4 middle TSs(plus one, minus two around TS with maximal amplitude) is out of some range of reasonable "
              "values </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - W means width of shape distribution. Width is defined as square root from dispersion. </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - TN means mean time position of adc signal. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - TX means TS number of maximum signal </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - m means megatile channels. For example Am means Amplitude criteria for megatile channels </h3> "
           << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile
      << "<h3> - c means calibration channels. For example Ac means Amplitude criteria for calibration channels </h3> "
      << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - Pm means Pedestals. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<h3> - pWm  means pedestal Width. </h3> " << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "</body> " << std::endl;
  htmlFile << "</html> " << std::endl;
  htmlFile.close();

  //======================================================================

  std::cout << "********" << std::endl;
  std::cout << "************    Start creating MAP html file: - rather long time needed, waiting please" << std::endl;
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

  htmlFile << "<h1> Remote Monitoring Tool, LED RUN = " << runnumber << ". </h1>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h2> 1. Analysis results for subdetectors </h2>" << std::endl;
  htmlFile << "<table width=\"400\">" << std::endl;
  htmlFile << "<tr>" << std::endl;

  htmlFile << "  <td><a href=\"HB.html\">HB</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"HE.html\">HE</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"HO.html\">HO</a></td>" << std::endl;
  htmlFile << "  <td><a href=\"HF.html\">HF</a></td>" << std::endl;

  /*
     htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HB.html\">HB</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HE.html\">HE</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HO.html\">HO</a></td>"<< std::endl;
     htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HF.html\">HF</a></td>"<< std::endl;
*/
  htmlFile << "</tr>" << std::endl;
  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h2> 2. Amplitude summed over all sub-detector channels vs first 1000 events of this Run </h2>"
           << std::endl;
  htmlFile << "<h3> 2.A. Total subdetector Amplitude vs iEvent </h3>" << std::endl;
  htmlFile << " <img src=\"EVENTDEPENDENCE.png\" />" << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h2> 3. HCAL status over all criteria and subdetectors </h2>" << std::endl;
  htmlFile << "<h3> 3.A. Channels in detector space </h3>" << std::endl;
  htmlFile << "<h4> Legend for channel status: green - good, red - bad, yellow - at least 2% gain drift, white - not "
              "applicable or out of range </h4>"
           << std::endl;
  htmlFile << " <img src=\"MAP.png\" />" << std::endl;
  htmlFile << "<br>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h3> 3.B. List of Bad channels </h3>" << std::endl;

  htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>" << std::endl;
  //htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;
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

  ind = 0;

  for (int i = 1; i <= NBad; i++) {
    if ((ind % 2) == 1) {
      raw_class = "<td class=\"s2\" align=\"center\">";
      raw_class1 = "<td class=\"s6\" align=\"center\">";
    } else {
      raw_class = "<td class=\"s3\" align=\"center\">";
      raw_class1 = "<td class=\"s7\" align=\"center\">";
    }
    const CellDB db;
    const CellDB ce = db.find("Eta", Eta[2][i]).find("Phi", Phi[2][i]).find("Depth", Depth[2][i]);
    if (ce.size() == 0) {
      //	       cout<<"Error: No such Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;
      continue;
    }
    //	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[2][i] <<", Phi="<< Phi[2][i] <<", Depth="<< Depth[2][i] <<" in database"<<endl;}

    if (ce.size() >= 1) {
      htmlFile << "<tr>" << std::endl;
      htmlFile << "<td class=\"s1\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
      htmlFile << raw_class << Eta[2][i] << "</td>" << std::endl;
      htmlFile << raw_class << Phi[2][i] << "</td>" << std::endl;
      htmlFile << raw_class << Depth[2][i] << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
      htmlFile << raw_class1 << Comment[2][i] << "</td>" << std::endl;
      htmlFile << "</tr>" << std::endl;

      ind += 1;
    }
  }
  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h3> 2.C.List of Gain unstable channels </h3>" << std::endl;
  htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>" << std::endl;
  //   htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

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
  htmlFile << "<td class=\"s5\" align=\"center\">Failed criteria</td>" << std::endl;
  htmlFile << "</tr>" << std::endl;

  for (int i = 1; i <= NWarn; i++) {
    if ((ind % 2) == 1) {
      raw_class = "<td class=\"s2\" align=\"center\">";
      raw_class1 = "<td class=\"s6\" align=\"center\">";
    } else {
      raw_class = "<td class=\"s3\" align=\"center\">";
      raw_class1 = "<td class=\"s7\" align=\"center\">";
    }
    const CellDB db;
    const CellDB ce = db.find("Eta", Eta[1][i]).find("Phi", Phi[1][i]).find("Depth", Depth[1][i]);
    if (ce.size() == 0) {
      //	      cout<<"Error: No such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;
      continue;
    }
    //	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}

    if (ce.size() >= 1) {
      htmlFile << "<tr>" << std::endl;
      htmlFile << "<td class=\"s1\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
      htmlFile << raw_class << Eta[1][i] << "</td>" << std::endl;
      htmlFile << raw_class << Phi[1][i] << "</td>" << std::endl;
      htmlFile << raw_class << Depth[1][i] << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
      htmlFile << raw_class1 << Comment[1][i] << "</td>" << std::endl;
      htmlFile << "</tr>" << std::endl;

      ind += 1;
    }
  }

  htmlFile << "</table>" << std::endl;
  htmlFile << "<br>" << std::endl;

  htmlFile << "<h3> 2.D.List of channels with bad Pedestals </h3>" << std::endl;
  htmlFile << "  <td><a href=\"HELP.html\"> Description of criteria for bad channel selection</a></td>" << std::endl;
  //   htmlFile << "  <td><a href=\"https://cms-cpt-software.web.cern.ch/cms-cpt-software/General/Validation/SVSuite/HcalRemoteMonitoring/RMT/LED_"<<runnumber<<"/HELP.html\"> Description of criteria for bad channel selection</a></td>"<< std::endl;

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
  htmlFile << "<td class=\"s5\" align=\"center\">Failed criteria</td>" << std::endl;
  htmlFile << "</tr>" << std::endl;

  for (int i = 1; i <= NPed; i++) {
    if ((ind % 2) == 1) {
      raw_class = "<td class=\"s2\" align=\"center\">";
      raw_class1 = "<td class=\"s6\" align=\"center\">";
    } else {
      raw_class = "<td class=\"s3\" align=\"center\">";
      raw_class1 = "<td class=\"s7\" align=\"center\">";
    }
    const CellDB db;
    const CellDB ce = db.find("Eta", Eta[3][i]).find("Phi", Phi[3][i]).find("Depth", Depth[3][i]);
    if (ce.size() == 0) {
      //	      cout<<"Error: No such Eta="<< Eta[3][i] << ", Phi="<< Phi[3][i] <<", Depth="<< Depth[3][i] <<" in database"<<endl;
      continue;
    }
    //	else if (ce.size()>1) { cout<<"Warning: More than one line correspond to such Eta="<< Eta[1][i] <<", Phi="<< Phi[1][i] <<", Depth="<< Depth[1][i] <<" in database"<<endl;}

    if (ce.size() >= 1) {
      htmlFile << "<tr>" << std::endl;
      htmlFile << "<td class=\"s1\" align=\"center\">" << ind + 1 << "</td>" << std::endl;
      htmlFile << raw_class << Eta[3][i] << "</td>" << std::endl;
      htmlFile << raw_class << Phi[3][i] << "</td>" << std::endl;
      htmlFile << raw_class << Depth[3][i] << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RBX << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RM << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Pixel << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].RMfiber << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberCh << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].QIE << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].ADC << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].VMECardID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].dccID << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].Spigot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].FiberIndex << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrSlot << "</td>" << std::endl;
      htmlFile << raw_class << ce[0].HtrTB << "</td>" << std::endl;
      htmlFile << raw_class1 << Comment[3][i] << "</td>" << std::endl;
      htmlFile << "</tr>" << std::endl;

      ind += 1;
    }
  }

  htmlFile << "</table>" << std::endl;

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
