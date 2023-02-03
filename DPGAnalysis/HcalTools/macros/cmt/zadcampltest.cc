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
#include <TPostScript.h>
#include <TClass.h>

//
using namespace std;
//
//

//inline void HERE(const char *msg) { std::cout << msg << std::endl; }

int main(int argc, char *argv[]) {
  std::string dirnm = "Analyzer";
  //======================================================================
  printf("reco: gROOT Reset \n");
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  //			gStyle->SetOptStat(0);   //  no statistics _or_
  //	        	  gStyle->SetOptStat(11111111);
  //gStyle->SetOptStat(1101);// name mean and rms
  //	gStyle->SetOptStat(0101);// name and entries
  //	   gStyle->SetOptStat(1100);// mean and rms only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //	gStyle->SetOptStat(1110000);// und over, integral !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  gStyle->SetOptStat(101110);  // entries, mean, rms, overflow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                               //	gStyle->SetOptStat(100000);//  over !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //
  //gStyle->SetOptFit(00010);// constant, mean and sigma only !!
  //	gStyle->SetOptFit(00001);// hi2/nu, constant, mean and sigma only !!
  gStyle->SetOptFit(0010);  // constant, mean and sigma only !!
  //	gStyle->SetOptFit(00011);// constant, mean and sigma only !!
  // gStyle->SetOptFit(1101);
  //	   gStyle->SetOptFit(1011);
  //
  //gStyle->SetStatX(0.98);
  //gStyle->SetStatY(0.99);
  //gStyle->SetStatW(0.30);
  //gStyle->SetStatH(0.25);
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
  //---=[ Histogram style ]=----------
  //        gStyle->SetHistFillColor(38);
  //	        gStyle->SetFrameFillColor(41);// jelto-kori4nev-svetl
  //	gStyle->SetFrameFillColor(5);// jeltyi
  //	gStyle->SetFrameFillColor(17);// seryi
  //	gStyle->SetFrameFillColor(18);// svetlo seryi
  //	gStyle->SetFrameFillColor(20);// svetlo kori4nev
  //        gStyle->SetFrameFillColor(33);// sine-seryi
  //	gStyle->SetFrameFillColor(40);// fiolet-seryi
  //	gStyle->SetFrameFillColor(23);// sv.kor

  //---=[ Pad style ]=----------------
  gStyle->SetPadTopMargin(TopOffset);
  gStyle->SetPadBottomMargin(LeftOffset);
  gStyle->SetPadRightMargin(TopOffset);
  gStyle->SetPadLeftMargin(LeftOffset);
  //---=[ SetCanvasDef           ]=----------------
  //======================================================================
  //
  // Connect the input file and get the 2-d histogram in memory
  //======================================================================
  //  TBrowser *b = new TBrowser

  //	TFile *hfile1= new TFile("test.root", "READ");
  //

  //	TFile *hfile1= new TFile("testNZS.root", "READ");
  //	TFile *hfile1= new TFile("test.root", "READ");

  //	TFile *hfile1= new TFile("newruns/Global_234034.root", "READ");
  //

  //	TFile *hfile1= new TFile("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/CMTweb/histos/Global_234556.root", "READ");
  //	TFile *hfile1= new TFile("Global_234034.root", "READ");
  //	TFile *hfile1= new TFile("test211006.root", "READ");
  //	TFile *hfile1= new TFile("test234457.root", "READ");

  //	TFile *hfile1= new TFile("Global_51.root", "READ");
  //	TFile *hfile1= new TFile("Global_235698.root", "READ");

  //	TFile *hfile1= new TFile("Global_39.root", "READ");
  //	TFile *hfile1= new TFile("test237165.root", "READ");
  //
  //	TFile *hfile1= new TFile("Laser_238187.root", "READ");
  //	TFile *hfile1= new TFile("Laser_238183.root", "READ");

  //	TFile *hfile1= new TFile("Global_255031.root", "READ");

  //	TFile *hfile1= new TFile("Global_256001.root", "READ");
  //	TFile *hfile1= new TFile("Global_256167.root", "READ");
  // 	TFile *hfile1= new TFile("Global_256348.root", "READ");
  //	TFile *hfile1= new TFile("Global_256630.root", "READ");

  //	TFile *hfile1= new TFile("../PYTHON_runlist_test/Global_283884_1.root", "READ");
  //        TFile *hfile1= new TFile("Global_test.root", "READ");

  //	TFile *hfile1= new TFile("LED_280702.root", "READ");
  //	TFile *hfile2= new TFile("LED_287824.root", "READ");

  //	TFile *hfile1= new TFile("LED_284352.root", "READ");
  //	TFile *hfile1= new TFile("LEDtest.root", "READ");
  //  TFile *hfile1 = new TFile("Global_346445.root", "READ");

  TFile *hfile1 = new TFile("Global_362365.root", "READ");
  //          TH1D *hist1(nullptr);
  //            hist1 = (TH1D *)dir->FindObjectAny("h_mapDepth1_HE");

  //	TFile *hfile2= new TFile("LED_284902.root", "READ");
  //	TFile *hfile2= new TFile("LED_284499.root", "READ");
  //	TFile *hfile2= new TFile("LED_284352.root", "READ");

  //	TFile *hfile2= new TFile("LED_286590.root", "READ");

  //    getchar();
  //
  TPostScript psfile("zadcampltest.ps", 111);

  //

  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);

  hfile1->ls();
  TDirectory *dir = (TDirectory *)hfile1->FindObjectAny(dirnm.c_str());

  //========================================================================================== 1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  //  TH2F *Zzzdepth1hef1 = (TH2F *)hfile1->Get("h_mapDepth1_HE");
  TH2F *Zzzdepth1hef1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef1->SetXTitle("#eta \b");
  Zzzdepth1hef1->SetYTitle("#phi \b");
  Zzzdepth1hef1->SetZTitle("h_mapDepth1_HE \b");
  Zzzdepth1hef1->Draw("COLZ");

  c1->cd(2);
  TH2F *Zzzdepth1hef2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef2->SetXTitle("#eta \b");
  Zzzdepth1hef2->SetYTitle("#phi \b");
  Zzzdepth1hef2->SetZTitle("h_mapDepth2_HE \b");
  Zzzdepth1hef2->Draw("COLZ");

  c1->cd(3);
  TH2F *Zzzdepth1hef3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef3->SetXTitle("#eta \b");
  Zzzdepth1hef3->SetYTitle("#phi \b");
  Zzzdepth1hef3->SetZTitle("h_mapDepth3_HE \b");
  Zzzdepth1hef3->Draw("COLZ");

  c1->cd(4);
  TH2F *Zzzdepth1hef4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef4->SetXTitle("#eta \b");
  Zzzdepth1hef4->SetYTitle("#phi \b");
  Zzzdepth1hef4->SetZTitle("h_mapDepth4_HE \b");
  Zzzdepth1hef4->Draw("COLZ");

  c1->cd(5);
  TH2F *Zzzdepth1hef5 = (TH2F *)dir->FindObjectAny("h_mapDepth5_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef5->SetXTitle("#eta \b");
  Zzzdepth1hef5->SetYTitle("#phi \b");
  Zzzdepth1hef5->SetZTitle("h_mapDepth5_HE \b");
  Zzzdepth1hef5->Draw("COLZ");

  c1->cd(6);
  TH2F *Zzzdepth1hef6 = (TH2F *)dir->FindObjectAny("h_mapDepth6_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef6->SetXTitle("#eta \b");
  Zzzdepth1hef6->SetYTitle("#phi \b");
  Zzzdepth1hef6->SetZTitle("h_mapDepth6_HE \b");
  Zzzdepth1hef6->Draw("COLZ");

  c1->cd(7);
  TH2F *Zzzdepth1hef7 = (TH2F *)dir->FindObjectAny("h_mapDepth7_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Zzzdepth1hef7->SetXTitle("#eta \b");
  Zzzdepth1hef7->SetYTitle("#phi \b");
  Zzzdepth1hef7->SetZTitle("h_mapDepth7_HE \b");
  Zzzdepth1hef7->Draw("COLZ");

  c1->cd(8);
  TH1F *jdjswi8 = (TH1F *)dir->FindObjectAny("h_ADCAmpl_HE");
  gPad->SetLogy(kFALSE);
  // gPad->SetLogx();
  jdjswi8->SetMarkerStyle(20);
  jdjswi8->SetMarkerSize(0.8);
  jdjswi8->GetYaxis()->SetLabelSize(0.04);
  jdjswi8->SetXTitle("h_ADCAmpl_HE \b");
  jdjswi8->SetMarkerColor(2);
  jdjswi8->SetLineColor(2);
  jdjswi8->Draw("");

  c1->Update();

  //========================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH1F *wqwhef1 = (TH1F *)dir->FindObjectAny("h_mapDepth1ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef1->SetZTitle("h_mapDepth1ADCAmpl_HE \b");
  wqwhef1->Draw("COLZ");

  c1->cd(2);
  TH1F *wqwhef2 = (TH1F *)dir->FindObjectAny("h_mapDepth2ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef2->SetZTitle("h_mapDepth2ADCAmpl_HE \b");
  wqwhef2->Draw("COLZ");

  c1->cd(3);
  TH1F *wqwhef3 = (TH1F *)dir->FindObjectAny("h_mapDepth3ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef3->SetZTitle("h_mapDepth3ADCAmpl_HE \b");
  wqwhef3->Draw("COLZ");

  c1->cd(4);
  TH1F *wqwhef4 = (TH1F *)dir->FindObjectAny("h_mapDepth4ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef4->SetZTitle("h_mapDepth4ADCAmpl_HE \b");
  wqwhef4->Draw("COLZ");

  c1->cd(5);
  TH1F *wqwhef5 = (TH1F *)dir->FindObjectAny("h_mapDepth5ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef5->SetZTitle("h_mapDepth5ADCAmpl_HE \b");
  wqwhef5->Draw("COLZ");

  c1->cd(6);
  TH1F *wqwhef6 = (TH1F *)dir->FindObjectAny("h_mapDepth6ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef6->SetZTitle("h_mapDepth6ADCAmpl_HE \b");
  wqwhef6->Draw("COLZ");

  c1->cd(7);
  TH1F *wqwhef7 = (TH1F *)dir->FindObjectAny("h_mapDepth7ADCAmpl_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  wqwhef7->SetZTitle("h_mapDepth7ADCAmpl_HE \b");
  wqwhef7->Draw("COLZ");

  c1->Update();

  //========================================================================================== 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH1F *azzdepth1hef1 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLS3");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef1->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef1->Draw("");

  c1->cd(2);
  TH1F *azzdepth1hef2 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLS4");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef2->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef2->Draw("");

  c1->cd(3);
  TH1F *azzdepth1hef3 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLS5");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef3->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef3->Draw("");

  c1->cd(4);
  TH1F *azzdepth1hef4 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLSdepth4HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef4->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef4->Draw("");

  c1->cd(5);
  TH1F *azzdepth1hef5 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLSdepth5HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef5->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef5->Draw("");

  c1->cd(6);
  TH1F *azzdepth1hef6 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLSdepth6HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef6->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef6->Draw("");

  c1->cd(7);
  TH1F *azzdepth1hef7 = (TH1F *)dir->FindObjectAny("h_sumADCAmplperLSdepth7HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  azzdepth1hef7->SetZTitle("h_sumADCAmplperLS3 \b");
  azzdepth1hef7->Draw("");

  c1->Update();

  //========================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH1F *poqw1 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLS3");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw1->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw1->Draw("");

  c1->cd(2);
  TH1F *poqw2 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLS4");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw2->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw2->Draw("");

  c1->cd(3);
  TH1F *poqw3 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLS5");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw3->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw3->Draw("");

  c1->cd(4);
  TH1F *poqw4 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLSdepth4HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw4->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw4->Draw("");

  c1->cd(5);
  TH1F *poqw5 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLSdepth5HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw5->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw5->Draw("");

  c1->cd(6);
  TH1F *poqw6 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLSdepth6HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw6->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw6->Draw("");

  c1->cd(7);
  TH1F *poqw7 = (TH1F *)dir->FindObjectAny("h_sumCutADCAmplperLSdepth7HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  poqw7->SetZTitle("h_sumCutADCAmplperLS3 \b");
  poqw7->Draw("");

  c1->Update();

  //========================================================================================== 5
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH1F *cdew1 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLS3");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew1->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew1->Draw("");

  c1->cd(2);
  TH1F *cdew2 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLS4");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew2->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew2->Draw("");

  c1->cd(3);
  TH1F *cdew3 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLS5");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew3->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew3->Draw("");

  c1->cd(4);
  TH1F *cdew4 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLSdepth4HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew4->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew4->Draw("");

  c1->cd(5);
  TH1F *cdew5 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLSdepth5HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew5->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew5->Draw("");

  c1->cd(6);
  TH1F *cdew6 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLSdepth6HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew6->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew6->Draw("");

  c1->cd(7);
  TH1F *cdew7 = (TH1F *)dir->FindObjectAny("h_sum0ADCAmplperLSdepth7HEu");
  gPad->SetGridy();
  gPad->SetGridx();
  cdew7->SetZTitle("h_sum0ADCAmplperLS3 \b");
  cdew7->Draw("");

  c1->Update();

  //======================================================================
  //==================================================================================================== end
  //======================================================================
  //======================================================================
  // close and delete all possible things:

  //   psfile->Close();
  psfile.Close();

  hfile1->Close();
  //    hfile1->Delete();
  hfile1->Close();
  //    hfile1->Delete();

  //  Exit Root
  gSystem->Exit(0);
  //======================================================================
}
