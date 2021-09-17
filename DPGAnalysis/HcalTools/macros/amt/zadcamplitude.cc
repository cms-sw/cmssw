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
  //======================================================================
  printf("reco: gROOT Reset \n");
  gROOT->Reset();
  gROOT->SetStyle("Plain");
  //		gStyle->SetOptStat(0);   //  no statistics _or_
  //	        	  gStyle->SetOptStat(11111111);
  //gStyle->SetOptStat(1101);// name mean and rms
  //	gStyle->SetOptStat(0101);// name and entries
  //	   gStyle->SetOptStat(1100);// mean and rms only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  gStyle->SetOptStat(1110000);  // und over, integral !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //	gStyle->SetOptStat(101110);                                          // entries, mean, rms, overflow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //	gStyle->SetOptStat(100000);//  over !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //
  //gStyle->SetOptFit(00010);// constant, mean and sigma only !!
  //	gStyle->SetOptFit(00001);// hi2/nu, constant, mean and sigma only !!
  //	gStyle->SetOptFit(0010);// constant, mean and sigma only !!
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

  //	TFile *hfile1= new TFile("/afs/cern.ch/cms/CAF/CMSALCA/ALCA_HCALCALIB/HCALMONITORING/CMTweb/histos/Global_234556.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_41.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600.root_no41", "READ");
  //	TFile *hfile1= new TFile("Global_325001_ls1to600.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_325001_40.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_325001_ls1to600.root", "READ");
  ////////////////////////////////////////////////////////////
  //	TFile *hfile1= new TFile("Global_321177_41_abortgap.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600_abortgap.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600_abortgap_no41.root", "READ");
  //	TFile *hfile1= new TFile("Global_325001_ls1to600_abortgap.root", "READ");
  //	TFile *hfile1= new TFile("Global_321624_1.root", "READ");
  //	TFile *hfile1= new TFile("Global_321625.root", "READ");
  //	TFile *hfile1= new TFile("Global_321313.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_325001.root", "READ");

  //	TFile *hfile1= new TFile("Global_RBX_325001test.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_321177test.root", "READ");
  //	TFile *hfile1= new TFile("LED_327785test.root", "READ");

  TFile *hfile1 = new TFile("Global_RBX_325001.root", "READ");
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //    getchar();
  //
  TPostScript psfile("zadcamplitude.ps", 111);

  //

  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);

  //=============================================================================================== 1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth1hb1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl_HB");
  TH2F *twoddepth1hb0 = (TH2F *)hfile1->Get("h_mapDepth1_HB");
  twoddepth1hb1->Sumw2();
  twoddepth1hb0->Sumw2();
  //    if(twoddepth1hb0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1hbff = (TH2F *)twoddepth1hb1->Clone("Cdepth1hbff");
  Cdepth1hbff->Divide(twoddepth1hb1, twoddepth1hb0, 1, 1, "B");  // average A
  Cdepth1hbff->Sumw2();
  //    }

  c1->cd(1);
  TH2F *twoedepth1hb1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl225_HB");
  TH2F *twoedepth1hb0 = (TH2F *)hfile1->Get("h_mapDepth1_HB");
  twoedepth1hb1->Sumw2();
  twoedepth1hb0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1hbfz225 = (TH2F *)twoedepth1hb1->Clone("Cdepth1hbfz225");
  Cdepth1hbfz225->Divide(twoedepth1hb1, twoedepth1hb0, 1, 1, "B");  // just RATE
  Cdepth1hbfz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth1hbfz225->SetMarkerStyle(20);
  Cdepth1hbfz225->SetMarkerSize(0.4);
  //    Cdepth1hbfz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth1hbfz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth1hbfz225->SetXTitle("#eta \b");
  Cdepth1hbfz225->SetYTitle("#phi \b");
  Cdepth1hbfz225->SetZTitle("Rate for ADCAmpl in each event & cell - HB Depth1 \b");
  Cdepth1hbfz225->SetMarkerColor(2);
  Cdepth1hbfz225->SetLineColor(2);
  Cdepth1hbfz225->SetMaximum(1.000);
  Cdepth1hbfz225->SetMinimum(0.00001);
  //      Cdepth1hbfz225->SetMinimum(0.0001);
  Cdepth1hbfz225->Draw("COLZ");

  c1->cd(2);

  //         TH1F *adepth1hb= (TH1F*)hfile1->Get("h_ADCAmpl_HB");
  //    TH1F *adepth1hb= (TH1F*)hfile1->Get("h_ADCAmplZoom1_HB");
  TH1F *adepth1hb = (TH1F *)hfile1->Get("h_ADCAmplZoom_HB");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth1hb->SetMarkerStyle(20);
  adepth1hb->SetMarkerSize(0.8);
  adepth1hb->GetYaxis()->SetLabelSize(0.04);
  adepth1hb->SetXTitle("ADCAmpl in each event & cell HB \b");
  adepth1hb->SetMarkerColor(2);
  adepth1hb->SetLineColor(2);
  adepth1hb->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth1hb = (TH2F *)Cdepth1hbff->Clone("Diffe_Depth1hb");
  for (int i = 1; i <= Cdepth1hbff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1hbff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth1hbff = (TH2F*)twoddepth1hb1->Clone("Cdepth1hbff");
      double ccc1 = Cdepth1hbff->GetBinContent(i, j);
      Diffe_Depth1hb->SetBinContent(i, j, 0.);
      //	if(ccc1>0.) cout<<" ibin=     "<< i <<" jbin=     "<< j <<" nevents=     "<< ccc1 <<endl;

      //	if(ccc1 <18|| ccc1>30.)  Diffe_Depth1hb->SetBinContent(i,j,ccc1);
      //	if(ccc1 <500.)  Diffe_Depth1hb->SetBinContent(i,j,ccc1);
      Diffe_Depth1hb->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1hb->SetMarkerStyle(20);
  Diffe_Depth1hb->SetMarkerSize(0.4);
  Diffe_Depth1hb->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1hb->SetTitle("any Error, HB Depth1hb \n");
  Diffe_Depth1hb->SetXTitle("#eta \b");
  Diffe_Depth1hb->SetYTitle("#phi \b");
  Diffe_Depth1hb->SetZTitle("<ADCAmpl> - HB Depth1 \b");
  Diffe_Depth1hb->SetMarkerColor(2);
  Diffe_Depth1hb->SetLineColor(2);
  //      Diffe_Depth1hb->SetMaximum(1.000);
  //      Diffe_Depth1hb->SetMinimum(0.0000001);
  Diffe_Depth1hb->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth1hb = new TH1F("diffADCAmpl_Depth1hb","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth1hb = new TH1F("diffADCAmpl_Depth1hb", "", 250, 0., 1000.);
  TH2F *Cdepth1hbfw = (TH2F *)Cdepth1hbff->Clone("Cdepth1hbfw");
  //    TH2F* Cdepth1hbfw = (TH2F*)twoddepth1hb1->Clone("diffADCAmpl_Depth1hb");
  for (int i = 1; i <= Cdepth1hbfw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1hbfw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth1hbfw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth1hbfw->GetBinContent(i, j);
        diffADCAmpl_Depth1hb->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth1hb->SetMarkerStyle(20);
  diffADCAmpl_Depth1hb->SetMarkerSize(0.4);
  diffADCAmpl_Depth1hb->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth1hb->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth1hb->SetMarkerColor(2);
  diffADCAmpl_Depth1hb->SetLineColor(2);
  diffADCAmpl_Depth1hb->Draw("");

  c1->Update();

  delete twoddepth1hb0;
  delete twoddepth1hb1;
  delete adepth1hb;
  delete Cdepth1hbfz225;
  delete Cdepth1hbff;
  delete diffADCAmpl_Depth1hb;
  delete Diffe_Depth1hb;

  //=============================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth2hb1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl_HB");
  TH2F *twoddepth2hb0 = (TH2F *)hfile1->Get("h_mapDepth2_HB");
  twoddepth2hb1->Sumw2();
  twoddepth2hb0->Sumw2();
  //    if(twoddepth2hb0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2hbff = (TH2F *)twoddepth2hb1->Clone("Cdepth2hbff");
  Cdepth2hbff->Divide(twoddepth2hb1, twoddepth2hb0, 1, 1, "B");
  Cdepth2hbff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth2hb1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl225_HB");
  TH2F *twoedepth2hb0 = (TH2F *)hfile1->Get("h_mapDepth2_HB");
  twoedepth2hb1->Sumw2();
  twoedepth2hb0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2hbfz225 = (TH2F *)twoedepth2hb1->Clone("Cdepth2hbfz225");
  Cdepth2hbfz225->Divide(twoedepth2hb1, twoedepth2hb0, 1, 1, "B");
  Cdepth2hbfz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth2hbfz225->SetMarkerStyle(20);
  Cdepth2hbfz225->SetMarkerSize(0.4);
  //    Cdepth2hbfz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth2hbfz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth2hbfz225->SetXTitle("#eta \b");
  Cdepth2hbfz225->SetYTitle("#phi \b");
  Cdepth2hbfz225->SetZTitle("Rate for ADCAmpl in each event & cell - HB Depth2 \b");
  Cdepth2hbfz225->SetMarkerColor(2);
  Cdepth2hbfz225->SetLineColor(2);
  Cdepth2hbfz225->SetMaximum(1.000);
  Cdepth2hbfz225->SetMinimum(0.00001);
  //      Cdepth2hbfz225->SetMinimum(0.0001);
  Cdepth2hbfz225->Draw("COLZ");

  c1->cd(2);

  //         TH1F *adepth2hb= (TH1F*)hfile1->Get("h_ADCAmpl_HB");
  TH1F *adepth2hb = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HB");
  //      TH1F *adepth2hb= (TH1F*)hfile1->Get("h_ADCAmplZoom_HB");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth2hb->SetMarkerStyle(20);
  adepth2hb->SetMarkerSize(0.8);
  adepth2hb->GetYaxis()->SetLabelSize(0.04);
  adepth2hb->SetXTitle("ADCAmpl in each event & cell HB \b");
  adepth2hb->SetMarkerColor(2);
  adepth2hb->SetLineColor(2);
  adepth2hb->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth2hb = (TH2F *)Cdepth2hbff->Clone("Diffe_Depth2hb");
  for (int i = 1; i <= Cdepth2hbff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2hbff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth2hbff = (TH2F*)twoddepth2hb1->Clone("Cdepth2hbff");
      double ccc1 = Cdepth2hbff->GetBinContent(i, j);
      Diffe_Depth2hb->SetBinContent(i, j, 0.);
      if (ccc1 < 500.)
        Diffe_Depth2hb->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2hb->SetMarkerStyle(20);
  Diffe_Depth2hb->SetMarkerSize(0.4);
  Diffe_Depth2hb->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2hb->SetTitle("any Error, HB Depth2hb \n");
  Diffe_Depth2hb->SetXTitle("#eta \b");
  Diffe_Depth2hb->SetYTitle("#phi \b");
  Diffe_Depth2hb->SetZTitle("<ADCAmpl> smalle 508 - HB Depth2hb \b");
  Diffe_Depth2hb->SetMarkerColor(2);
  Diffe_Depth2hb->SetLineColor(2);
  //      Diffe_Depth2hb->SetMaximum(1.000);
  //      Diffe_Depth2hb->SetMinimum(0.0000001);
  Diffe_Depth2hb->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth2hb = new TH1F("diffADCAmpl_Depth2hb","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth2hb = new TH1F("diffADCAmpl_Depth2hb", "", 250, 0., 1000.);
  TH2F *Cdepth2hbfw = (TH2F *)Cdepth2hbff->Clone("diffADCAmpl_Depth2hb");
  //      TH2F* Cdepth2hbfw = (TH2F*)twoddepth2hb1->Clone("diffADCAmpl_Depth2hb");
  for (int i = 1; i <= Cdepth2hbfw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2hbfw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth2hbfw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth2hbfw->GetBinContent(i, j);
        diffADCAmpl_Depth2hb->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth2hb->SetMarkerStyle(20);
  diffADCAmpl_Depth2hb->SetMarkerSize(0.4);
  diffADCAmpl_Depth2hb->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth2hb->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth2hb->SetMarkerColor(2);
  diffADCAmpl_Depth2hb->SetLineColor(2);
  diffADCAmpl_Depth2hb->Draw("");

  c1->Update();

  delete twoddepth2hb0;
  delete twoddepth2hb1;
  delete adepth2hb;
  delete Cdepth2hbfz225;
  delete Cdepth2hbff;
  delete diffADCAmpl_Depth2hb;
  delete Diffe_Depth2hb;

  //=============================================================================================== 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth1he1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl_HE");
  TH2F *twoddepth1he0 = (TH2F *)hfile1->Get("h_mapDepth1_HE");
  twoddepth1he1->Sumw2();
  twoddepth1he0->Sumw2();
  //    if(twoddepth1he0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1heff = (TH2F *)twoddepth1he1->Clone("Cdepth1heff");
  Cdepth1heff->Divide(twoddepth1he1, twoddepth1he0, 1, 1, "B");
  Cdepth1heff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth1he1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl225_HE");
  TH2F *twoedepth1he0 = (TH2F *)hfile1->Get("h_mapDepth1_HE");
  twoedepth1he1->Sumw2();
  twoedepth1he0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1hefz225 = (TH2F *)twoedepth1he1->Clone("Cdepth1hefz225");
  Cdepth1hefz225->Divide(twoedepth1he1, twoedepth1he0, 1, 1, "B");
  Cdepth1hefz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth1hefz225->SetMarkerStyle(20);
  Cdepth1hefz225->SetMarkerSize(0.4);
  //    Cdepth1hefz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth1hefz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth1hefz225->SetXTitle("#eta \b");
  Cdepth1hefz225->SetYTitle("#phi \b");
  Cdepth1hefz225->SetZTitle("Rate for ADCAmpl in each event & cell - HE Depth1 \b");
  Cdepth1hefz225->SetMarkerColor(2);
  Cdepth1hefz225->SetLineColor(2);
  Cdepth1hefz225->SetMaximum(1.000);
  Cdepth1hefz225->SetMinimum(0.00001);
  //      Cdepth1hefz225->SetMinimum(0.0001);
  Cdepth1hefz225->Draw("COLZ");

  c1->cd(2);

  //         TH1F *adepth1he= (TH1F*)hfile1->Get("h_ADCAmpl_HE");
  TH1F *adepth1he = (TH1F *)hfile1->Get("h_ADCAmpl345_HE");
  //      TH1F *adepth1he= (TH1F*)hfile1->Get("h_ADCAmplZoom_HE");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth1he->SetMarkerStyle(20);
  adepth1he->SetMarkerSize(0.8);
  adepth1he->GetYaxis()->SetLabelSize(0.04);
  adepth1he->SetXTitle("ADCAmpl in each event & cell HE \b");
  adepth1he->SetMarkerColor(2);
  adepth1he->SetLineColor(2);
  adepth1he->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth1he = (TH2F *)Cdepth1heff->Clone("Diffe_Depth1he");
  for (int i = 1; i <= Cdepth1heff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1heff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth1heff = (TH2F*)twoddepth1he1->Clone("Cdepth1heff");
      double ccc1 = Cdepth1heff->GetBinContent(i, j);
      //	if(ccc1>0.) cout<<"HE ibin=     "<< i <<" jbin=     "<< j <<" nevents=     "<< ccc1 <<endl;
      Diffe_Depth1he->SetBinContent(i, j, 0.);
      if (ccc1 < 20000000.)
        Diffe_Depth1he->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1he->SetMarkerStyle(20);
  Diffe_Depth1he->SetMarkerSize(0.4);
  Diffe_Depth1he->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1he->SetTitle("any Error, HE Depth1he \n");
  Diffe_Depth1he->SetXTitle("#eta \b");
  Diffe_Depth1he->SetYTitle("#phi \b");
  Diffe_Depth1he->SetZTitle("<ADCAmpl> smalle 20000008 - HE Depth1he \b");
  Diffe_Depth1he->SetMarkerColor(2);
  Diffe_Depth1he->SetLineColor(2);
  //      Diffe_Depth1he->SetMaximum(1.000);
  //      Diffe_Depth1he->SetMinimum(0.0000001);
  Diffe_Depth1he->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth1he = new TH1F("diffADCAmpl_Depth1he","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth1he = new TH1F("diffADCAmpl_Depth1he", "", 100, 0., 5000.);
  TH2F *Cdepth1hefw = (TH2F *)Cdepth1heff->Clone("diffADCAmpl_Depth1he");
  //      TH2F* Cdepth1hefw = (TH2F*)twoddepth1he1->Clone("diffADCAmpl_Depth1he");
  for (int i = 1; i <= Cdepth1hefw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1hefw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth1hefw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth1hefw->GetBinContent(i, j);
        diffADCAmpl_Depth1he->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth1he->SetMarkerStyle(20);
  diffADCAmpl_Depth1he->SetMarkerSize(0.4);
  diffADCAmpl_Depth1he->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth1he->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth1he->SetMarkerColor(2);
  diffADCAmpl_Depth1he->SetLineColor(2);
  diffADCAmpl_Depth1he->Draw("");

  c1->Update();

  delete twoddepth1he0;
  delete twoddepth1he1;
  delete adepth1he;
  delete Cdepth1hefz225;
  delete Cdepth1heff;
  delete diffADCAmpl_Depth1he;
  delete Diffe_Depth1he;

  //=============================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth2he1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl_HE");
  TH2F *twoddepth2he0 = (TH2F *)hfile1->Get("h_mapDepth2_HE");
  twoddepth2he1->Sumw2();
  twoddepth2he0->Sumw2();
  //    if(twoddepth2he0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2heff = (TH2F *)twoddepth2he1->Clone("Cdepth2heff");
  Cdepth2heff->Divide(twoddepth2he1, twoddepth2he0, 1, 1, "B");
  Cdepth2heff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth2he1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl225_HE");
  TH2F *twoedepth2he0 = (TH2F *)hfile1->Get("h_mapDepth2_HE");
  twoedepth2he1->Sumw2();
  twoedepth2he0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2hefz225 = (TH2F *)twoedepth2he1->Clone("Cdepth2hefz225");
  Cdepth2hefz225->Divide(twoedepth2he1, twoedepth2he0, 1, 1, "B");
  Cdepth2hefz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth2hefz225->SetMarkerStyle(20);
  Cdepth2hefz225->SetMarkerSize(0.4);
  //    Cdepth2hefz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth2hefz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth2hefz225->SetXTitle("#eta \b");
  Cdepth2hefz225->SetYTitle("#phi \b");
  Cdepth2hefz225->SetZTitle("Rate for ADCAmpl in each event & cell - HE Depth2 \b");
  Cdepth2hefz225->SetMarkerColor(2);
  Cdepth2hefz225->SetLineColor(2);
  Cdepth2hefz225->SetMaximum(1.000);
  Cdepth2hefz225->SetMinimum(0.00001);
  //      Cdepth2hefz225->SetMinimum(0.0001);
  Cdepth2hefz225->Draw("COLZ");

  c1->cd(2);

  TH1F *adepth2he = (TH1F *)hfile1->Get("h_ADCAmpl_HE");
  //     TH1F *adepth2he= (TH1F*)hfile1->Get("h_ADCAmplZoom1_HE");
  //      TH1F *adepth2he= (TH1F*)hfile1->Get("h_ADCAmplZoom_HE");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth2he->SetMarkerStyle(20);
  adepth2he->SetMarkerSize(0.8);
  adepth2he->GetYaxis()->SetLabelSize(0.04);
  adepth2he->SetXTitle("ADCAmpl in each event & cell HE \b");
  adepth2he->SetMarkerColor(2);
  adepth2he->SetLineColor(2);
  adepth2he->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth2he = (TH2F *)Cdepth2heff->Clone("Diffe_Depth2he");
  for (int i = 1; i <= Cdepth2heff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2heff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth2heff = (TH2F*)twoddepth2he1->Clone("Cdepth2heff");
      double ccc1 = Cdepth2heff->GetBinContent(i, j);
      Diffe_Depth2he->SetBinContent(i, j, 0.);
      if (ccc1 < 20000000.)
        Diffe_Depth2he->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2he->SetMarkerStyle(20);
  Diffe_Depth2he->SetMarkerSize(0.4);
  Diffe_Depth2he->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2he->SetTitle("any Error, HE Depth2he \n");
  Diffe_Depth2he->SetXTitle("#eta \b");
  Diffe_Depth2he->SetYTitle("#phi \b");
  Diffe_Depth2he->SetZTitle("<ADCAmpl> smalle 20000000 - HE Depth2he \b");
  Diffe_Depth2he->SetMarkerColor(2);
  Diffe_Depth2he->SetLineColor(2);
  //      Diffe_Depth2he->SetMaximum(1.000);
  //      Diffe_Depth2he->SetMinimum(0.0000001);
  Diffe_Depth2he->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth2he = new TH1F("diffADCAmpl_Depth2he","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth2he = new TH1F("diffADCAmpl_Depth2he", "", 100, 0., 5000.);
  TH2F *Cdepth2hefw = (TH2F *)Cdepth2heff->Clone("diffADCAmpl_Depth2he");
  //      TH2F* Cdepth2hefw = (TH2F*)twoddepth2he1->Clone("diffADCAmpl_Depth2he");
  for (int i = 1; i <= Cdepth2hefw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2hefw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth2hefw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth2hefw->GetBinContent(i, j);
        diffADCAmpl_Depth2he->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth2he->SetMarkerStyle(20);
  diffADCAmpl_Depth2he->SetMarkerSize(0.4);
  diffADCAmpl_Depth2he->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth2he->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth2he->SetMarkerColor(2);
  diffADCAmpl_Depth2he->SetLineColor(2);
  diffADCAmpl_Depth2he->Draw("");

  c1->Update();

  delete twoddepth2he0;
  delete twoddepth2he1;
  delete adepth2he;
  delete Cdepth2hefz225;
  delete Cdepth2heff;
  delete diffADCAmpl_Depth2he;
  delete Diffe_Depth2he;

  //=============================================================================================== 5
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth3he1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl_HE");
  TH2F *twoddepth3he0 = (TH2F *)hfile1->Get("h_mapDepth1_HE");
  twoddepth3he1->Sumw2();
  twoddepth3he0->Sumw2();
  //    if(twoddepth3he0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth3heff = (TH2F *)twoddepth3he1->Clone("Cdepth3heff");
  Cdepth3heff->Divide(twoddepth3he1, twoddepth3he0, 1, 1, "B");
  Cdepth3heff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth3he1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl225_HE");
  TH2F *twoedepth3he0 = (TH2F *)hfile1->Get("h_mapDepth1_HE");
  twoedepth3he1->Sumw2();
  twoedepth3he0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth3hefz225 = (TH2F *)twoedepth3he1->Clone("Cdepth3hefz225");
  Cdepth3hefz225->Divide(twoedepth3he1, twoedepth3he0, 1, 1, "B");
  Cdepth3hefz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth3hefz225->SetMarkerStyle(20);
  Cdepth3hefz225->SetMarkerSize(0.4);
  //    Cdepth3hefz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth3hefz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth3hefz225->SetXTitle("#eta \b");
  Cdepth3hefz225->SetYTitle("#phi \b");
  Cdepth3hefz225->SetZTitle("Rate for ADCAmpl in each event & cell - HE Depth1 \b");
  Cdepth3hefz225->SetMarkerColor(2);
  Cdepth3hefz225->SetLineColor(2);
  Cdepth3hefz225->SetMaximum(1.000);
  Cdepth3hefz225->SetMinimum(0.00001);
  //      Cdepth3hefz225->SetMinimum(0.0001);
  Cdepth3hefz225->Draw("COLZ");

  c1->cd(2);

  //         TH1F *adepth3he= (TH1F*)hfile1->Get("h_ADCAmpl_HE");
  TH1F *adepth3he = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HE");
  //      TH1F *adepth3he= (TH1F*)hfile1->Get("h_ADCAmplZoom_HE");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth3he->SetMarkerStyle(20);
  adepth3he->SetMarkerSize(0.8);
  adepth3he->GetYaxis()->SetLabelSize(0.04);
  adepth3he->SetXTitle("ADCAmpl in each event & cell HE \b");
  adepth3he->SetMarkerColor(2);
  adepth3he->SetLineColor(2);
  adepth3he->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth3he = (TH2F *)Cdepth3heff->Clone("Diffe_Depth3he");
  for (int i = 1; i <= Cdepth3heff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth3heff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth3heff = (TH2F*)twoddepth3he1->Clone("Cdepth3heff");
      double ccc1 = Cdepth3heff->GetBinContent(i, j);
      Diffe_Depth3he->SetBinContent(i, j, 0.);
      if (ccc1 < 20000000.)
        Diffe_Depth3he->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth3he->SetMarkerStyle(20);
  Diffe_Depth3he->SetMarkerSize(0.4);
  Diffe_Depth3he->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth3he->SetTitle("any Error, HE Depth3he \n");
  Diffe_Depth3he->SetXTitle("#eta \b");
  Diffe_Depth3he->SetYTitle("#phi \b");
  Diffe_Depth3he->SetZTitle("<ADCAmpl>smalle 20000000 - HE Depth3he \b");
  Diffe_Depth3he->SetMarkerColor(2);
  Diffe_Depth3he->SetLineColor(2);
  //      Diffe_Depth3he->SetMaximum(1.000);
  //      Diffe_Depth3he->SetMinimum(0.0000001);
  Diffe_Depth3he->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth3he = new TH1F("diffADCAmpl_Depth3he","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth3he = new TH1F("diffADCAmpl_Depth3he", "", 100, 0., 5000.);
  TH2F *Cdepth3hefw = (TH2F *)Cdepth3heff->Clone("diffADCAmpl_Depth3he");
  //      TH2F* Cdepth3hefw = (TH2F*)twoddepth3he1->Clone("diffADCAmpl_Depth3he");
  for (int i = 1; i <= Cdepth3hefw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth3hefw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth3hefw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth3hefw->GetBinContent(i, j);
        diffADCAmpl_Depth3he->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth3he->SetMarkerStyle(20);
  diffADCAmpl_Depth3he->SetMarkerSize(0.4);
  diffADCAmpl_Depth3he->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth3he->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth3he->SetMarkerColor(2);
  diffADCAmpl_Depth3he->SetLineColor(2);
  diffADCAmpl_Depth3he->Draw("");

  c1->Update();

  delete twoddepth3he0;
  delete twoddepth3he1;
  delete adepth3he;
  delete Cdepth3hefz225;
  delete Cdepth3heff;
  delete diffADCAmpl_Depth3he;
  delete Diffe_Depth3he;

  //=============================================================================================== 6
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth1hf1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl_HF");
  TH2F *twoddepth1hf0 = (TH2F *)hfile1->Get("h_mapDepth1_HF");
  twoddepth1hf1->Sumw2();
  twoddepth1hf0->Sumw2();
  //    if(twoddepth1hf0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1hfff = (TH2F *)twoddepth1hf1->Clone("Cdepth1hfff");
  Cdepth1hfff->Divide(twoddepth1hf1, twoddepth1hf0, 1, 1, "B");
  Cdepth1hfff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth1hf1 = (TH2F *)hfile1->Get("h_mapDepth1ADCAmpl225_HF");
  TH2F *twoedepth1hf0 = (TH2F *)hfile1->Get("h_mapDepth1_HF");
  twoedepth1hf1->Sumw2();
  twoedepth1hf0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth1hffz225 = (TH2F *)twoedepth1hf1->Clone("Cdepth1hffz225");
  Cdepth1hffz225->Divide(twoedepth1hf1, twoedepth1hf0, 1, 1, "B");
  Cdepth1hffz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth1hffz225->SetMarkerStyle(20);
  Cdepth1hffz225->SetMarkerSize(0.4);
  //    Cdepth1hffz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth1hffz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth1hffz225->SetXTitle("#eta \b");
  Cdepth1hffz225->SetYTitle("#phi \b");
  Cdepth1hffz225->SetZTitle("Rate for ADCAmpl in each event & cell - HF Depth1 \b");
  Cdepth1hffz225->SetMarkerColor(2);
  Cdepth1hffz225->SetLineColor(2);
  Cdepth1hffz225->SetMaximum(1.000);
  Cdepth1hffz225->SetMinimum(0.00001);
  //      Cdepth1hffz225->SetMinimum(0.0001);
  Cdepth1hffz225->Draw("COLZ");

  c1->cd(2);

  TH1F *adepth1hf = (TH1F *)hfile1->Get("h_ADCAmpl_HF");
  //      TH1F *adepth1hf= (TH1F*)hfile1->Get("h_ADCAmplZoom1_HF");
  //      TH1F *adepth1hf= (TH1F*)hfile1->Get("h_ADCAmplZoom_HF");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth1hf->SetMarkerStyle(20);
  adepth1hf->SetMarkerSize(0.8);
  adepth1hf->GetYaxis()->SetLabelSize(0.04);
  adepth1hf->SetXTitle("ADCAmpl in each event & cell HF \b");
  adepth1hf->SetMarkerColor(2);
  adepth1hf->SetLineColor(2);
  adepth1hf->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth1hf = (TH2F *)Cdepth1hfff->Clone("Diffe_Depth1hf");
  for (int i = 1; i <= Cdepth1hfff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1hfff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth1hfff = (TH2F*)twoddepth1hf1->Clone("Cdepth1hfff");
      double ccc1 = Cdepth1hfff->GetBinContent(i, j);
      Diffe_Depth1hf->SetBinContent(i, j, 0.);
      if (ccc1 < 20000000.)
        Diffe_Depth1hf->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1hf->SetMarkerStyle(20);
  Diffe_Depth1hf->SetMarkerSize(0.4);
  Diffe_Depth1hf->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1hf->SetTitle("any Error, HF Depth1hf \n");
  Diffe_Depth1hf->SetXTitle("#eta \b");
  Diffe_Depth1hf->SetYTitle("#phi \b");
  Diffe_Depth1hf->SetZTitle("<ADCAmpl> smalle 20000000 - HF Depth1hf \b");
  Diffe_Depth1hf->SetMarkerColor(2);
  Diffe_Depth1hf->SetLineColor(2);
  //      Diffe_Depth1hf->SetMaximum(1.000);
  //      Diffe_Depth1hf->SetMinimum(0.0000001);
  Diffe_Depth1hf->Draw("COLZ");

  c1->cd(4);
  TH1F *diffADCAmpl_Depth1hf = new TH1F("diffADCAmpl_Depth1hf", "", 100, 0., 300.);
  //      TH1F* diffADCAmpl_Depth1hf = new TH1F("diffADCAmpl_Depth1hf","", 1000, 0.,1000.);
  TH2F *Cdepth1hffw = (TH2F *)Cdepth1hfff->Clone("diffADCAmpl_Depth1hf");
  //      TH2F* Cdepth1hffw = (TH2F*)twoddepth1hf1->Clone("diffADCAmpl_Depth1hf");
  for (int i = 1; i <= Cdepth1hffw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth1hffw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth1hffw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth1hffw->GetBinContent(i, j);
        diffADCAmpl_Depth1hf->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth1hf->SetMarkerStyle(20);
  diffADCAmpl_Depth1hf->SetMarkerSize(0.4);
  diffADCAmpl_Depth1hf->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth1hf->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth1hf->SetMarkerColor(2);
  diffADCAmpl_Depth1hf->SetLineColor(2);
  diffADCAmpl_Depth1hf->Draw("");

  c1->Update();

  delete twoddepth1hf0;
  delete twoddepth1hf1;
  delete adepth1hf;
  delete Cdepth1hffz225;
  delete Cdepth1hfff;
  delete diffADCAmpl_Depth1hf;
  delete Diffe_Depth1hf;

  //=============================================================================================== 7
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth2hf1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl_HF");
  TH2F *twoddepth2hf0 = (TH2F *)hfile1->Get("h_mapDepth2_HF");
  twoddepth2hf1->Sumw2();
  twoddepth2hf0->Sumw2();
  //    if(twoddepth2hf0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2hfff = (TH2F *)twoddepth2hf1->Clone("Cdepth2hfff");
  Cdepth2hfff->Divide(twoddepth2hf1, twoddepth2hf0, 1, 1, "B");
  Cdepth2hfff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth2hf1 = (TH2F *)hfile1->Get("h_mapDepth2ADCAmpl225_HF");
  TH2F *twoedepth2hf0 = (TH2F *)hfile1->Get("h_mapDepth2_HF");
  twoedepth2hf1->Sumw2();
  twoedepth2hf0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth2hffz225 = (TH2F *)twoedepth2hf1->Clone("Cdepth2hffz225");
  Cdepth2hffz225->Divide(twoedepth2hf1, twoedepth2hf0, 1, 1, "B");
  Cdepth2hffz225->Sumw2();
  //    }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth2hffz225->SetMarkerStyle(20);
  Cdepth2hffz225->SetMarkerSize(0.4);
  //    Cdepth2hffz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth2hffz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth2hffz225->SetXTitle("#eta \b");
  Cdepth2hffz225->SetYTitle("#phi \b");
  Cdepth2hffz225->SetZTitle("Rate for ADCAmpl in each event & cell - HF Depth2 \b");
  Cdepth2hffz225->SetMarkerColor(2);
  Cdepth2hffz225->SetLineColor(2);
  Cdepth2hffz225->SetMaximum(1.000);
  Cdepth2hffz225->SetMinimum(0.00001);
  //      Cdepth2hffz225->SetMinimum(0.0001);
  Cdepth2hffz225->Draw("COLZ");

  c1->cd(2);

  //         TH1F *adepth2hf= (TH1F*)hfile1->Get("h_ADCAmpl_HF");
  TH1F *adepth2hf = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HF");
  //      TH1F *adepth2hf= (TH1F*)hfile1->Get("h_ADCAmplZoom_HF");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth2hf->SetMarkerStyle(20);
  adepth2hf->SetMarkerSize(0.8);
  adepth2hf->GetYaxis()->SetLabelSize(0.04);
  adepth2hf->SetXTitle("ADCAmpl in each event & cell HF \b");
  adepth2hf->SetMarkerColor(2);
  adepth2hf->SetLineColor(2);
  adepth2hf->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth2hf = (TH2F *)Cdepth2hfff->Clone("Diffe_Depth2hf");
  for (int i = 1; i <= Cdepth2hfff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2hfff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth2hfff = (TH2F*)twoddepth2hf1->Clone("Cdepth2hfff");
      double ccc1 = Cdepth2hfff->GetBinContent(i, j);
      Diffe_Depth2hf->SetBinContent(i, j, 0.);
      if (ccc1 < 20000000.)
        Diffe_Depth2hf->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2hf->SetMarkerStyle(20);
  Diffe_Depth2hf->SetMarkerSize(0.4);
  Diffe_Depth2hf->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2hf->SetTitle("any Error, HF Depth2hf \n");
  Diffe_Depth2hf->SetXTitle("#eta \b");
  Diffe_Depth2hf->SetYTitle("#phi \b");
  Diffe_Depth2hf->SetZTitle("<ADCAmpl> smalle 20000000 - HF Depth2hf \b");
  Diffe_Depth2hf->SetMarkerColor(2);
  Diffe_Depth2hf->SetLineColor(2);
  //      Diffe_Depth2hf->SetMaximum(1.000);
  //      Diffe_Depth2hf->SetMinimum(0.0000001);
  Diffe_Depth2hf->Draw("COLZ");

  c1->cd(4);
  TH1F *diffADCAmpl_Depth2hf = new TH1F("diffADCAmpl_Depth2hf", "", 100, 0., 300.);
  //      TH1F* diffADCAmpl_Depth2hf = new TH1F("diffADCAmpl_Depth2hf","", 1000, 0.,1000.);
  TH2F *Cdepth2hffw = (TH2F *)Cdepth2hfff->Clone("diffADCAmpl_Depth2hf");
  //      TH2F* Cdepth2hffw = (TH2F*)twoddepth2hf1->Clone("diffADCAmpl_Depth2hf");
  for (int i = 1; i <= Cdepth2hffw->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth2hffw->GetYaxis()->GetNbins(); j++) {
      if (Cdepth2hffw->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth2hffw->GetBinContent(i, j);
        diffADCAmpl_Depth2hf->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth2hf->SetMarkerStyle(20);
  diffADCAmpl_Depth2hf->SetMarkerSize(0.4);
  diffADCAmpl_Depth2hf->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth2hf->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth2hf->SetMarkerColor(2);
  diffADCAmpl_Depth2hf->SetLineColor(2);
  diffADCAmpl_Depth2hf->Draw("");

  c1->Update();

  delete twoddepth2hf0;
  delete twoddepth2hf1;
  delete adepth2hf;
  delete Cdepth2hffz225;
  delete Cdepth2hfff;
  delete diffADCAmpl_Depth2hf;
  delete Diffe_Depth2hf;

  //=============================================================================================== 8
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twoddepth4ho1 = (TH2F *)hfile1->Get("h_mapDepth4ADCAmpl_HO");
  TH2F *twoddepth4ho0 = (TH2F *)hfile1->Get("h_mapDepth4_HO");
  twoddepth4ho1->Sumw2();
  twoddepth4ho0->Sumw2();
  //    if(twoddepth4ho0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth4hoff = (TH2F *)twoddepth4ho1->Clone("Cdepth4hoff");
  Cdepth4hoff->Divide(twoddepth4ho1, twoddepth4ho0, 1, 1, "B");
  Cdepth4hoff->Sumw2();
  //    }
  c1->cd(1);
  TH2F *twoedepth4ho1 = (TH2F *)hfile1->Get("h_mapDepth4ADCAmpl225_HO");
  TH2F *twoedepth4ho0 = (TH2F *)hfile1->Get("h_mapDepth4_HO");
  twoedepth4ho1->Sumw2();
  twoedepth4ho0->Sumw2();
  //    if(twoe0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cdepth4hofz225 = (TH2F *)twoedepth4ho1->Clone("Cdepth4hofz225");
  Cdepth4hofz225->Divide(twoedepth4ho1, twoedepth4ho0, 1, 1, "B");
  Cdepth4hofz225->Sumw2();
  //    }
  for (int i = 1; i <= Cdepth4hofz225->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth4hofz225->GetYaxis()->GetNbins(); j++) {
      double ccc1 = Cdepth4hofz225->GetBinContent(i, j);
      //	if(ccc1> 0.1) cout<<"HO ibin=     "<< i <<" jbin=     "<< j <<" Rate=     "<< ccc1 <<endl;
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cdepth4hofz225->SetMarkerStyle(20);
  Cdepth4hofz225->SetMarkerSize(0.4);
  //    Cdepth4hofz225->GetYaxis()->SetLabelSize(0.04);
  Cdepth4hofz225->GetZaxis()->SetLabelSize(0.08);
  Cdepth4hofz225->SetXTitle("#eta \b");
  Cdepth4hofz225->SetYTitle("#phi \b");
  Cdepth4hofz225->SetZTitle("Rate for ADCAmpl in each event & cell - HO Depth4 \b");
  Cdepth4hofz225->SetMarkerColor(2);
  Cdepth4hofz225->SetLineColor(2);
  Cdepth4hofz225->SetMaximum(1.000);
  Cdepth4hofz225->SetMinimum(0.1);
  //      Cdepth4hofz225->SetMinimum(0.0001);
  Cdepth4hofz225->Draw("COLZ");

  c1->cd(2);

  TH1F *adepth4ho = (TH1F *)hfile1->Get("h_ADCAmpl_HO");
  //     TH1F *adepth4ho= (TH1F*)hfile1->Get("h_ADCAmplZoom1_HO");
  //      TH1F *adepth4ho= (TH1F*)hfile1->Get("h_ADCAmplZoom_HO");
  gPad->SetLogy();
  //      gPad->SetLogx();
  adepth4ho->SetMarkerStyle(20);
  adepth4ho->SetMarkerSize(0.8);
  adepth4ho->GetYaxis()->SetLabelSize(0.04);
  adepth4ho->SetXTitle("ADCAmpl in each event & cell HO \b");
  adepth4ho->SetMarkerColor(2);
  adepth4ho->SetLineColor(2);
  adepth4ho->Draw("");

  c1->cd(3);

  ///////////////////////////////////////
  TH2F *Diffe_Depth4ho = (TH2F *)Cdepth4hoff->Clone("Diffe_Depth4ho");
  for (int i = 1; i <= Cdepth4hoff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth4hoff->GetYaxis()->GetNbins(); j++) {
      //	TH2F* Cdepth4hoff = (TH2F*)twoddepth4ho1->Clone("Cdepth4hoff");
      double ccc1 = Cdepth4hoff->GetBinContent(i, j);
      Diffe_Depth4ho->SetBinContent(i, j, 0.);
      //	if(ccc1> 1000.|| (ccc1>0.&&ccc1<27.)) cout<<"HO ibin=     "<< i <<" jbin=     "<< j <<" A=     "<< ccc1 <<endl;
      //	if(ccc1> 1000.|| (i==46&&j==5)|| (i==56&&j==13)) cout<<"HO ibin=     "<< i <<" jbin=     "<< j <<" A=     "<< ccc1 <<endl;
      //	if(ccc1 < 20000000.)  Diffe_Depth4ho->SetBinContent(i,j,ccc1);
      if (ccc1 > 160. || ccc1 < 250.)
        Diffe_Depth4ho->SetBinContent(i, j, ccc1);
      //		if(ccc1 > 250.|| ccc1<400.)  Diffe_Depth4ho->SetBinContent(i,j,ccc1);
      //		if(ccc1 > 500.)  Diffe_Depth4ho->SetBinContent(i,j,ccc1);
      //	Diffe_Depth4ho->SetBinContent(i,j,ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth4ho->SetMarkerStyle(20);
  Diffe_Depth4ho->SetMarkerSize(0.4);
  Diffe_Depth4ho->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth4ho->SetTitle("any Error, HO Depth4ho \n");
  Diffe_Depth4ho->SetXTitle("#eta \b");
  Diffe_Depth4ho->SetYTitle("#phi \b");
  Diffe_Depth4ho->SetZTitle("<ADCAmpl> smalle 20000000 - HO Depth4ho \b");
  Diffe_Depth4ho->SetMarkerColor(2);
  Diffe_Depth4ho->SetLineColor(2);
  //      Diffe_Depth4ho->SetMaximum(1.000);
  //      Diffe_Depth4ho->SetMinimum(0.0000001);
  Diffe_Depth4ho->Draw("COLZ");

  c1->cd(4);
  //    TH1F* diffADCAmpl_Depth4ho = new TH1F("diffADCAmpl_Depth4ho","", 100, 0.,3000.);
  TH1F *diffADCAmpl_Depth4ho = new TH1F("diffADCAmpl_Depth4ho", "", 250, 0., 1000.);

  //      TH2F* Cdepth4hofw = (TH2F*)Cdepth4hoff->Clone("diffADCAmpl_Depth4ho");
  //      TH2F* Cdepth4hofw = (TH2F*)twoddepth4ho1->Clone("diffADCAmpl_Depth4ho");

  for (int i = 1; i <= Cdepth4hoff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Cdepth4hoff->GetYaxis()->GetNbins(); j++) {
      if (Cdepth4hoff->GetBinContent(i, j) != 0) {
        double ccc1 = Cdepth4hoff->GetBinContent(i, j);
        diffADCAmpl_Depth4ho->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffADCAmpl_Depth4ho->SetMarkerStyle(20);
  diffADCAmpl_Depth4ho->SetMarkerSize(0.4);
  diffADCAmpl_Depth4ho->GetYaxis()->SetLabelSize(0.04);
  diffADCAmpl_Depth4ho->SetXTitle("<ADCAmpl> in each cell \b");
  diffADCAmpl_Depth4ho->SetMarkerColor(2);
  diffADCAmpl_Depth4ho->SetLineColor(2);
  diffADCAmpl_Depth4ho->SetMinimum(0.8);
  diffADCAmpl_Depth4ho->Draw("");

  c1->Update();

  delete twoddepth4ho0;
  delete twoddepth4ho1;
  delete adepth4ho;
  delete Cdepth4hofz225;
  delete Cdepth4hoff;
  delete diffADCAmpl_Depth4ho;
  delete Diffe_Depth4ho;

  //========================================================================================== 9
  //======================================================================
  //======================================================================
  //================
  /*
    // fullAmplitude:
///////////////////////////////////////////////////////////////////////////////////////
    h_ADCAmpl345Zoom_HB = new TH1F("h_ADCAmpl345Zoom_HB"," ", 100, 0.,400.);
    h_ADCAmpl345Zoom1_HB = new TH1F("h_ADCAmpl345Zoom1_HB"," ", 100, 0.,100.);
    h_ADCAmpl345_HB = new TH1F("h_ADCAmpl345_HB"," ", 100, 10.,3000.);

    h_ADCAmplZoom_HB = new TH1F("h_ADCAmplZoom_HB"," ", 100, 0.,400.);
    h_ADCAmplZoom1_HB = new TH1F("h_ADCAmplZoom1_HB"," ", 100, -20.,80.);
    h_ADCAmpl_HB = new TH1F("h_ADCAmpl_HB"," ", 100, 10.,5000.);

    h_AmplitudeHBrest = new TH1F("h_AmplitudeHBrest"," ", 100, 0.,10000.);
    h_AmplitudeHBrest1 = new TH1F("h_AmplitudeHBrest1"," ", 100, 0.,1000000.);
    h_AmplitudeHBrest6 = new TH1F("h_AmplitudeHBrest6"," ", 100, 0.,2000000.);
*/
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  c1->cd(1);
  TH1F *aaaaaa0 = (TH1F *)hfile1->Get("h_ADCAmplZoom_HB");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaaa0->SetMarkerStyle(20);
  aaaaaa0->SetMarkerSize(0.8);
  aaaaaa0->GetYaxis()->SetLabelSize(0.04);
  aaaaaa0->SetXTitle("h_ADCAmplZoom_HB \b");
  aaaaaa0->SetMarkerColor(2);
  aaaaaa0->SetLineColor(2);
  aaaaaa0->Draw("");

  c1->cd(2);
  TH1F *aaaaaa3 = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HB");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaaa3->SetMarkerStyle(20);
  aaaaaa3->SetMarkerSize(0.8);
  aaaaaa3->GetYaxis()->SetLabelSize(0.04);
  aaaaaa3->SetXTitle("ADCAmpl in each event & cell HB \b");
  aaaaaa3->SetMarkerColor(2);
  aaaaaa3->SetLineColor(2);
  aaaaaa3->Draw("");

  c1->cd(3);
  TH1F *aaaaaa5 = (TH1F *)hfile1->Get("h_ADCAmpl_HB");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaaa5->SetMarkerStyle(20);
  aaaaaa5->SetMarkerSize(0.8);
  aaaaaa5->GetYaxis()->SetLabelSize(0.04);
  aaaaaa5->SetXTitle("ADCAmpl in each event & cell HB \b");
  aaaaaa5->SetMarkerColor(2);
  aaaaaa5->SetLineColor(2);
  aaaaaa5->Draw("");

  c1->cd(4);
  TH1F *aaaaaa1 = (TH1F *)hfile1->Get("h_AmplitudeHBrest");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaaa1->SetMarkerStyle(20);
  aaaaaa1->SetMarkerSize(0.8);
  aaaaaa1->GetYaxis()->SetLabelSize(0.04);
  aaaaaa1->SetXTitle("h_AmplitudeHBrest \b");
  aaaaaa1->SetMarkerColor(2);
  aaaaaa1->SetLineColor(2);
  aaaaaa1->Draw("");

  c1->cd(5);
  TH1F *aaaaaa2 = (TH1F *)hfile1->Get("h_AmplitudeHBrest1");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaaa2->SetMarkerStyle(20);
  aaaaaa2->SetMarkerSize(0.8);
  aaaaaa2->GetYaxis()->SetLabelSize(0.04);
  aaaaaa2->SetXTitle("h_AmplitudeHBrest1 \b");
  aaaaaa2->SetMarkerColor(2);
  aaaaaa2->SetLineColor(2);
  aaaaaa2->Draw("");

  c1->cd(6);
  TH1F *aaaaaa4 = (TH1F *)hfile1->Get("h_AmplitudeHBrest6");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaaa4->SetMarkerStyle(20);
  aaaaaa4->SetMarkerSize(0.8);
  aaaaaa4->GetYaxis()->SetLabelSize(0.04);
  aaaaaa4->SetXTitle("h_AmplitudeHBrest6 \b");
  aaaaaa4->SetMarkerColor(2);
  aaaaaa4->SetLineColor(2);
  aaaaaa4->Draw("");

  c1->Update();

  //========================================================================================== 10
  //======================================================================
  //======================================================================
  //================
  /*
    // fullAmplitude:
///////////////////////////////////////////////////////////////////////////////////////

    h_ADCAmpl_HE = new TH1F("h_ADCAmpl_HE"," ", 200, 0.,2000000.);
    h_ADCAmpl345_HE = new TH1F("h_ADCAmpl345_HE"," ", 70, 0.,700000.);

// for SiPM:
const int npfit = 220; float anpfit = 220.;
    h_ADCAmplZoom1_HE = new TH1F("h_ADCAmplZoom1_HE"," ",npfit, 0.,anpfit);// for amplmaxts 1TS w/ max
    h_ADCAmpl345Zoom1_HE = new TH1F("h_ADCAmpl345Zoom1_HE"," ", npfit, 0.,anpfit);// for ampl3ts 3TSs
    h_ADCAmpl345Zoom_HE = new TH1F("h_ADCAmpl345Zoom_HE"," ", npfit, 0.,anpfit); // for ampl 4TSs
*/
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  c1->cd(1);
  TH1F *aaaaab0 = (TH1F *)hfile1->Get("h_ADCAmpl345_HE");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaab0->SetMarkerStyle(20);
  aaaaab0->SetMarkerSize(0.8);
  aaaaab0->GetYaxis()->SetLabelSize(0.04);
  aaaaab0->SetXTitle("h_ADCAmpl345_HE \b");
  aaaaab0->SetMarkerColor(2);
  aaaaab0->SetLineColor(2);
  aaaaab0->Draw("");

  c1->cd(2);
  TH1F *aaaaab3 = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HE");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaab3->SetMarkerStyle(20);
  aaaaab3->SetMarkerSize(0.8);
  aaaaab3->GetYaxis()->SetLabelSize(0.04);
  aaaaab3->SetXTitle("for amplmaxts 1TS w/ max HE \b");
  aaaaab3->SetMarkerColor(2);
  aaaaab3->SetLineColor(2);
  aaaaab3->Draw("");
  TH1F *aaaaab4 = (TH1F *)hfile1->Get("h_ADCAmpl345Zoom1_HE");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaab4->SetMarkerStyle(20);
  aaaaab4->SetMarkerSize(0.8);
  aaaaab4->GetYaxis()->SetLabelSize(0.04);
  aaaaab4->SetMarkerColor(4);
  aaaaab4->SetLineColor(4);
  aaaaab4->Draw("Same");

  c1->cd(3);
  TH1F *aaaaab5 = (TH1F *)hfile1->Get("h_ADCAmpl_HE");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaab5->SetMarkerStyle(20);
  aaaaab5->SetMarkerSize(0.8);
  aaaaab5->GetYaxis()->SetLabelSize(0.04);
  aaaaab5->SetXTitle("ADCAmpl in each event & cell HE \b");
  aaaaab5->SetMarkerColor(2);
  aaaaab5->SetLineColor(2);
  aaaaab5->Draw("");

  c1->cd(4);
  TH1F *aaaaab1 = (TH1F *)hfile1->Get("h_ADCAmplrest_HE");
  //    TH1F *aaaaab1= (TH1F*)hfile1->Get("h_AmplitudeHEtest");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaab1->SetMarkerStyle(20);
  aaaaab1->SetMarkerSize(0.8);
  aaaaab1->GetYaxis()->SetLabelSize(0.04);
  aaaaab1->SetXTitle("h_ADCAmplrest_HE \b");
  aaaaab1->SetMarkerColor(2);
  aaaaab1->SetLineColor(2);
  aaaaab1->Draw("");

  c1->cd(5);
  TH1F *aaaaab2 = (TH1F *)hfile1->Get("h_ADCAmplrest1_HE");
  //    TH1F *aaaaab2= (TH1F*)hfile1->Get("h_AmplitudeHEtest1");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaab2->SetMarkerStyle(20);
  aaaaab2->SetMarkerSize(0.8);
  aaaaab2->GetYaxis()->SetLabelSize(0.04);
  aaaaab2->SetXTitle("h_ADCAmplrest1_HE \b");
  aaaaab2->SetMarkerColor(2);
  aaaaab2->SetLineColor(2);
  aaaaab2->Draw("");

  c1->cd(6);
  TH1F *aaaaab6 = (TH1F *)hfile1->Get("h_ADCAmplrest6_HE");
  //    TH1F *aaaaab6= (TH1F*)hfile1->Get("h_AmplitudeHEtest6");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaab6->SetMarkerStyle(20);
  aaaaab6->SetMarkerSize(0.8);
  aaaaab6->GetYaxis()->SetLabelSize(0.04);
  aaaaab6->SetXTitle("h_ADCAmplrest6_HE \b");
  aaaaab6->SetMarkerColor(2);
  aaaaab6->SetLineColor(2);
  aaaaab6->Draw("");

  c1->Update();

  //========================================================================================== 11
  //======================================================================
  //======================================================================
  //================
  /*
    // fullAmplitude:
///////////////////////////////////////////////////////////////////////////////////////
    h_ADCAmplZoom1_HF = new TH1F("h_ADCAmplZoom1_HF"," ", 100, 0.,1000000.);
    h_ADCAmpl_HF = new TH1F("h_ADCAmpl_HF"," ", 250, 0.,500000.);

*/
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH1F *aaaaac0 = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HF");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaac0->SetMarkerStyle(20);
  aaaaac0->SetMarkerSize(0.8);
  aaaaac0->GetYaxis()->SetLabelSize(0.04);
  aaaaac0->SetXTitle("h_ADCAmplZoom1_HF \b");
  aaaaac0->SetMarkerColor(2);
  aaaaac0->SetLineColor(2);
  aaaaac0->Draw("");

  c1->cd(2);
  TH1F *aaaaac3 = (TH1F *)hfile1->Get("h_ADCAmpl_HF");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaac3->SetMarkerStyle(20);
  aaaaac3->SetMarkerSize(0.8);
  aaaaac3->GetYaxis()->SetLabelSize(0.04);
  aaaaac3->SetXTitle("h_ADCAmpl_HF \b");
  aaaaac3->SetMarkerColor(2);
  aaaaac3->SetLineColor(2);
  aaaaac3->Draw("");

  c1->cd(3);
  TH1F *aaaaac5 = (TH1F *)hfile1->Get("h_ADCAmplrest1_HF");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaac5->SetMarkerStyle(20);
  aaaaac5->SetMarkerSize(0.8);
  aaaaac5->GetYaxis()->SetLabelSize(0.04);
  aaaaac5->SetXTitle("h_ADCAmplrest1_HF  \b");
  aaaaac5->SetMarkerColor(2);
  aaaaac5->SetLineColor(2);
  aaaaac5->Draw("");

  c1->cd(4);
  TH1F *aaaaac1 = (TH1F *)hfile1->Get("h_ADCAmplrest6_HF");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaac1->SetMarkerStyle(20);
  aaaaac1->SetMarkerSize(0.8);
  aaaaac1->GetYaxis()->SetLabelSize(0.04);
  aaaaac1->SetXTitle("h_ADCAmplrest6_HF \b");
  aaaaac1->SetMarkerColor(2);
  aaaaac1->SetLineColor(2);
  aaaaac1->Draw("");

  c1->Update();

  //========================================================================================== 12
  //======================================================================
  //======================================================================
  //================
  /*
    // fullAmplitude:
///////////////////////////////////////////////////////////////////////////////////////
    h_ADCAmpl_HO = new TH1F("h_ADCAmpl_HO"," ", 100, 0.,7000.);
    h_ADCAmplZoom1_HO = new TH1F("h_ADCAmplZoom1_HO"," ", 100, -20.,280.);
    h_ADCAmpl_HO_copy = new TH1F("h_ADCAmpl_HO_copy"," ", 100, 0.,30000.);
*/
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  c1->cd(1);
  TH1F *aaaaad0 = (TH1F *)hfile1->Get("h_ADCAmplrest1_HO");
  gPad->SetLogy();
  // gPad->SetLogx();
  aaaaad0->SetMarkerStyle(20);
  aaaaad0->SetMarkerSize(0.8);
  aaaaad0->GetYaxis()->SetLabelSize(0.04);
  aaaaad0->SetXTitle("h_ADCAmplrest1_HO \b");
  aaaaad0->SetMarkerColor(2);
  aaaaad0->SetLineColor(2);
  aaaaad0->Draw("");

  c1->cd(2);
  TH1F *aaaaad3 = (TH1F *)hfile1->Get("h_ADCAmplrest6_HO");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaad3->SetMarkerStyle(20);
  aaaaad3->SetMarkerSize(0.8);
  aaaaad3->GetYaxis()->SetLabelSize(0.04);
  aaaaad3->SetXTitle("h_ADCAmplrest6_HO \b");
  aaaaad3->SetMarkerColor(2);
  aaaaad3->SetLineColor(2);
  aaaaad3->Draw("");

  c1->cd(3);
  TH1F *aaaaad5 = (TH1F *)hfile1->Get("h_ADCAmpl_HO");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaad5->SetMarkerStyle(20);
  aaaaad5->SetMarkerSize(0.8);
  aaaaad5->GetYaxis()->SetLabelSize(0.04);
  aaaaad5->SetXTitle("h_ADCAmpl_HO \b");
  aaaaad5->SetMarkerColor(2);
  aaaaad5->SetLineColor(2);
  aaaaad5->Draw("");

  c1->cd(4);
  TH1F *aaaaad1 = (TH1F *)hfile1->Get("h_ADCAmplZoom1_HO");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaad1->SetMarkerStyle(20);
  aaaaad1->SetMarkerSize(0.8);
  aaaaad1->GetYaxis()->SetLabelSize(0.04);
  aaaaad1->SetXTitle("h_ADCAmplZoom1_HO \b");
  aaaaad1->SetMarkerColor(2);
  aaaaad1->SetLineColor(2);
  aaaaad1->Draw("");

  c1->cd(5);
  TH1F *aaaaad2 = (TH1F *)hfile1->Get("h_ADCAmpl_HO_copy");
  gPad->SetLogy();
  //    gPad->SetLogx();
  aaaaad2->SetMarkerStyle(20);
  aaaaad2->SetMarkerSize(0.8);
  aaaaad2->GetYaxis()->SetLabelSize(0.04);
  aaaaad2->SetXTitle("h_ADCAmpl_HO_copy \b");
  aaaaad2->SetMarkerColor(2);
  aaaaad2->SetLineColor(2);
  aaaaad2->Draw("");

  c1->Update();

  //========================================================================================== 13
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH1F *time = (TH1F *)hfile1->Get("h_tdc_HE_time");
  TH1F *time0 = (TH1F *)hfile1->Get("h_tdc_HE_time0");
  TH1F *timer = (TH1F *)time->Clone("timer");
  time->Sumw2();
  time0->Sumw2();
  timer->Divide(time, time0, 1, 1, "B");  //
  timer->Sumw2();
  TH1F *timet = (TH1F *)hfile1->Get("h_tdc_HE_tdc");
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  time->SetMarkerStyle(20);
  time->SetMarkerSize(0.4);
  time->GetYaxis()->SetLabelSize(0.04);
  time->SetXTitle("time50 weighted by A  HE \b");
  time->SetMarkerColor(2);
  time->SetLineColor(2);
  time->Draw("");
  c1->cd(2);
  time0->SetMarkerStyle(20);
  time0->SetMarkerSize(0.4);
  time0->GetYaxis()->SetLabelSize(0.04);
  time0->SetXTitle("time50 HE \b");
  time0->SetMarkerColor(2);
  time0->SetLineColor(2);
  time0->Draw("");
  c1->cd(3);
  timer->SetMarkerStyle(20);
  timer->SetMarkerSize(0.4);
  timer->GetYaxis()->SetLabelSize(0.04);
  timer->SetXTitle("shape HE \b");
  timer->SetMarkerColor(2);
  timer->SetLineColor(0);
  timer->SetMinimum(0.);
  timer->SetMaximum(50000.);
  timer->Draw("");
  c1->cd(4);
  gPad->SetLogy();
  timet->SetMarkerStyle(20);
  timet->SetMarkerSize(0.4);
  timet->GetYaxis()->SetLabelSize(0.04);
  timet->SetXTitle("initial tdc HE \b");
  timet->SetMarkerColor(2);
  timet->SetLineColor(2);
  timet->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 14
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  TH1F *ampldefault = (TH1F *)hfile1->Get("h_tdc_HE_ampldefault");
  TH1F *ampldefault50 = (TH1F *)hfile1->Get("h_tdc_HE_ampldefault50");
  TH1F *ampldefault63 = (TH1F *)hfile1->Get("h_tdc_HE_ampldefault63");
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefault->SetMarkerStyle(20);
  ampldefault->SetMarkerSize(0.4);
  ampldefault->GetYaxis()->SetLabelSize(0.04);
  ampldefault->SetXTitle("A_TS HE \b");
  ampldefault->SetMarkerColor(2);
  ampldefault->SetLineColor(2);
  gPad->SetLogy();
  ampldefault->Draw("");

  c1->cd(2);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefault50->SetMarkerStyle(20);
  ampldefault50->SetMarkerSize(0.4);
  ampldefault50->GetYaxis()->SetLabelSize(0.04);
  ampldefault50->SetXTitle("A_TS_50 HE \b");
  ampldefault50->SetMarkerColor(2);
  ampldefault50->SetLineColor(2);
  gPad->SetLogy();
  ampldefault50->Draw("");

  c1->cd(3);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefault63->SetMarkerStyle(20);
  ampldefault63->SetMarkerSize(0.4);
  ampldefault63->GetYaxis()->SetLabelSize(0.04);
  ampldefault63->SetXTitle("A_TS_63 HE \b");
  ampldefault63->SetMarkerColor(2);
  ampldefault63->SetLineColor(2);
  gPad->SetLogy();
  ampldefault63->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 15
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 1);

  TH1F *timeVSampldefault = (TH1F *)hfile1->Get("h_tdc_HE_timeVSampldefault");
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  timeVSampldefault->SetMarkerStyle(20);
  timeVSampldefault->SetMarkerSize(0.4);
  timeVSampldefault->GetYaxis()->SetLabelSize(0.04);
  timeVSampldefault->SetXTitle("timeVStampldefault HE \b");
  timeVSampldefault->SetMarkerColor(2);
  timeVSampldefault->SetLineColor(2);
  timeVSampldefault->Draw("box");

  c1->Update();
  //========================================================================================== 16
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  TH1F *shape = (TH1F *)hfile1->Get("h_shape_good_channels_HE");
  TH1F *shape0 = (TH1F *)hfile1->Get("h_shape0_good_channels_HE");
  TH1F *shaper = (TH1F *)shape->Clone("shaper");
  shape->Sumw2();
  shape0->Sumw2();
  shaper->Divide(shape, shape0, 1, 1, "B");  //
  shaper->Sumw2();
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  shape->SetMarkerStyle(20);
  shape->SetMarkerSize(0.4);
  shape->GetYaxis()->SetLabelSize(0.04);
  shape->SetXTitle("TS weighted by A good HE \b");
  shape->SetMarkerColor(2);
  shape->SetLineColor(2);
  shape->Draw("");
  c1->cd(3);
  shape0->SetMarkerStyle(20);
  shape0->SetMarkerSize(0.4);
  shape0->GetYaxis()->SetLabelSize(0.04);
  shape0->SetXTitle("TS good HE \b");
  shape0->SetMarkerColor(2);
  shape0->SetLineColor(2);
  shape0->Draw("");
  c1->cd(5);
  shaper->SetMarkerStyle(20);
  shaper->SetMarkerSize(0.4);
  shaper->GetYaxis()->SetLabelSize(0.04);
  shaper->SetXTitle("shape good HE per event, per channel\b");
  shaper->SetMarkerColor(2);
  shaper->SetLineColor(2);
  shaper->Draw("");

  TH1F *badsh = (TH1F *)hfile1->Get("h_shape_bad_channels_HE");
  TH1F *badsh0 = (TH1F *)hfile1->Get("h_shape0_bad_channels_HE");
  TH1F *badshr = (TH1F *)badsh->Clone("badshr");
  badsh->Sumw2();
  badsh0->Sumw2();
  badshr->Divide(badsh, badsh0, 1, 1, "B");  //
  badshr->Sumw2();
  c1->cd(2);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  badsh->SetMarkerStyle(20);
  badsh->SetMarkerSize(0.4);
  badsh->GetYaxis()->SetLabelSize(0.04);
  badsh->SetXTitle("TS weighted by A bad HE \b");
  badsh->SetMarkerColor(2);
  badsh->SetLineColor(2);
  badsh->Draw("");
  c1->cd(4);
  badsh0->SetMarkerStyle(20);
  badsh0->SetMarkerSize(0.4);
  badsh0->GetYaxis()->SetLabelSize(0.04);
  badsh0->SetXTitle("TS bad HE \b");
  badsh0->SetMarkerColor(2);
  badsh0->SetLineColor(2);
  badsh0->Draw("");
  c1->cd(6);
  badshr->SetMarkerStyle(20);
  badshr->SetMarkerSize(0.4);
  badshr->GetYaxis()->SetLabelSize(0.04);
  badshr->SetXTitle("shape bad HE per event, per channel\b");
  badshr->SetMarkerColor(2);
  badshr->SetLineColor(2);
  badshr->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 17
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH1F *timeHFt = (TH1F *)hfile1->Get("h_tdc_HF_tdc");
  TH1F *timeHF = (TH1F *)hfile1->Get("h_tdc_HF_time");
  TH1F *timeHF0 = (TH1F *)hfile1->Get("h_tdc_HF_time0");
  TH1F *timeHFr = (TH1F *)timeHF->Clone("timeHFr");
  timeHF->Sumw2();
  timeHF0->Sumw2();
  timeHFr->Divide(timeHF, timeHF0, 1, 1, "B");  //
  timeHFr->Sumw2();
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  timeHF->SetMarkerStyle(20);
  timeHF->SetMarkerSize(0.4);
  timeHF->GetYaxis()->SetLabelSize(0.04);
  timeHF->SetXTitle("time50 weighted by A  HF \b");
  timeHF->SetMarkerColor(2);
  timeHF->SetLineColor(0);
  timeHF->Draw("");
  c1->cd(2);
  timeHF0->SetMarkerStyle(20);
  timeHF0->SetMarkerSize(0.4);
  timeHF0->GetYaxis()->SetLabelSize(0.04);
  timeHF0->SetXTitle("time50 HF \b");
  timeHF0->SetMarkerColor(2);
  timeHF0->SetLineColor(0);
  timeHF0->Draw("");
  c1->cd(3);
  timeHFr->SetMarkerStyle(20);
  timeHFr->SetMarkerSize(0.4);
  timeHFr->GetYaxis()->SetLabelSize(0.04);
  timeHFr->SetXTitle("shape HF \b");
  timeHFr->SetMarkerColor(2);
  timeHFr->SetLineColor(0);
  timeHFr->SetMinimum(0.);
  timeHFr->SetMaximum(1000.);
  timeHFr->Draw("");
  c1->cd(4);
  gPad->SetLogy();
  timeHFt->SetMarkerStyle(20);
  timeHFt->SetMarkerSize(0.4);
  timeHFt->GetYaxis()->SetLabelSize(0.04);
  timeHFt->SetXTitle("initial tdc HF \b");
  timeHFt->SetMarkerColor(2);
  timeHFt->SetLineColor(2);
  timeHFt->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 18
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  TH1F *ampldefaultHF = (TH1F *)hfile1->Get("h_tdc_HF_ampldefault");
  TH1F *ampldefaultHF50 = (TH1F *)hfile1->Get("h_tdc_HF_ampldefault50");
  TH1F *ampldefaultHF63 = (TH1F *)hfile1->Get("h_tdc_HF_ampldefault63");
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefaultHF->SetMarkerStyle(20);
  ampldefaultHF->SetMarkerSize(0.4);
  ampldefaultHF->GetYaxis()->SetLabelSize(0.04);
  ampldefaultHF->SetXTitle("A_TS HF \b");
  ampldefaultHF->SetMarkerColor(2);
  ampldefaultHF->SetLineColor(2);
  gPad->SetLogy();
  ampldefaultHF->Draw("");

  c1->cd(2);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefaultHF50->SetMarkerStyle(20);
  ampldefaultHF50->SetMarkerSize(0.4);
  ampldefaultHF50->GetYaxis()->SetLabelSize(0.04);
  ampldefaultHF50->SetXTitle("A_TS_50 HF \b");
  ampldefaultHF50->SetMarkerColor(2);
  ampldefaultHF50->SetLineColor(2);
  gPad->SetLogy();
  ampldefaultHF50->Draw("");

  c1->cd(3);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  ampldefaultHF63->SetMarkerStyle(20);
  ampldefaultHF63->SetMarkerSize(0.4);
  ampldefaultHF63->GetYaxis()->SetLabelSize(0.04);
  ampldefaultHF63->SetXTitle("A_TS_63 HF \b");
  ampldefaultHF63->SetMarkerColor(2);
  ampldefaultHF63->SetLineColor(2);
  gPad->SetLogy();
  ampldefaultHF63->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 19
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 1);

  TH1F *timeHFVSampldefault = (TH1F *)hfile1->Get("h_tdc_HF_timeVSampldefault");
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  timeHFVSampldefault->SetMarkerStyle(20);
  timeHFVSampldefault->SetMarkerSize(0.4);
  timeHFVSampldefault->GetYaxis()->SetLabelSize(0.04);
  timeHFVSampldefault->SetXTitle("timeVStampldefault HF \b");
  timeHFVSampldefault->SetMarkerColor(2);
  timeHFVSampldefault->SetLineColor(2);
  timeHFVSampldefault->Draw("box");

  c1->Update();
  //========================================================================================== 20
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  TH1F *shapeHF = (TH1F *)hfile1->Get("h_shape_good_channels_HF");
  TH1F *shapeHF0 = (TH1F *)hfile1->Get("h_shape0_good_channels_HF");
  TH1F *shapeHFr = (TH1F *)shapeHF->Clone("shapeHFr");
  shapeHF->Sumw2();
  shapeHF0->Sumw2();
  shapeHFr->Divide(shapeHF, shapeHF0, 1, 1, "B");  //
  shapeHFr->Sumw2();
  c1->cd(1);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  shapeHF->SetMarkerStyle(20);
  shapeHF->SetMarkerSize(0.4);
  shapeHF->GetYaxis()->SetLabelSize(0.04);
  shapeHF->SetXTitle("TS weighted by A good HF \b");
  shapeHF->SetMarkerColor(2);
  shapeHF->SetLineColor(2);
  shapeHF->Draw("");
  c1->cd(3);
  shapeHF0->SetMarkerStyle(20);
  shapeHF0->SetMarkerSize(0.4);
  shapeHF0->GetYaxis()->SetLabelSize(0.04);
  shapeHF0->SetXTitle("TS good HF \b");
  shapeHF0->SetMarkerColor(2);
  shapeHF0->SetLineColor(2);
  shapeHF0->Draw("");
  c1->cd(5);
  shapeHFr->SetMarkerStyle(20);
  shapeHFr->SetMarkerSize(0.4);
  shapeHFr->GetYaxis()->SetLabelSize(0.04);
  shapeHFr->SetXTitle("shape good HF per event, per channel\b");
  shapeHFr->SetMarkerColor(2);
  shapeHFr->SetLineColor(2);
  shapeHFr->Draw("");

  TH1F *badshHF = (TH1F *)hfile1->Get("h_shape_bad_channels_HF");
  TH1F *badshHF0 = (TH1F *)hfile1->Get("h_shape0_bad_channels_HF");
  TH1F *badshHFr = (TH1F *)badshHF->Clone("badshHFr");
  badshHF->Sumw2();
  badshHF0->Sumw2();
  badshHFr->Divide(badshHF, badshHF0, 1, 1, "B");  //
  badshHFr->Sumw2();
  c1->cd(2);
  //            gPad->SetLogy();
  //    gPad->SetLogx();
  badshHF->SetMarkerStyle(20);
  badshHF->SetMarkerSize(0.4);
  badshHF->GetYaxis()->SetLabelSize(0.04);
  badshHF->SetXTitle("TS weighted by A bad HF \b");
  badshHF->SetMarkerColor(2);
  badshHF->SetLineColor(2);
  badshHF->Draw("");
  c1->cd(4);
  badshHF0->SetMarkerStyle(20);
  badshHF0->SetMarkerSize(0.4);
  badshHF0->GetYaxis()->SetLabelSize(0.04);
  badshHF0->SetXTitle("TS bad HF \b");
  badshHF0->SetMarkerColor(2);
  badshHF0->SetLineColor(2);
  badshHF0->Draw("");
  c1->cd(6);
  badshHFr->SetMarkerStyle(20);
  badshHFr->SetMarkerSize(0.4);
  badshHFr->GetYaxis()->SetLabelSize(0.04);
  badshHFr->SetXTitle("shape bad HF per event, per channel\b");
  badshHFr->SetMarkerColor(2);
  badshHFr->SetLineColor(2);
  badshHFr->Draw("");

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
