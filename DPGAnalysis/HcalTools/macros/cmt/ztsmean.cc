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
  gStyle->SetOptStat(0);  //  no statistics _or_
                          //	        	  gStyle->SetOptStat(11111111);
                          //gStyle->SetOptStat(1101);// mame mean and rms
                          //	gStyle->SetOptStat(0101);// name and entries
                          //	gStyle->SetOptStat(1110000);// und over, integral !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                          //	   gStyle->SetOptStat(1100);// mean and rms only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                          //				gStyle->SetOptStat(1110000);// und over, integral !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //	gStyle->SetOptStat(101110);                                          // entries, meam, rms, overflow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

  //	TFile *hfile1= new TFile("/afs/cern.ch/user/z/zhokin/public/html/RMT/lastMarch2014/cmt/Global_229713.root", "READ");
  //	TFile *hfile1= new TFile("test.root", "READ");
  //

  TFile *hfile1 = new TFile("Global_362596.root", "READ");
  //TFile *hfile1 = new TFile("Global_362365.root", "READ");

  TPostScript psfile("ztsmean.ps", 111);

  //

  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);

  hfile1->ls();
  TDirectory *dir = (TDirectory *)hfile1->FindObjectAny(dirnm.c_str());

  //=============================================================================================== 1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *twod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HB");
  TH2F *twod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  twod1->Sumw2();
  twod0->Sumw2();
  //    if(twod0->IsA()->InheritsFrom("TH2F")){
  TH2F *Ceff = (TH2F *)twod1->Clone("Ceff");
  Ceff->Divide(twod1, twod0, 1, 1, "B");
  Ceff->Sumw2();
  //   }
  c1->cd(1);
  TH2F *twop1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HB");
  TH2F *twop0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  twop1->Sumw2();
  twop0->Sumw2();
  //   if(twop0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cefz225 = (TH2F *)twop1->Clone("Cefz225");
  Cefz225->Divide(twop1, twop0, 1, 1, "B");
  Cefz225->Sumw2();
  //   }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cefz225->SetMarkerStyle(20);
  Cefz225->SetMarkerSize(0.4);
  //    Cefz225->GetYaxis()->SetLabelSize(0.04);
  Cefz225->GetZaxis()->SetLabelSize(0.08);
  Cefz225->SetXTitle("#eta \b");
  Cefz225->SetYTitle("#phi \b");
  Cefz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HB Depth1 \b");
  Cefz225->SetMarkerColor(2);
  Cefz225->SetLineColor(2);
  Cefz225->SetMaximum(1.000);
  Cefz225->SetMinimum(0.0001);
  Cefz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa1 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HB");
  gPad->SetLogy();
  aaaaaa1->SetMarkerStyle(20);
  aaaaaa1->SetMarkerSize(0.8);
  aaaaaa1->GetYaxis()->SetLabelSize(0.04);
  aaaaaa1->SetXTitle("TSmeanA in each event & cell HB \b");
  aaaaaa1->SetMarkerColor(2);
  aaaaaa1->SetLineColor(2);
  aaaaaa1->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth1_HB = (TH2F *)Ceff->Clone("Diffe_Depth1_HB");
  for (int i = 1; i <= Ceff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Ceff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = Ceff->GetBinContent(i, j);
      Diffe_Depth1_HB->SetBinContent(i, j, 0.);
      //	  Diffe_Depth1_HB->SetBinContent(i,j,ccc1);
      if (ccc1 < 2.5 || ccc1 > 4.5)
        Diffe_Depth1_HB->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1_HB->SetMarkerStyle(20);
  Diffe_Depth1_HB->SetMarkerSize(0.4);
  Diffe_Depth1_HB->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1_HB->SetTitle("any Error, HB Depth1 \n");
  Diffe_Depth1_HB->SetXTitle("#eta \b");
  Diffe_Depth1_HB->SetYTitle("#phi \b");
  Diffe_Depth1_HB->SetZTitle("<TSmeanA> out 2.5-4.5- HB Depth1 \b");
  Diffe_Depth1_HB->SetMarkerColor(2);
  Diffe_Depth1_HB->SetLineColor(2);
  Diffe_Depth1_HB->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth1_HB = new TH1F("diffTSmeanA_Depth1_HB", "", 100, 1.0, 6.0);
  for (int i = 1; i <= Ceff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Ceff->GetYaxis()->GetNbins(); j++) {
      if (Ceff->GetBinContent(i, j) != 0) {
        double ccc1 = Ceff->GetBinContent(i, j);
        diffTSmeanA_Depth1_HB->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth1_HB->SetMarkerStyle(20);
  diffTSmeanA_Depth1_HB->SetMarkerSize(0.4);
  diffTSmeanA_Depth1_HB->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth1_HB->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth1_HB->SetMarkerColor(2);
  diffTSmeanA_Depth1_HB->SetLineColor(2);
  diffTSmeanA_Depth1_HB->Draw("");

  c1->Update();

  //=============================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *awod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HB");
  TH2F *awod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  awod1->Sumw2();
  awod0->Sumw2();
  //   if(awod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C2ff = (TH2F *)awod1->Clone("C2ff");
  C2ff->Divide(awod1, awod0, 1, 1, "B");
  C2ff->Sumw2();
  //  }
  c1->cd(1);
  TH2F *bwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HB");
  TH2F *bwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  bwod1->Sumw2();
  bwod0->Sumw2();
  //   if(bwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C2fz225 = (TH2F *)bwod1->Clone("C2fz225");
  C2fz225->Divide(bwod1, bwod0, 1, 1, "B");
  C2fz225->Sumw2();
  //   }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C2fz225->SetMarkerStyle(20);
  C2fz225->SetMarkerSize(0.4);
  C2fz225->GetZaxis()->SetLabelSize(0.08);
  C2fz225->SetXTitle("#eta \b");
  C2fz225->SetYTitle("#phi \b");
  C2fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HB Depth2 \b");
  C2fz225->SetMarkerColor(2);
  C2fz225->SetLineColor(2);
  C2fz225->SetMaximum(1.000);
  C2fz225->SetMinimum(0.0001);
  C2fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa2 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HB");
  gPad->SetLogy();
  aaaaaa2->SetMarkerStyle(20);
  aaaaaa2->SetMarkerSize(0.8);
  aaaaaa2->GetYaxis()->SetLabelSize(0.04);
  aaaaaa2->SetXTitle("TSmeanA in each event & cell HB \b");
  aaaaaa2->SetMarkerColor(2);
  aaaaaa2->SetLineColor(2);
  aaaaaa2->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth2_HB = (TH2F *)C2ff->Clone("Diffe_Depth2_HB");
  for (int i = 1; i <= C2ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C2ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C2ff->GetBinContent(i, j);
      Diffe_Depth2_HB->SetBinContent(i, j, 0.);
      if (ccc1 < 2.4 || ccc1 > 4.4)
        Diffe_Depth2_HB->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2_HB->SetMarkerStyle(20);
  Diffe_Depth2_HB->SetMarkerSize(0.4);
  Diffe_Depth2_HB->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2_HB->SetTitle("any Error, HB Depth2 \n");
  Diffe_Depth2_HB->SetXTitle("#eta \b");
  Diffe_Depth2_HB->SetYTitle("#phi \b");
  Diffe_Depth2_HB->SetZTitle("<TSmeanA> out 2.4-4.4 - HB Depth2 \b");
  Diffe_Depth2_HB->SetMarkerColor(2);
  Diffe_Depth2_HB->SetLineColor(2);
  Diffe_Depth2_HB->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth2_HB = new TH1F("diffTSmeanA_Depth2_HB", "", 100, 1.0, 6.0);
  for (int i = 1; i <= C2ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C2ff->GetYaxis()->GetNbins(); j++) {
      if (C2ff->GetBinContent(i, j) != 0) {
        double ccc1 = C2ff->GetBinContent(i, j);
        diffTSmeanA_Depth2_HB->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth2_HB->SetMarkerStyle(20);
  diffTSmeanA_Depth2_HB->SetMarkerSize(0.4);
  diffTSmeanA_Depth2_HB->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth2_HB->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth2_HB->SetMarkerColor(2);
  diffTSmeanA_Depth2_HB->SetLineColor(2);
  diffTSmeanA_Depth2_HB->Draw("");

  c1->Update();

  //=============================================================================================== 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *cwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HE");
  TH2F *cwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  cwod1->Sumw2();
  cwod0->Sumw2();
  //  if(cwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C3ff = (TH2F *)cwod1->Clone("C3ff");
  C3ff->Divide(cwod1, cwod0, 1, 1, "B");
  C3ff->Sumw2();
  //   }
  c1->cd(1);
  TH2F *dwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HE");
  TH2F *dwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  dwod1->Sumw2();
  dwod0->Sumw2();
  //   if(dwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C3fz225 = (TH2F *)dwod1->Clone("C3fz225");
  C3fz225->Divide(dwod1, dwod0, 1, 1, "B");
  C3fz225->Sumw2();
  //   }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C3fz225->SetMarkerStyle(20);
  C3fz225->SetMarkerSize(0.4);
  C3fz225->GetZaxis()->SetLabelSize(0.08);
  C3fz225->SetXTitle("#eta \b");
  C3fz225->SetYTitle("#phi \b");
  C3fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HE Depth1 \b");
  C3fz225->SetMarkerColor(2);
  C3fz225->SetLineColor(2);
  C3fz225->SetMaximum(1.000);
  C3fz225->SetMinimum(0.0001);
  C3fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa3 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HE");
  gPad->SetLogy();
  aaaaaa3->SetMarkerStyle(20);
  aaaaaa3->SetMarkerSize(0.8);
  aaaaaa3->GetYaxis()->SetLabelSize(0.04);
  aaaaaa3->SetXTitle("TSmeanA in each event & cell HE \b");
  aaaaaa3->SetMarkerColor(2);
  aaaaaa3->SetLineColor(2);
  aaaaaa3->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth1_HE = (TH2F *)C3ff->Clone("Diffe_Depth1_HE");
  for (int i = 1; i <= C3ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C3ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C3ff->GetBinContent(i, j);
      Diffe_Depth1_HE->SetBinContent(i, j, 0.);
      if (ccc1 < 2.4 || ccc1 > 4.6)
        Diffe_Depth1_HE->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1_HE->SetMarkerStyle(20);
  Diffe_Depth1_HE->SetMarkerSize(0.4);
  Diffe_Depth1_HE->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1_HE->SetTitle("any Error, HE Depth1 \n");
  Diffe_Depth1_HE->SetXTitle("#eta \b");
  Diffe_Depth1_HE->SetYTitle("#phi \b");
  Diffe_Depth1_HE->SetZTitle("<TSmeanA> out 2.4-4.6 - HE Depth1 \b");
  Diffe_Depth1_HE->SetMarkerColor(2);
  Diffe_Depth1_HE->SetLineColor(2);
  Diffe_Depth1_HE->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth1_HE = new TH1F("diffTSmeanA_Depth1_HE", "", 100, 0.0, 7.0);
  for (int i = 1; i <= C3ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C3ff->GetYaxis()->GetNbins(); j++) {
      if (C3ff->GetBinContent(i, j) != 0) {
        double ccc1 = C3ff->GetBinContent(i, j);
        diffTSmeanA_Depth1_HE->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth1_HE->SetMarkerStyle(20);
  diffTSmeanA_Depth1_HE->SetMarkerSize(0.4);
  diffTSmeanA_Depth1_HE->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth1_HE->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth1_HE->SetMarkerColor(2);
  diffTSmeanA_Depth1_HE->SetLineColor(2);
  diffTSmeanA_Depth1_HE->Draw("");

  c1->Update();

  //=============================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *ewod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HE");
  TH2F *ewod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  ewod1->Sumw2();
  ewod0->Sumw2();
  //  if(ewod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C4ff = (TH2F *)ewod1->Clone("C4ff");
  C4ff->Divide(ewod1, ewod0, 1, 1, "B");
  C4ff->Sumw2();
  // }
  c1->cd(1);
  TH2F *fwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HE");
  TH2F *fwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  fwod1->Sumw2();
  fwod0->Sumw2();
  //  if(fwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C4fz225 = (TH2F *)fwod1->Clone("C4fz225");
  C4fz225->Divide(fwod1, fwod0, 1, 1, "B");
  C4fz225->Sumw2();
  //  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C4fz225->SetMarkerStyle(20);
  C4fz225->SetMarkerSize(0.4);
  C4fz225->GetZaxis()->SetLabelSize(0.08);
  C4fz225->SetXTitle("#eta \b");
  C4fz225->SetYTitle("#phi \b");
  C4fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HE Depth2 \b");
  C4fz225->SetMarkerColor(2);
  C4fz225->SetLineColor(2);
  C4fz225->SetMaximum(1.000);
  C4fz225->SetMinimum(0.0001);
  C4fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa4 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HE");
  gPad->SetLogy();
  aaaaaa4->SetMarkerStyle(20);
  aaaaaa4->SetMarkerSize(0.8);
  aaaaaa4->GetYaxis()->SetLabelSize(0.04);
  aaaaaa4->SetXTitle("TSmeanA in each event & cell HE \b");
  aaaaaa4->SetMarkerColor(2);
  aaaaaa4->SetLineColor(2);
  aaaaaa4->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth2_HE = (TH2F *)C4ff->Clone("Diffe_Depth2_HE");
  for (int i = 1; i <= C4ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C4ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C4ff->GetBinContent(i, j);
      Diffe_Depth2_HE->SetBinContent(i, j, 0.);
      if (ccc1 < 2.4 || ccc1 > 4.6)
        Diffe_Depth2_HE->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2_HE->SetMarkerStyle(20);
  Diffe_Depth2_HE->SetMarkerSize(0.4);
  Diffe_Depth2_HE->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2_HE->SetTitle("any Error, HE Depth2 \n");
  Diffe_Depth2_HE->SetXTitle("#eta \b");
  Diffe_Depth2_HE->SetYTitle("#phi \b");
  Diffe_Depth2_HE->SetZTitle("<TSmeanA> out 2.4-4.6 - HE Depth2 \b");
  Diffe_Depth2_HE->SetMarkerColor(2);
  Diffe_Depth2_HE->SetLineColor(2);
  Diffe_Depth2_HE->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth2_HE = new TH1F("diffTSmeanA_Depth2_HE", "", 100, 0.0, 7.0);
  for (int i = 1; i <= C4ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C4ff->GetYaxis()->GetNbins(); j++) {
      if (C4ff->GetBinContent(i, j) != 0) {
        double ccc1 = C4ff->GetBinContent(i, j);
        diffTSmeanA_Depth2_HE->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth2_HE->SetMarkerStyle(20);
  diffTSmeanA_Depth2_HE->SetMarkerSize(0.4);
  diffTSmeanA_Depth2_HE->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth2_HE->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth2_HE->SetMarkerColor(2);
  diffTSmeanA_Depth2_HE->SetLineColor(2);
  diffTSmeanA_Depth2_HE->Draw("");

  c1->Update();

  //=============================================================================================== 5
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *gwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HE");
  TH2F *gwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  gwod1->Sumw2();
  gwod0->Sumw2();
  //  if(gwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C5ff = (TH2F *)gwod1->Clone("C5ff");
  C5ff->Divide(gwod1, gwod0, 1, 1, "B");
  C5ff->Sumw2();
  //  }
  c1->cd(1);
  TH2F *jwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA225_HE");
  TH2F *jwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  jwod1->Sumw2();
  jwod0->Sumw2();
  // if(jwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C5fz225 = (TH2F *)jwod1->Clone("C5fz225");
  C5fz225->Divide(jwod1, jwod0, 1, 1, "B");
  C5fz225->Sumw2();
  //  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C5fz225->SetMarkerStyle(20);
  C5fz225->SetMarkerSize(0.4);
  C5fz225->GetZaxis()->SetLabelSize(0.08);
  C5fz225->SetXTitle("#eta \b");
  C5fz225->SetYTitle("#phi \b");
  C5fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HE Depth3 \b");
  C5fz225->SetMarkerColor(2);
  C5fz225->SetLineColor(2);
  C5fz225->SetMaximum(1.000);
  C5fz225->SetMinimum(0.0001);
  C5fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa5 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HE");
  gPad->SetLogy();
  aaaaaa5->SetMarkerStyle(20);
  aaaaaa5->SetMarkerSize(0.8);
  aaaaaa5->GetYaxis()->SetLabelSize(0.04);
  aaaaaa5->SetXTitle("TSmeanA in each event & cell HE \b");
  aaaaaa5->SetMarkerColor(2);
  aaaaaa5->SetLineColor(2);
  aaaaaa5->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth3_HE = (TH2F *)C5ff->Clone("Diffe_Depth3_HE");
  for (int i = 1; i <= C5ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C5ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C5ff->GetBinContent(i, j);
      Diffe_Depth3_HE->SetBinContent(i, j, 0.);
      if (ccc1 < 2.4 || ccc1 > 4.6)
        Diffe_Depth3_HE->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth3_HE->SetMarkerStyle(20);
  Diffe_Depth3_HE->SetMarkerSize(0.4);
  Diffe_Depth3_HE->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth3_HE->SetTitle("any Error, HE Depth3 \n");
  Diffe_Depth3_HE->SetXTitle("#eta \b");
  Diffe_Depth3_HE->SetYTitle("#phi \b");
  Diffe_Depth3_HE->SetZTitle("<TSmeanA> out 2.4-4.6 - HE Depth3 \b");
  Diffe_Depth3_HE->SetMarkerColor(2);
  Diffe_Depth3_HE->SetLineColor(2);
  Diffe_Depth3_HE->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth3_HE = new TH1F("diffTSmeanA_Depth3_HE", "", 100, 0.0, 7.0);
  for (int i = 1; i <= C5ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C5ff->GetYaxis()->GetNbins(); j++) {
      if (C5ff->GetBinContent(i, j) != 0) {
        double ccc1 = C5ff->GetBinContent(i, j);
        diffTSmeanA_Depth3_HE->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth3_HE->SetMarkerStyle(20);
  diffTSmeanA_Depth3_HE->SetMarkerSize(0.4);
  diffTSmeanA_Depth3_HE->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth3_HE->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth3_HE->SetMarkerColor(2);
  diffTSmeanA_Depth3_HE->SetLineColor(2);
  diffTSmeanA_Depth3_HE->Draw("");

  c1->Update();

  //=============================================================================================== 6
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *iwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HF");
  TH2F *iwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  iwod1->Sumw2();
  iwod0->Sumw2();
  // if(iwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C6ff = (TH2F *)iwod1->Clone("C6ff");
  C6ff->Divide(iwod1, iwod0, 1, 1, "B");
  C6ff->Sumw2();
  //  }
  c1->cd(1);
  TH2F *kwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA225_HF");
  TH2F *kwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  kwod1->Sumw2();
  kwod0->Sumw2();
  //  if(kwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C6fz225 = (TH2F *)kwod1->Clone("C6fz225");
  C6fz225->Divide(kwod1, kwod0, 1, 1, "B");
  C6fz225->Sumw2();
  //  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C6fz225->SetMarkerStyle(20);
  C6fz225->SetMarkerSize(0.4);
  //    C6fz225->GetYaxis()->SetLabelSize(0.04);
  C6fz225->GetZaxis()->SetLabelSize(0.08);
  C6fz225->SetXTitle("#eta \b");
  C6fz225->SetYTitle("#phi \b");
  C6fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HF Depth1 \b");
  C6fz225->SetMarkerColor(2);
  C6fz225->SetLineColor(2);
  C6fz225->SetMaximum(1.000);
  C6fz225->SetMinimum(0.0001);
  C6fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa6 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HF");
  gPad->SetLogy();
  aaaaaa6->SetMarkerStyle(20);
  aaaaaa6->SetMarkerSize(0.8);
  aaaaaa6->GetYaxis()->SetLabelSize(0.04);
  aaaaaa6->SetXTitle("TSmeanA in each event & cell HF \b");
  aaaaaa6->SetMarkerColor(2);
  aaaaaa6->SetLineColor(2);
  aaaaaa6->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth1_HF = (TH2F *)C6ff->Clone("Diffe_Depth1_HF");
  for (int i = 1; i <= C6ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C6ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C6ff->GetBinContent(i, j);
      Diffe_Depth1_HF->SetBinContent(i, j, 0.);
      if (ccc1 < 0.5 || ccc1 > 1.5)
        Diffe_Depth1_HF->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth1_HF->SetMarkerStyle(20);
  Diffe_Depth1_HF->SetMarkerSize(0.4);
  Diffe_Depth1_HF->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth1_HF->SetTitle("any Error, HF Depth1 \n");
  Diffe_Depth1_HF->SetXTitle("#eta \b");
  Diffe_Depth1_HF->SetYTitle("#phi \b");
  Diffe_Depth1_HF->SetZTitle("<TSmeanA> out 0.5-1.5   -  HF Depth1 \b");
  Diffe_Depth1_HF->SetMarkerColor(2);
  Diffe_Depth1_HF->SetLineColor(2);
  Diffe_Depth1_HF->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth1_HF = new TH1F("diffTSmeanA_Depth1_HF", "", 100, 0.0, 2.0);
  for (int i = 1; i <= C6ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C6ff->GetYaxis()->GetNbins(); j++) {
      if (C6ff->GetBinContent(i, j) != 0) {
        double ccc1 = C6ff->GetBinContent(i, j);
        diffTSmeanA_Depth1_HF->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth1_HF->SetMarkerStyle(20);
  diffTSmeanA_Depth1_HF->SetMarkerSize(0.4);
  diffTSmeanA_Depth1_HF->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth1_HF->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth1_HF->SetMarkerColor(2);
  diffTSmeanA_Depth1_HF->SetLineColor(2);
  diffTSmeanA_Depth1_HF->Draw("");

  c1->Update();

  //=============================================================================================== 7
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *lwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HF");
  TH2F *lwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  lwod1->Sumw2();
  lwod0->Sumw2();
  //  if(lwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C7ff = (TH2F *)lwod1->Clone("C7ff");
  C7ff->Divide(lwod1, lwod0, 1, 1, "B");
  C7ff->Sumw2();
  //  }
  c1->cd(1);
  TH2F *mwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA225_HF");
  TH2F *mwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  mwod1->Sumw2();
  mwod0->Sumw2();
  //  if(mwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C7fz225 = (TH2F *)mwod1->Clone("C7fz225");
  C7fz225->Divide(mwod1, mwod0, 1, 1, "B");
  C7fz225->Sumw2();
  //  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C7fz225->SetMarkerStyle(20);
  C7fz225->SetMarkerSize(0.4);
  C7fz225->GetZaxis()->SetLabelSize(0.08);
  C7fz225->SetXTitle("#eta \b");
  C7fz225->SetYTitle("#phi \b");
  C7fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5 - HF Depth2 \b");
  C7fz225->SetMarkerColor(2);
  C7fz225->SetLineColor(2);
  C7fz225->SetMaximum(1.000);
  C7fz225->SetMinimum(0.0001);
  C7fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa7 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HF");
  gPad->SetLogy();
  aaaaaa7->SetMarkerStyle(20);
  aaaaaa7->SetMarkerSize(0.8);
  aaaaaa7->GetYaxis()->SetLabelSize(0.04);
  aaaaaa7->SetXTitle("TSmeanA in each event & cell HF \b");
  aaaaaa7->SetMarkerColor(2);
  aaaaaa7->SetLineColor(2);
  aaaaaa7->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth2_HF = (TH2F *)C7ff->Clone("Diffe_Depth2_HF");
  for (int i = 1; i <= C7ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C7ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C7ff->GetBinContent(i, j);
      Diffe_Depth2_HF->SetBinContent(i, j, 0.);
      if (ccc1 < 0.5 || ccc1 > 1.5)
        Diffe_Depth2_HF->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth2_HF->SetMarkerStyle(20);
  Diffe_Depth2_HF->SetMarkerSize(0.4);
  Diffe_Depth2_HF->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth2_HF->SetTitle("any Error, HF Depth2 \n");
  Diffe_Depth2_HF->SetXTitle("#eta \b");
  Diffe_Depth2_HF->SetYTitle("#phi \b");
  Diffe_Depth2_HF->SetZTitle("<TSmeanA> out 0.5-1.5   -HF Depth2 \b");
  Diffe_Depth2_HF->SetMarkerColor(2);
  Diffe_Depth2_HF->SetLineColor(2);
  Diffe_Depth2_HF->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth2_HF = new TH1F("diffTSmeanA_Depth2_HF", "", 100, 0.0, 2.0);
  for (int i = 1; i <= C7ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C7ff->GetYaxis()->GetNbins(); j++) {
      if (C7ff->GetBinContent(i, j) != 0) {
        double ccc1 = C7ff->GetBinContent(i, j);
        diffTSmeanA_Depth2_HF->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth2_HF->SetMarkerStyle(20);
  diffTSmeanA_Depth2_HF->SetMarkerSize(0.4);
  diffTSmeanA_Depth2_HF->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth2_HF->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth2_HF->SetMarkerColor(2);
  diffTSmeanA_Depth2_HF->SetLineColor(2);
  diffTSmeanA_Depth2_HF->Draw("");

  c1->Update();

  //=============================================================================================== 8
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  TH2F *nwod1 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HO");
  TH2F *nwod0 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  nwod1->Sumw2();
  nwod0->Sumw2();
  //  if(nwod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C8ff = (TH2F *)nwod1->Clone("C8ff");
  C8ff->Divide(nwod1, nwod0, 1, 1, "B");
  C8ff->Sumw2();
  //  }
  c1->cd(1);
  TH2F *owod1 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA225_HO");
  TH2F *owod0 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  owod1->Sumw2();
  owod0->Sumw2();
  //   if(owod0->IsA()->InheritsFrom("TH2F")){
  TH2F *C8fz225 = (TH2F *)owod1->Clone("C8fz225");
  C8fz225->Divide(owod1, owod0, 1, 1, "B");
  C8fz225->Sumw2();
  //  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  C8fz225->SetMarkerStyle(20);
  C8fz225->SetMarkerSize(0.4);
  C8fz225->GetZaxis()->SetLabelSize(0.08);
  C8fz225->SetXTitle("#eta \b");
  C8fz225->SetYTitle("#phi \b");
  C8fz225->SetZTitle("Rate for TSmeanA in each event & cell out 2.3-5.5 - HO Depth4 \b");
  C8fz225->SetMarkerColor(2);
  C8fz225->SetLineColor(2);
  C8fz225->SetMaximum(1.000);
  C8fz225->SetMinimum(0.000005);
  C8fz225->Draw("COLZ");

  c1->cd(2);
  TH1F *aaaaaa8 = (TH1F *)dir->FindObjectAny("h_TSmeanA_HO");
  gPad->SetLogy();
  aaaaaa8->SetMarkerStyle(20);
  aaaaaa8->SetMarkerSize(0.8);
  aaaaaa8->GetYaxis()->SetLabelSize(0.04);
  aaaaaa8->SetXTitle("TSmeanA in each event & cell HO \b");
  aaaaaa8->SetMarkerColor(2);
  aaaaaa8->SetLineColor(2);
  aaaaaa8->Draw("");

  c1->cd(3);
  ///////////////////////////////////////
  TH2F *Diffe_Depth4_HO = (TH2F *)C8ff->Clone("Diffe_Depth4_HO");
  for (int i = 1; i <= C8ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C8ff->GetYaxis()->GetNbins(); j++) {
      double ccc1 = C8ff->GetBinContent(i, j);
      Diffe_Depth4_HO->SetBinContent(i, j, 0.);
      if (ccc1 < 4.0 || ccc1 > 5.0)
        Diffe_Depth4_HO->SetBinContent(i, j, ccc1);
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Diffe_Depth4_HO->SetMarkerStyle(20);
  Diffe_Depth4_HO->SetMarkerSize(0.4);
  Diffe_Depth4_HO->GetZaxis()->SetLabelSize(0.08);
  //    Diffe_Depth4_HO->SetTitle("any Error, HO Depth4 \n");
  Diffe_Depth4_HO->SetXTitle("#eta \b");
  Diffe_Depth4_HO->SetYTitle("#phi \b");
  Diffe_Depth4_HO->SetZTitle("<TSmeanA> out 4.0-5.0 - HO Depth4 \b");
  Diffe_Depth4_HO->SetMarkerColor(2);
  Diffe_Depth4_HO->SetLineColor(2);
  Diffe_Depth4_HO->Draw("COLZ");

  c1->cd(4);
  TH1F *diffTSmeanA_Depth4_HO = new TH1F("diffTSmeanA_Depth4_HO", "", 100, 4.0, 5.0);
  for (int i = 1; i <= C8ff->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= C8ff->GetYaxis()->GetNbins(); j++) {
      if (C8ff->GetBinContent(i, j) != 0) {
        double ccc1 = C8ff->GetBinContent(i, j);
        diffTSmeanA_Depth4_HO->Fill(ccc1);
      }
    }
  }
  gPad->SetLogy();
  diffTSmeanA_Depth4_HO->SetMarkerStyle(20);
  diffTSmeanA_Depth4_HO->SetMarkerSize(0.4);
  diffTSmeanA_Depth4_HO->GetYaxis()->SetLabelSize(0.04);
  diffTSmeanA_Depth4_HO->SetXTitle("<TSmeanA> in each cell \b");
  diffTSmeanA_Depth4_HO->SetMarkerColor(2);
  diffTSmeanA_Depth4_HO->SetLineColor(2);
  diffTSmeanA_Depth4_HO->Draw("");

  c1->Update();

  //======================================================================
  //======================================================================================== end
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
