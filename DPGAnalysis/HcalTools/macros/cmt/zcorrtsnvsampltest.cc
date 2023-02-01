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
  gStyle->SetOptStat(0);  //  no statistics _or_
  //	        	  gStyle->SetOptStat(11111111);
  //gStyle->SetOptStat(1101);// name mean and rms
  //	gStyle->SetOptStat(0101);// name and entries
  //	   gStyle->SetOptStat(1100);// mean and rms only !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //	gStyle->SetOptStat(1110000);// und over, integral !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //  gStyle->SetOptStat(101110);  // entries, mean, rms, overflow !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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

  TFile *hfile1 = new TFile("Global_362596.root", "READ");
  //TFile *hfile1 = new TFile("Global_362365.root", "READ");

  TPostScript psfile("zcorrtsnvsampltest.ps", 111);

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

  c1->cd(1);
  TH2F *two11 = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two11->SetMarkerStyle(20);
  two11->SetMarkerSize(0.4);
  two11->SetYTitle("timing HB \b");
  two11->SetXTitle("Q HB\b");
  two11->SetMarkerColor(1);
  two11->SetLineColor(1);
  //               gPad->SetLogx();
  //               gPad->SetLogy();
  two11->Draw("BOX");
  //   two11->Draw("");
  //    two11->Draw("COLZ");

  c1->cd(2);
  TH2F *two12 = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  two12->SetMarkerStyle(20);
  two12->SetMarkerSize(0.4);
  two12->SetYTitle("timing HE \b");
  two12->SetXTitle("Q HE\b");
  two12->SetMarkerColor(1);
  two12->SetLineColor(1);
  //   gPad->SetLogy();
  two12->Draw("BOX");
  //    two12->Draw("SCAT");

  c1->cd(3);
  TH2F *two22 = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HF");
  gPad->SetGridy();
  gPad->SetGridx();
  two22->SetMarkerStyle(20);
  two22->SetMarkerSize(0.4);
  two22->SetYTitle("timing HF \b");
  two22->SetXTitle("Q HF\b");
  two22->SetMarkerColor(1);
  two22->SetLineColor(1);
  two22->Draw("BOX");
  //    two22->Draw("ARR");

  c1->cd(4);
  TH2F *two23 = (TH2F *)dir->FindObjectAny("h2_TSnVsAyear2023_HO");
  gPad->SetGridy();
  gPad->SetGridx();
  two23->SetMarkerStyle(20);
  two23->SetMarkerSize(0.4);
  two23->SetYTitle("timing HO \b");
  two23->SetXTitle("Q HO\b");
  two23->SetMarkerColor(1);
  two23->SetLineColor(1);
  two23->Draw("BOX");

  c1->Update();

  //========================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 4);

  c1->cd(1);
  TH1F *TSNvsQ_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HB->SetMarkerStyle(20);
  TSNvsQ_HB->SetMarkerSize(0.6);
  TSNvsQ_HB->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HB->SetXTitle("Q,fb HB \b");
  TSNvsQ_HB->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HB->SetMarkerColor(4);
  TSNvsQ_HB->SetLineColor(0);
  TSNvsQ_HB->SetMinimum(0.8);
  //TSNvsQ_HB->GetXaxis()->SetRangeUser(0, MaxLumDanila + 5.);
  TSNvsQ_HB->Draw("E");

  c1->cd(2);
  TH1F *TSNvsQ_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HE");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HE->SetMarkerStyle(20);
  TSNvsQ_HE->SetMarkerSize(0.6);
  TSNvsQ_HE->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HE->SetXTitle("Q,fb HE \b");
  TSNvsQ_HE->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HE->SetMarkerColor(4);
  TSNvsQ_HE->SetLineColor(0);
  TSNvsQ_HE->SetMinimum(0.8);
  //TSNvsQ_HE->GetXaxis()->SetRangeUser(0, MaxLumDanila + 5.);
  TSNvsQ_HE->Draw("E");

  c1->cd(3);
  TH1F *TSNvsQ_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HF");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HF->SetMarkerStyle(20);
  TSNvsQ_HF->SetMarkerSize(0.6);
  TSNvsQ_HF->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HF->SetXTitle("Q,fb HF \b");
  TSNvsQ_HF->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HF->SetMarkerColor(4);
  TSNvsQ_HF->SetLineColor(0);
  TSNvsQ_HF->SetMinimum(0.8);
  //TSNvsQ_HF->GetXaxis()->SetRangeUser(0, MaxLumDanila + 5.);
  TSNvsQ_HF->Draw("E");

  c1->cd(4);
  TH1F *TSNvsQ_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HO");
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  TSNvsQ_HO->SetMarkerStyle(20);
  TSNvsQ_HO->SetMarkerSize(0.6);
  TSNvsQ_HO->GetYaxis()->SetLabelSize(0.04);
  TSNvsQ_HO->SetXTitle("Q,fb HO \b");
  TSNvsQ_HO->SetYTitle("iev*ieta*iphi*idepth \b");
  TSNvsQ_HO->SetMarkerColor(4);
  TSNvsQ_HO->SetLineColor(0);
  TSNvsQ_HO->SetMinimum(0.8);
  //TSNvsQ_HO->GetXaxis()->SetRangeUser(0, MaxLumDanila + 5.);
  TSNvsQ_HO->Draw("E");

  c1->Update();

  //========================================================================================== 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 4);

  c1->cd(1);
  TH1F *twod1_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HB");
  TH1F *twod0_HB = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HB");
  twod1_HB->Sumw2();
  twod0_HB->Sumw2();
  TH1F *Ceff_HB = (TH1F *)twod1_HB->Clone("Ceff_HB");
  //  Ceff_HB->Sumw2();
  for (int x = 1; x <= twod1_HB->GetXaxis()->GetNbins(); x++) {
    twod1_HB->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HB->Divide(twod1_HB, twod0_HB, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HB->SetMarkerStyle(20);
  Ceff_HB->SetMarkerSize(0.4);
  Ceff_HB->SetXTitle("Q, fc \b");
  Ceff_HB->SetYTitle("<timing>HB \b");
  Ceff_HB->SetMarkerColor(2);
  Ceff_HB->SetLineColor(2);
  Ceff_HB->SetMaximum(140.);
  Ceff_HB->SetMinimum(30.);
  Ceff_HB->Draw("E");
  //Ceff_HB->Draw("COLZ");

  c1->cd(2);
  TH1F *twod1_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HE");
  TH1F *twod0_HE = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HE");
  //  twod1_HE->Sumw2();
  //  twod0_HE->Sumw2();
  TH1F *Ceff_HE = (TH1F *)twod1_HE->Clone("Ceff_HE");
  //  Ceff_HE->Sumw2();
  for (int x = 1; x <= twod1_HE->GetXaxis()->GetNbins(); x++) {
    twod1_HE->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HE->Divide(twod1_HE, twod0_HE, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HE->SetMarkerStyle(20);
  Ceff_HE->SetMarkerSize(0.4);
  Ceff_HE->SetXTitle("Q, fc \b");
  Ceff_HE->SetYTitle("<timing>HE \b");
  Ceff_HE->SetMarkerColor(2);
  Ceff_HE->SetLineColor(2);
  Ceff_HE->SetMaximum(150.);
  Ceff_HE->SetMinimum(25.);
  Ceff_HE->Draw("E");
  //Ceff_HE->Draw("COLZ");

  c1->cd(3);
  TH1F *twod1_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HF");
  TH1F *twod0_HF = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HF");
  //  twod1_HF->Sumw2();
  //  twod0_HF->Sumw2();
  TH1F *Ceff_HF = (TH1F *)twod1_HF->Clone("Ceff_HF");
  //  Ceff_HF->Sumw2();
  for (int x = 1; x <= twod1_HF->GetXaxis()->GetNbins(); x++) {
    twod1_HF->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HF->Divide(twod1_HF, twod0_HF, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HF->SetMarkerStyle(20);
  Ceff_HF->SetMarkerSize(0.4);
  Ceff_HF->SetXTitle("Q, fc \b");
  Ceff_HF->SetYTitle("<timing>HF \b");
  Ceff_HF->SetMarkerColor(2);
  Ceff_HF->SetLineColor(2);
  Ceff_HF->SetMaximum(50.);
  Ceff_HF->SetMinimum(0.);
  Ceff_HF->Draw("E");
  //Ceff_HF->Draw("COLZ");

  c1->cd(4);
  TH1F *twod1_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear2023_HO");
  TH1F *twod0_HO = (TH1F *)dir->FindObjectAny("h1_TSnVsAyear20230_HO");
  // twod1_HO->Sumw2();
  //  twod0_HO->Sumw2();
  TH1F *Ceff_HO = (TH1F *)twod1_HO->Clone("Ceff_HO");
  //  Ceff_HO->Sumw2();
  for (int x = 1; x <= twod1_HO->GetXaxis()->GetNbins(); x++) {
    twod1_HO->SetBinError(float(x), 0.001);
  }  //end x
  Ceff_HO->Divide(twod1_HO, twod0_HO, 1, 1, "B");
  gPad->SetGridy();
  gPad->SetGridx();
  Ceff_HO->SetMarkerStyle(20);
  Ceff_HO->SetMarkerSize(0.4);
  Ceff_HO->SetXTitle("Q, fc \b");
  Ceff_HO->SetYTitle("<timing>HO \b");
  Ceff_HO->SetMarkerColor(2);
  Ceff_HO->SetLineColor(2);
  Ceff_HO->SetMaximum(150.);
  Ceff_HO->SetMinimum(70.);
  Ceff_HO->Draw("E");
  //Ceff_HO->Draw("COLZ");

  c1->Update();

  //========================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH2F *dva1_HBDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HB");
  TH2F *dva0_HBDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  //dva1_HBDepth1->Sumw2();
  //dva0_HBDepth1->Sumw2();
  TH2F *Seff_HBDepth1 = (TH2F *)dva1_HBDepth1->Clone("Seff_HBDepth1");
  Seff_HBDepth1->Divide(dva1_HBDepth1, dva0_HBDepth1, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HBDepth1->SetMarkerStyle(20);
  Seff_HBDepth1->SetMarkerSize(0.4);
  Seff_HBDepth1->SetXTitle("#eta \b");
  Seff_HBDepth1->SetYTitle("#phi \b");
  Seff_HBDepth1->SetZTitle("<timing> HB Depth1 \b");
  Seff_HBDepth1->SetMarkerColor(2);
  Seff_HBDepth1->SetLineColor(2);
  Seff_HBDepth1->SetMaximum(90.);
  Seff_HBDepth1->SetMinimum(85.);
  Seff_HBDepth1->Draw("COLZ");

  c1->cd(2);
  TH2F *dva1_HBDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HB");
  TH2F *dva0_HBDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HB");
  //dva1_HBDepth2->Sumw2();
  //dva0_HBDepth2->Sumw2();
  TH2F *Seff_HBDepth2 = (TH2F *)dva1_HBDepth2->Clone("Seff_HBDepth2");
  Seff_HBDepth2->Divide(dva1_HBDepth2, dva0_HBDepth2, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HBDepth2->SetMarkerStyle(20);
  Seff_HBDepth2->SetMarkerSize(0.4);
  Seff_HBDepth2->SetXTitle("#eta \b");
  Seff_HBDepth2->SetYTitle("#phi \b");
  Seff_HBDepth2->SetZTitle("<timing> HB Depth2 \b");
  Seff_HBDepth2->SetMarkerColor(2);
  Seff_HBDepth2->SetLineColor(2);
  Seff_HBDepth2->SetMaximum(90.);
  Seff_HBDepth2->SetMinimum(85.);
  Seff_HBDepth2->Draw("COLZ");

  c1->cd(3);
  TH2F *dva1_HBDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HB");
  TH2F *dva0_HBDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HB");
  //dva1_HBDepth3->Sumw2();
  //dva0_HBDepth3->Sumw2();
  TH2F *Seff_HBDepth3 = (TH2F *)dva1_HBDepth3->Clone("Seff_HBDepth3");
  Seff_HBDepth3->Divide(dva1_HBDepth3, dva0_HBDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HBDepth3->SetMarkerStyle(20);
  Seff_HBDepth3->SetMarkerSize(0.4);
  Seff_HBDepth3->SetXTitle("#eta \b");
  Seff_HBDepth3->SetYTitle("#phi \b");
  Seff_HBDepth3->SetZTitle("<timing> HB Depth3 \b");
  Seff_HBDepth3->SetMarkerColor(2);
  Seff_HBDepth3->SetLineColor(2);
  Seff_HBDepth3->SetMaximum(90.);
  Seff_HBDepth3->SetMinimum(85.);
  Seff_HBDepth3->Draw("COLZ");

  c1->cd(4);
  TH2F *dva1_HBDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HB");
  TH2F *dva0_HBDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HB");
  //dva1_HBDepth4->Sumw2();
  //dva0_HBDepth4->Sumw2();
  TH2F *Seff_HBDepth4 = (TH2F *)dva1_HBDepth4->Clone("Seff_HBDepth4");
  Seff_HBDepth4->Divide(dva1_HBDepth4, dva0_HBDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HBDepth4->SetMarkerStyle(20);
  Seff_HBDepth4->SetMarkerSize(0.4);
  Seff_HBDepth4->SetXTitle("#eta \b");
  Seff_HBDepth4->SetYTitle("#phi \b");
  Seff_HBDepth4->SetZTitle("<timing> HB Depth4 \b");
  Seff_HBDepth4->SetMarkerColor(2);
  Seff_HBDepth4->SetLineColor(2);
  Seff_HBDepth4->SetMaximum(90.);
  Seff_HBDepth4->SetMinimum(85.);
  Seff_HBDepth4->Draw("COLZ");

  c1->Update();

  //========================================================================================== 5
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH2F *dva1_HEDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HE");
  TH2F *dva0_HEDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HE");
  //dva1_HEDepth1->Sumw2();
  //dva0_HEDepth1->Sumw2();
  TH2F *Seff_HEDepth1 = (TH2F *)dva1_HEDepth1->Clone("Seff_HEDepth1");
  Seff_HEDepth1->Divide(dva1_HEDepth1, dva0_HEDepth1, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth1->SetMarkerStyle(20);
  Seff_HEDepth1->SetMarkerSize(0.4);
  Seff_HEDepth1->SetXTitle("#eta \b");
  Seff_HEDepth1->SetYTitle("#phi \b");
  Seff_HEDepth1->SetZTitle("<timing> HE Depth1 \b");
  Seff_HEDepth1->SetMarkerColor(2);
  Seff_HEDepth1->SetLineColor(2);
  Seff_HEDepth1->SetMaximum(90.);
  Seff_HEDepth1->SetMinimum(85.);
  Seff_HEDepth1->Draw("COLZ");

  c1->cd(2);
  TH2F *dva1_HEDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HE");
  TH2F *dva0_HEDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HE");
  //dva1_HEDepth2->Sumw2();
  //dva0_HEDepth2->Sumw2();
  TH2F *Seff_HEDepth2 = (TH2F *)dva1_HEDepth2->Clone("Seff_HEDepth2");
  Seff_HEDepth2->Divide(dva1_HEDepth2, dva0_HEDepth2, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth2->SetMarkerStyle(20);
  Seff_HEDepth2->SetMarkerSize(0.4);
  Seff_HEDepth2->SetXTitle("#eta \b");
  Seff_HEDepth2->SetYTitle("#phi \b");
  Seff_HEDepth2->SetZTitle("<timing> HE Depth2 \b");
  Seff_HEDepth2->SetMarkerColor(2);
  Seff_HEDepth2->SetLineColor(2);
  Seff_HEDepth2->SetMaximum(90.);
  Seff_HEDepth2->SetMinimum(85.);
  Seff_HEDepth2->Draw("COLZ");

  c1->cd(3);
  TH2F *dva1_HEDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HE");
  TH2F *dva0_HEDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HE");
  //dva1_HEDepth3->Sumw2();
  //dva0_HEDepth3->Sumw2();
  TH2F *Seff_HEDepth3 = (TH2F *)dva1_HEDepth3->Clone("Seff_HEDepth3");
  Seff_HEDepth3->Divide(dva1_HEDepth3, dva0_HEDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth3->SetMarkerStyle(20);
  Seff_HEDepth3->SetMarkerSize(0.4);
  Seff_HEDepth3->SetXTitle("#eta \b");
  Seff_HEDepth3->SetYTitle("#phi \b");
  Seff_HEDepth3->SetZTitle("<timing> HE Depth3 \b");
  Seff_HEDepth3->SetMarkerColor(2);
  Seff_HEDepth3->SetLineColor(2);
  Seff_HEDepth3->SetMaximum(90.);
  Seff_HEDepth3->SetMinimum(85.);
  Seff_HEDepth3->Draw("COLZ");

  c1->cd(4);
  TH2F *dva1_HEDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HE");
  TH2F *dva0_HEDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HE");
  //dva1_HEDepth4->Sumw2();
  //dva0_HEDepth4->Sumw2();
  TH2F *Seff_HEDepth4 = (TH2F *)dva1_HEDepth4->Clone("Seff_HEDepth4");
  Seff_HEDepth4->Divide(dva1_HEDepth4, dva0_HEDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth4->SetMarkerStyle(20);
  Seff_HEDepth4->SetMarkerSize(0.4);
  Seff_HEDepth4->SetXTitle("#eta \b");
  Seff_HEDepth4->SetYTitle("#phi \b");
  Seff_HEDepth4->SetZTitle("<timing> HE Depth4 \b");
  Seff_HEDepth4->SetMarkerColor(2);
  Seff_HEDepth4->SetLineColor(2);
  Seff_HEDepth4->SetMaximum(90.);
  Seff_HEDepth4->SetMinimum(85.);
  Seff_HEDepth4->Draw("COLZ");

  c1->cd(5);
  TH2F *dva1_HEDepth5 = (TH2F *)dir->FindObjectAny("h_mapDepth5TSmeanA_HE");
  TH2F *dva0_HEDepth5 = (TH2F *)dir->FindObjectAny("h_mapDepth5_HE");
  //dva1_HEDepth5->Sumw2();
  //dva0_HEDepth5->Sumw2();
  TH2F *Seff_HEDepth5 = (TH2F *)dva1_HEDepth5->Clone("Seff_HEDepth5");
  Seff_HEDepth5->Divide(dva1_HEDepth5, dva0_HEDepth5, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth5->SetMarkerStyle(20);
  Seff_HEDepth5->SetMarkerSize(0.4);
  Seff_HEDepth5->SetXTitle("#eta \b");
  Seff_HEDepth5->SetYTitle("#phi \b");
  Seff_HEDepth5->SetZTitle("<timing> HE Depth5 \b");
  Seff_HEDepth5->SetMarkerColor(2);
  Seff_HEDepth5->SetLineColor(2);
  Seff_HEDepth5->SetMaximum(90.);
  Seff_HEDepth5->SetMinimum(85.);
  Seff_HEDepth5->Draw("COLZ");

  c1->cd(6);
  TH2F *dva1_HEDepth6 = (TH2F *)dir->FindObjectAny("h_mapDepth6TSmeanA_HE");
  TH2F *dva0_HEDepth6 = (TH2F *)dir->FindObjectAny("h_mapDepth6_HE");
  //dva1_HEDepth6->Sumw2();
  //dva0_HEDepth6->Sumw2();
  TH2F *Seff_HEDepth6 = (TH2F *)dva1_HEDepth6->Clone("Seff_HEDepth6");
  Seff_HEDepth6->Divide(dva1_HEDepth6, dva0_HEDepth6, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth6->SetMarkerStyle(20);
  Seff_HEDepth6->SetMarkerSize(0.4);
  Seff_HEDepth6->SetXTitle("#eta \b");
  Seff_HEDepth6->SetYTitle("#phi \b");
  Seff_HEDepth6->SetZTitle("<timing> HE Depth6 \b");
  Seff_HEDepth6->SetMarkerColor(2);
  Seff_HEDepth6->SetLineColor(2);
  Seff_HEDepth6->SetMaximum(90.);
  Seff_HEDepth6->SetMinimum(85.);
  Seff_HEDepth6->Draw("COLZ");

  c1->cd(7);
  TH2F *dva1_HEDepth7 = (TH2F *)dir->FindObjectAny("h_mapDepth7TSmeanA_HE");
  TH2F *dva0_HEDepth7 = (TH2F *)dir->FindObjectAny("h_mapDepth7_HE");
  //dva1_HEDepth7->Sumw2();
  //dva0_HEDepth7->Sumw2();
  TH2F *Seff_HEDepth7 = (TH2F *)dva1_HEDepth7->Clone("Seff_HEDepth7");
  Seff_HEDepth7->Divide(dva1_HEDepth7, dva0_HEDepth7, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HEDepth7->SetMarkerStyle(20);
  Seff_HEDepth7->SetMarkerSize(0.4);
  Seff_HEDepth7->SetXTitle("#eta \b");
  Seff_HEDepth7->SetYTitle("#phi \b");
  Seff_HEDepth7->SetZTitle("<timing> HE Depth7 \b");
  Seff_HEDepth7->SetMarkerColor(2);
  Seff_HEDepth7->SetLineColor(2);
  Seff_HEDepth7->SetMaximum(90.);
  Seff_HEDepth7->SetMinimum(85.);
  Seff_HEDepth7->Draw("COLZ");

  c1->Update();

  //========================================================================================== 6
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH2F *dva1_HFDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1TSmeanA_HF");
  TH2F *dva0_HFDepth1 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HF");
  //dva1_HFDepth1->Sumw2();
  //dva0_HFDepth1->Sumw2();
  TH2F *Seff_HFDepth1 = (TH2F *)dva1_HFDepth1->Clone("Seff_HFDepth1");
  Seff_HFDepth1->Divide(dva1_HFDepth1, dva0_HFDepth1, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HFDepth1->SetMarkerStyle(20);
  Seff_HFDepth1->SetMarkerSize(0.4);
  Seff_HFDepth1->SetXTitle("#eta \b");
  Seff_HFDepth1->SetYTitle("#phi \b");
  Seff_HFDepth1->SetZTitle("<timing> HF Depth1 \b");
  Seff_HFDepth1->SetMarkerColor(2);
  Seff_HFDepth1->SetLineColor(2);
  Seff_HFDepth1->SetMaximum(27.);
  Seff_HFDepth1->SetMinimum(23.);
  Seff_HFDepth1->Draw("COLZ");

  c1->cd(2);
  TH2F *dva1_HFDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2TSmeanA_HF");
  TH2F *dva0_HFDepth2 = (TH2F *)dir->FindObjectAny("h_mapDepth2_HF");
  //dva1_HFDepth2->Sumw2();
  //dva0_HFDepth2->Sumw2();
  TH2F *Seff_HFDepth2 = (TH2F *)dva1_HFDepth2->Clone("Seff_HFDepth2");
  Seff_HFDepth2->Divide(dva1_HFDepth2, dva0_HFDepth2, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HFDepth2->SetMarkerStyle(20);
  Seff_HFDepth2->SetMarkerSize(0.4);
  Seff_HFDepth2->SetXTitle("#eta \b");
  Seff_HFDepth2->SetYTitle("#phi \b");
  Seff_HFDepth2->SetZTitle("<timing> HF Depth2 \b");
  Seff_HFDepth2->SetMarkerColor(2);
  Seff_HFDepth2->SetLineColor(2);
  Seff_HFDepth2->SetMaximum(27.);
  Seff_HFDepth2->SetMinimum(23.);
  Seff_HFDepth2->Draw("COLZ");

  c1->cd(3);
  TH2F *dva1_HFDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3TSmeanA_HF");
  TH2F *dva0_HFDepth3 = (TH2F *)dir->FindObjectAny("h_mapDepth3_HF");
  //dva1_HFDepth3->Sumw2();
  //dva0_HFDepth3->Sumw2();
  TH2F *Seff_HFDepth3 = (TH2F *)dva1_HFDepth3->Clone("Seff_HFDepth3");
  Seff_HFDepth3->Divide(dva1_HFDepth3, dva0_HFDepth3, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HFDepth3->SetMarkerStyle(20);
  Seff_HFDepth3->SetMarkerSize(0.4);
  Seff_HFDepth3->SetXTitle("#eta \b");
  Seff_HFDepth3->SetYTitle("#phi \b");
  Seff_HFDepth3->SetZTitle("<timing> HF Depth3 \b");
  Seff_HFDepth3->SetMarkerColor(2);
  Seff_HFDepth3->SetLineColor(2);
  Seff_HFDepth3->SetMaximum(27.);
  Seff_HFDepth3->SetMinimum(23.);
  Seff_HFDepth3->Draw("COLZ");

  c1->cd(4);
  TH2F *dva1_HFDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HF");
  TH2F *dva0_HFDepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HF");
  //dva1_HFDepth4->Sumw2();
  //dva0_HFDepth4->Sumw2();
  TH2F *Seff_HFDepth4 = (TH2F *)dva1_HFDepth4->Clone("Seff_HFDepth4");
  Seff_HFDepth4->Divide(dva1_HFDepth4, dva0_HFDepth4, 25., 1., "B");
  gPad->SetGridy();
  gPad->SetGridx();
  //gPad->SetLogz();
  Seff_HFDepth4->SetMarkerStyle(20);
  Seff_HFDepth4->SetMarkerSize(0.4);
  Seff_HFDepth4->SetXTitle("#eta \b");
  Seff_HFDepth4->SetYTitle("#phi \b");
  Seff_HFDepth4->SetZTitle("<timing> HF Depth4 \b");
  Seff_HFDepth4->SetMarkerColor(2);
  Seff_HFDepth4->SetLineColor(2);
  Seff_HFDepth4->SetMaximum(27.);
  Seff_HFDepth4->SetMinimum(23.);
  Seff_HFDepth4->Draw("COLZ");

  c1->Update();

  //========================================================================================== 7
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 1);

  c1->cd(1);
  TH2F *dva1_HODepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4TSmeanA_HO");
  for (int i = 1; i <= dva1_HODepth4->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= dva1_HODepth4->GetYaxis()->GetNbins(); j++) {
      double ccc1 = dva1_HODepth4->GetBinContent(i, j);
      if (ccc1 > 0.)
        std::cout << "******    dva1_HODepth4   **************   i =  " << i << " j =  " << j << " ccc1 =  " << ccc1
                  << std::endl;
    }
  }
  TH2F *dva0_HODepth4 = (TH2F *)dir->FindObjectAny("h_mapDepth4_HO");
  for (int i = 1; i <= dva0_HODepth4->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= dva0_HODepth4->GetYaxis()->GetNbins(); j++) {
      double ccc1 = dva0_HODepth4->GetBinContent(i, j);
      if (ccc1 > 0.)
        std::cout << "******   dva0_HODepth4   **************   i =  " << i << " j =  " << j << " ccc1 =  " << ccc1
                  << std::endl;
    }
  }
  TH2F *Seff_HODepth4 = (TH2F *)dva1_HODepth4->Clone("Seff_HODepth4");
  Seff_HODepth4->Divide(dva1_HODepth4, dva0_HODepth4, 25., 1., "B");
  for (int i = 1; i <= Seff_HODepth4->GetXaxis()->GetNbins(); i++) {
    for (int j = 1; j <= Seff_HODepth4->GetYaxis()->GetNbins(); j++) {
      double ccc1 = Seff_HODepth4->GetBinContent(i, j);
      if (ccc1 > 0.)
        std::cout << "******    Seff_HODepth4   **************   i =  " << i << " j =  " << j << " ccc1 =  " << ccc1
                  << std::endl;
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();
  Seff_HODepth4->SetMarkerStyle(20);
  Seff_HODepth4->SetMarkerSize(0.4);
  Seff_HODepth4->SetXTitle("#eta \b");
  Seff_HODepth4->SetYTitle("#phi \b");
  Seff_HODepth4->SetZTitle("<timing> HB Depth1 \b");
  Seff_HODepth4->SetMarkerColor(2);
  Seff_HODepth4->SetLineColor(2);
  Seff_HODepth4->SetMaximum(115.);
  Seff_HODepth4->SetMinimum(110.);
  Seff_HODepth4->Draw("COLZ");

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
