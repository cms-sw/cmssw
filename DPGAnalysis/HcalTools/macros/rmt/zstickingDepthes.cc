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

  //	TFile *hfile1= new TFile("LED_377804.root", "READ");
  	TFile *hfile1= new TFile("LED_378456.root", "READ");
	//  TFile *hfile1 = new TFile("Global_370580.root", "READ");
//  TFile *hfile1 = new TFile("Global_369927.root", "READ");
  //TFile *hfile1 = new TFile("Global_362365.root", "READ");

  TPostScript psfile("zstickingDepthes.ps", 111);

  //

  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);

  hfile1->ls();
  TDirectory *dir = (TDirectory *)hfile1->FindObjectAny(dirnm.c_str());

  //=============================================================================================== 1
  //======================================================================  RMS: Amplitude Amplitude225
  //======================================================================
  c1->Clear();
  c1->Divide(2, 1);
  //  cout << "HB8-      MaxMil0HB8=     " << MaxMil0HB8 << " MaxMilHB8=     " << MaxMilHB8 << endl;
  
    cout << "00000000000000000000000000    " << endl;

  TH2F *twod1 = (TH2F *)dir->FindObjectAny("h_mapDepth1Amplitude_HB");
  TH2F *twod0 = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  twod1->Sumw2();
  twod0->Sumw2();
  //    if(twod0->IsA()->InheritsFrom("TH2F")){
  TH2F *Ceff = (TH2F *)twod1->Clone("Ceff");
  Ceff->Divide(twod1, twod0, 1, 1, "B");
  Ceff->Sumw2();
  //   }
  c1->cd(1);
    cout << "11111111111111111111111111    " << endl;
  TH2F *twop1 = (TH2F *)dir->FindObjectAny("h_mapDepth1Amplitude225_HB");
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
  Cefz225->SetZTitle("Rate for Width(RMS) in each event & cell out 0.5-2.0 HB Depth1 \b");
  Cefz225->SetMarkerColor(2);
  Cefz225->SetLineColor(2);
  Cefz225->SetMaximum(1.000);
  Cefz225->SetMinimum(0.0001);
  Cefz225->Draw("COLZ");

  c1->cd(2);
    cout << "22222222222222222222222222    " << endl;
  TH1F *aaaaaa1 = (TH1F *)dir->FindObjectAny("h_Amplitude_HB");
  gPad->SetLogy();
  aaaaaa1->SetMarkerStyle(20);
  aaaaaa1->SetMarkerSize(0.8);
  aaaaaa1->GetYaxis()->SetLabelSize(0.04);
  aaaaaa1->SetXTitle("Width in each event & cell HB \b");
  aaaaaa1->SetMarkerColor(2);
  aaaaaa1->SetLineColor(2);
  aaaaaa1->Draw("");


  c1->Update();

  //=============================================================================================== 2
  //====================================================================== Ratio:   Ampl       Ampl047
  //======================================================================
  c1->Clear();
  c1->Divide(2, 1);
  //  cout << "HB8-      MaxMil0HB8=     " << MaxMil0HB8 << " MaxMilHB8=     " << MaxMilHB8 << endl;
  
    cout << "00000000000000000000000000    " << endl;

  TH2F *twod1R = (TH2F *)dir->FindObjectAny("h_mapDepth1Ampl_HB");
  TH2F *twod0R = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  twod1R->Sumw2();
  twod0R->Sumw2();
  //    if(twod0->IsA()->InheritsFrom("TH2F")){
  TH2F *CeffR = (TH2F *)twod1R->Clone("CeffR");
  CeffR->Divide(twod1R, twod0R, 1, 1, "B");
  CeffR->Sumw2();
  //   }
  c1->cd(1);
    cout << "11111111111111111111111111    " << endl;
  TH2F *twop1R = (TH2F *)dir->FindObjectAny("h_mapDepth1Ampl047_HB");
  TH2F *twop0R = (TH2F *)dir->FindObjectAny("h_mapDepth1_HB");
  twop1R->Sumw2();
  twop0R->Sumw2();
  //   if(twop0->IsA()->InheritsFrom("TH2F")){
  TH2F *Cefz225R = (TH2F *)twop1R->Clone("Cefz225R");
  Cefz225R->Divide(twop1R, twop0R, 1, 1, "B");
  Cefz225R->Sumw2();
  //   }
  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogz();
  Cefz225R->SetMarkerStyle(20);
  Cefz225R->SetMarkerSize(0.4);
  //    Cefz225R->GetYaxis()->SetLabelSize(0.04);
  Cefz225R->GetZaxis()->SetLabelSize(0.08);
  Cefz225R->SetXTitle("#eta \b");
  Cefz225R->SetYTitle("#phi \b");
  Cefz225R->SetZTitle("Rate for Ratio in each event & cell out 0.75-1.04 HB Depth1 \b");
  Cefz225R->SetMarkerColor(2);
  Cefz225R->SetLineColor(2);
  Cefz225R->SetMaximum(1.000);
  Cefz225R->SetMinimum(0.0001);
  Cefz225R->Draw("COLZ");

  c1->cd(2);
    cout << "22222222222222222222222222    " << endl;
  TH1F *aaaaaa1R = (TH1F *)dir->FindObjectAny("h_Ampl_HB");
  gPad->SetLogy();
  aaaaaa1R->SetMarkerStyle(20);
  aaaaaa1R->SetMarkerSize(0.8);
  aaaaaa1R->GetYaxis()->SetLabelSize(0.04);
  aaaaaa1R->SetXTitle("Ratio in each event & cell HB \b");
  aaaaaa1R->SetMarkerColor(2);
  aaaaaa1R->SetLineColor(2);
  aaaaaa1R->Draw("");


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
