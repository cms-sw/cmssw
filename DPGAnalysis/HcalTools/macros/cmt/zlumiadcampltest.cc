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
  //======================================================================
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
  TPostScript psfile("zlumiadcampltest.ps", 111);

  //

  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);
  //  TCanvas *c1 = new TCanvas("c1", "Hcal4test", 1000, 500);

  hfile1->ls();
  TDirectory *dir = (TDirectory *)hfile1->FindObjectAny(dirnm.c_str());

  //========================================================================================== 1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  //+++++++++++++++++++++++++++++
  // Lumi iLumi and number of events
  //+++++++++++++++++++++++++++++
  // with TfileService implementation, change everywhere below:     hfile->Get     to     dir->FindObjectAny
  c1->Clear();
  c1->Divide(2, 1);
  c1->cd(1);
  TH1F *LumLum = (TH1F *)dir->FindObjectAny("h_lsnumber_per_eachLS");
  int MaxLumDanila = LumLum->GetBinContent(LumLum->GetMaximumBin());  // old variant of Danila
  cout << " MaxLumDanila=     " << MaxLumDanila << endl;
  //  gPad->SetGridy();
  //  gPad->SetGridx();
  LumLum->SetMarkerStyle(21);
  LumLum->SetMarkerSize(0.8);
  LumLum->GetYaxis()->SetLabelSize(0.04);
  LumLum->SetTitle("Cont. number per LS  \b");
  LumLum->SetXTitle("Cont.number \b");
  LumLum->SetYTitle("Ls \b");
  LumLum->SetMarkerColor(4);
  LumLum->SetLineColor(0);
  LumLum->SetMinimum(0.8);
  LumLum->GetXaxis()->SetRangeUser(0, MaxLumDanila + 5.);
  LumLum->Draw("Error");

  c1->cd(2);
  TH1F *LumiEv = (TH1F *)dir->FindObjectAny("h_nevents_per_eachRealLS");
  int MaxLum0 = LumiEv->GetBinContent(LumiEv->GetMaximumBin());
  int MaxLum = 0;
  for (int i = 1; i <= LumiEv->GetXaxis()->GetNbins(); i++) {
    if (LumiEv->GetBinContent(i)) {
      MaxLum = i;
    }
  }
  cout << " Nev in bin of MaxLum =     " << MaxLum0 << " MaxLum=     " << MaxLum << endl;

  //  gPad->SetGridy();
  //  gPad->SetGridx();
  gPad->SetLogy();
  //            gPad->SetLogx();
  LumiEv->GetYaxis()->SetLabelSize(0.04);
  LumiEv->SetTitle("Number of events per LS");
  LumiEv->SetXTitle("LS");
  LumiEv->SetYTitle("Number of events ");
  LumiEv->SetMarkerStyle(21);
  LumiEv->SetMarkerSize(0.8);
  LumiEv->SetMarkerColor(4);
  LumiEv->SetLineColor(0);
  LumiEv->SetMinimum(0.8);
  LumiEv->GetXaxis()->SetRangeUser(0, MaxLum + 5.);
  LumiEv->Draw("Error");

  c1->Update();
  //  cHB->Print("LumiEvent.png");
  //  cHB->Clear();

  //========================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================

  //      h_lsnumber_per_eachLS->Fill(float(lscounter), float(lumi));
  //        h_nevents_per_eachLS->Fill(float(lscounter), float(nevcounter));  //
  //        h_nls_per_run->Fill(float(lscounterrun));
  //        h_nevents_per_eachRealLS->Fill(float(lscounterM1), float(nevcounter));  //
  //      h_nevents_per_LS->Fill(float(nevcounter));
  //      h_nevents_per_LSzoom->Fill(float(nevcounter));
  //        h_nls_per_run10->Fill(float(lscounterrun10));

  c1->Clear();
  c1->Divide(2, 4);

  c1->cd(1);
  TH1F *MilMil = (TH1F *)dir->FindObjectAny("h_lsnumber_per_eachLS");
  int MaxMilDanila = MilMil->GetBinContent(MilMil->GetMaximumBin());  // old variant of Danila
  cout << " MaxMilDanila=     " << MaxMilDanila << endl;
  //    gPad->SetGridy();
  //    gPad->SetGridx();
  MilMil->GetYaxis()->SetLabelSize(0.04);
  MilMil->SetTitle("Cont. number per LS  \b");
  MilMil->SetXTitle("Cont.number \b");
  MilMil->SetYTitle("Ls \b");
  MilMil->SetMarkerStyle(20);
  MilMil->SetMarkerSize(0.2);
  MilMil->SetMarkerColor(4);
  MilMil->SetLineColor(0);
  MilMil->SetMinimum(0.8);
  MilMil->GetXaxis()->SetRangeUser(0, MaxMilDanila);
  MilMil->Draw("Error");

  c1->cd(2);
  TH1F *MiliEv = (TH1F *)dir->FindObjectAny("h_nevents_per_eachLS");
  int MaxMil0 = MiliEv->GetBinContent(MiliEv->GetMaximumBin());
  int MaxMil = 0;
  for (int i = 1; i <= MiliEv->GetXaxis()->GetNbins(); i++) {
    if (MiliEv->GetBinContent(i)) {
      MaxMil = i;
    }
  }
  cout << " MaxMil0=     " << MaxMil0 << " MaxMil=     " << MaxMil << endl;

  //    gPad->SetGridy();
  //    gPad->SetGridx();
  gPad->SetLogy();
  //            gPad->SetLogx();
  MiliEv->GetYaxis()->SetLabelSize(0.04);
  MiliEv->SetTitle("Number of events per LS");
  MiliEv->SetXTitle("LS");
  MiliEv->SetYTitle("Number of events ");
  MiliEv->SetMarkerStyle(20);
  MiliEv->SetMarkerSize(0.2);
  MiliEv->SetMarkerColor(4);
  MiliEv->SetLineColor(0);
  //      MiliEv->SetMinimum(0.8);
  MiliEv->GetXaxis()->SetRangeUser(0, MaxMil);
  MiliEv->Draw("Error");

  c1->cd(3);
  TH1F *GiriEv = (TH1F *)dir->FindObjectAny("h_nls_per_run");
  int MaxGir0 = GiriEv->GetBinContent(GiriEv->GetMaximumBin());
  int MaxGir = 0;
  for (int i = 1; i <= GiriEv->GetXaxis()->GetNbins(); i++) {
    if (GiriEv->GetBinContent(i)) {
      MaxGir = i;
    }
  }
  cout << "nls_per_run =     " << MaxGir0 << " Maxnls_per_run=     " << MaxGir << endl;

  //    gPad->SetGridy();
  //    gPad->SetGridx();
  //  gPad->SetLogy();
  //            gPad->SetLogx();
  GiriEv->GetYaxis()->SetLabelSize(0.04);
  GiriEv->SetTitle("Number of LS per run");
  GiriEv->SetXTitle("irun");
  GiriEv->SetYTitle("Number of LS ");
  GiriEv->SetMarkerStyle(20);
  GiriEv->SetMarkerSize(0.8);
  GiriEv->SetMarkerColor(4);
  GiriEv->SetLineColor(0);
  //      GiriEv->SetMinimum(0.8);
  GiriEv->GetXaxis()->SetRangeUser(0, MaxGir);
  GiriEv->Draw("Error");

  c1->cd(4);
  TH1F *SumiEv = (TH1F *)dir->FindObjectAny("h_nevents_per_eachRealLS");
  int MaxSum0 = SumiEv->GetBinContent(SumiEv->GetMaximumBin());
  int MaxSum = 0;
  for (int i = 1; i <= SumiEv->GetXaxis()->GetNbins(); i++) {
    if (SumiEv->GetBinContent(i)) {
      MaxSum = i;
    }
  }
  cout << " MaxSum0=     " << MaxSum0 << " MaxSum=     " << MaxSum << endl;

  gPad->SetGridy();
  gPad->SetGridx();
  gPad->SetLogy();
  //            gPad->SetLogx();
  SumiEv->GetYaxis()->SetLabelSize(0.04);
  SumiEv->SetTitle("Number of events per RealLS");
  SumiEv->SetXTitle("LS");
  SumiEv->SetYTitle("Number of events ");
  SumiEv->SetMarkerStyle(20);
  SumiEv->SetMarkerSize(0.2);
  SumiEv->SetMarkerColor(4);
  SumiEv->SetLineColor(0);
  //      SumiEv->SetMinimum(0.8);
  SumiEv->GetXaxis()->SetRangeUser(0, MaxSum);
  SumiEv->Draw("Error");

  c1->cd(5);
  TH1F *TiriEv = (TH1F *)dir->FindObjectAny("h_nevents_per_LS");
  int MaxTir0 = TiriEv->GetBinContent(TiriEv->GetMaximumBin());
  int MaxTir = 0;
  for (int i = 1; i <= TiriEv->GetXaxis()->GetNbins(); i++) {
    if (TiriEv->GetBinContent(i)) {
      MaxTir = i;
    }
  }
  cout << " MaxTir0=     " << MaxTir0 << " MaxTir=     " << MaxTir << endl;

  TiriEv->GetYaxis()->SetLabelSize(0.04);
  TiriEv->SetTitle("Number of events per LS");
  TiriEv->SetXTitle("LS");
  TiriEv->SetYTitle("Number of events ");
  TiriEv->SetMarkerStyle(20);
  TiriEv->SetMarkerSize(0.8);
  TiriEv->SetMarkerColor(4);
  TiriEv->SetLineColor(0);
  //      TiriEv->SetMinimum(0.8);
  TiriEv->GetXaxis()->SetRangeUser(0, MaxTir);
  TiriEv->Draw("Error");

  c1->cd(6);
  TH1F *MasiEv = (TH1F *)dir->FindObjectAny("h_nevents_per_LSzoom");
  int MaxMas0 = MasiEv->GetBinContent(MasiEv->GetMaximumBin());
  int MaxMas = 0;
  for (int i = 1; i <= MasiEv->GetXaxis()->GetNbins(); i++) {
    if (MasiEv->GetBinContent(i)) {
      MaxMas = i;
    }
  }
  cout << " MaxMas0=     " << MaxMas0 << " MaxMas=     " << MaxMas << endl;

  MasiEv->GetYaxis()->SetLabelSize(0.04);
  MasiEv->SetTitle("Number of events per LS");
  MasiEv->SetXTitle("LS");
  MasiEv->SetYTitle("Number of events ");
  MasiEv->SetMarkerStyle(20);
  MasiEv->SetMarkerSize(0.8);
  MasiEv->SetMarkerColor(4);
  MasiEv->SetLineColor(0);
  //      MasiEv->SetMinimum(0.8);
  MasiEv->GetXaxis()->SetRangeUser(0, MaxMas);
  MasiEv->Draw("Error");

  c1->cd(7);
  TH1F *LediEv = (TH1F *)dir->FindObjectAny("h_nls_per_run10");
  int MaxLed0 = LediEv->GetBinContent(LediEv->GetMaximumBin());
  int MaxLed = 0;
  for (int i = 1; i <= LediEv->GetXaxis()->GetNbins(); i++) {
    if (LediEv->GetBinContent(i)) {
      MaxLed = i;
    }
  }
  cout << " NlsPERrun=     " << MaxLed0 << " MaxbinHisto=     " << MaxLed << endl;

  LediEv->GetYaxis()->SetLabelSize(0.04);
  LediEv->SetTitle("Number of ls(ev>10) per run");
  LediEv->SetXTitle("run");
  LediEv->SetYTitle("Number of ls ");
  LediEv->SetMarkerStyle(20);
  LediEv->SetMarkerSize(0.8);
  LediEv->SetMarkerColor(4);
  LediEv->SetLineColor(0);
  //      LediEv->SetMinimum(0.8);
  LediEv->GetXaxis()->SetRangeUser(0, MaxLed);
  LediEv->Draw("Error");

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
