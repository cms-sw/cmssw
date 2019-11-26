#include <iostream>
#include <fstream>
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
#include "TH1.h"
#include "TH2.h"
#include "TCanvas.h"

//
using namespace std;
//
//

//inline void HERE(const char *msg) { std::cout << msg << std::endl; }

int main(int argc, char* argv[]) {
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
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////// normal gap 321177:
  //	TFile *hfile1= new TFile("Global_321177_41.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600.root_no41", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_321177_ls1to600.root", "READ");

  //////////////////////////////////////////////////////////////////////////////// normal gap 325001:
  //	TFile *hfile1= new TFile("Global_325001_ls1to600.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_325001_40.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_325001_ls1to600.root", "READ");

  //	TFile *hfile1= new TFile("Global_RBX_325001.root", "READ");

  //	TFile *hfile1= new TFile("Global_321624_1.root", "READ");
  //	TFile *hfile1= new TFile("Global_321625.root", "READ");
  //	TFile *hfile1= new TFile("Global_321313.root", "READ");

  //////////////////////////////////////////////////////////////////////////////// abort gap:
  //	TFile *hfile1= new TFile("Global_321177_41_abortgap.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600_abortgap.root", "READ");
  //	TFile *hfile1= new TFile("Global_321177_ls1to600_abortgap_no41.root", "READ");
  //	TFile *hfile1= new TFile("Global_325001_ls1to600_abortgap.root", "READ");

  TFile* hfile1 = new TFile("Global_RBX_325001.root", "READ");
  //	TFile *hfile1= new TFile("Global_RBX_321177.root", "READ");

  //	TFile *hfile1= new TFile("Global_321758.root", "READ");
  //	TFile *hfile1= new TFile("Global_321773.root", "READ");
  //	TFile *hfile1= new TFile("Global_321774.root", "READ");
  //	TFile *hfile1= new TFile("Global_321775.root", "READ");

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //    getchar();
  //
  TPostScript psfile("zlsgainstabilityCMT.ps", 111);
  //
  TCanvas* c1 = new TCanvas("c1", "Hcal4test", 200, 10, 700, 900);
  //========================================================================================== 1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 2);

  c1->cd(1);
  TH1F* Rate1 = (TH1F*)hfile1->Get("h_nevents_per_eachRealLS");

  //      gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  //      Rate1->SetXTitle("nevents_per_eachRealLS \b");
  //      Rate1->Draw("");

  int maxbins = 0;
  int nx = Rate1->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Rate1->GetBinContent(i);
    if (ccc1 > 0.) {
      maxbins = i;
      if (i > maxbins)
        maxbins = i;
    }
    //	if(ccc1>0.) cout<<" ibin=     "<< i <<" nevents=     "<< ccc1 <<endl;
  }
  cout << "111 maxbins=     " << maxbins << endl;
  /////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH1F* ADCAmplperLS = new TH1F("ADCAmplperLS", "", maxbins, 1., maxbins + 1.);
  //          TH1F* ADCAmplperLS  = new TH1F("ADCAmplperLS ","", 600, 1.,601.);
  nx = Rate1->GetXaxis()->GetNbins();
  for (int i = 1; i <= maxbins; i++) {
    double ccc1 = Rate1->GetBinContent(i);
    //	  if(ccc1>0.)	  cout<<" depth1_HB iLS = "<<i<<" <As> per LS= "<<ccc1<<endl;
    //	cout<<" ibin=     "<< i <<" nevents=     "<< ccc1 <<endl;
    //	  if(ccc1>0.) ADCAmplperLS ->Fill(float(i), ccc1);
    ADCAmplperLS->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  ADCAmplperLS->SetMarkerStyle(20);
  ADCAmplperLS->SetMarkerSize(0.4);
  ADCAmplperLS->GetYaxis()->SetLabelSize(0.04);
  ADCAmplperLS->SetXTitle("nevents_per_eachRealLS \b");
  ADCAmplperLS->SetMarkerColor(2);
  ADCAmplperLS->SetLineColor(0);
  //          ADCAmplperLS ->SetMaximum(30.0);
  //          ADCAmplperLS ->SetMinimum(20.0);
  ADCAmplperLS->Draw("Error");

  c1->cd(2);

  TH1F* Rate2 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_lscounterM1");  // 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  //      TH1F *Rate2= (TH1F*)hfile1->Get("h_nevents_per_eachRealLS");// norm: no 0
  //    TH1F *Rate2= (TH1F*)hfile1->Get("h_sumADCAmplperLS1");//Fill( float(lscounterM1) ,bbbc);// norm: no 0
  //     TH1F *Rate2= (TH1F*)hfile1->Get("h_sum0ADCAmplperLS2");//Fill( float(lscounterM1) ,bbb1);// norm: no 0

  TH1F* Rate2redone = new TH1F("Rate2redone", "", maxbins, 1., maxbins + 1.);
  //    for (int i=1;i<=Rate2->GetXaxis()->GetNbins();i++) {
  for (int i = 1; i <= maxbins; i++) {
    double ccc1 = Rate2->GetBinContent(i);
    //if(ccc1 <= 0. )	    	    cout<<"1 Page  i=  "<< i <<"      A= "<< ccc1 <<endl;
    //	cout<<"1 Page  i=  "<< i <<"      A= "<< ccc1 <<endl;
    Rate2redone->Fill(float(i), ccc1);
    //	if(ccc1>0.) Rate2redone ->Fill(float(i), ccc1);
  }
  Rate2redone->SetMarkerStyle(20);
  Rate2redone->SetMarkerSize(0.4);
  Rate2redone->GetYaxis()->SetLabelSize(0.04);
  Rate2redone->SetXTitle("sumADCAmplEtaPhiLsS \b");
  Rate2redone->SetMarkerColor(2);
  Rate2redone->SetLineColor(0);
  // Rate2redone ->SetMaximum(30.0);Rate2redone ->SetMinimum(20.0);
  Rate2redone->Draw("Error");

  c1->Update();

  //========================================================================================== 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH1F* Rate3 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_bbbc");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate3->SetXTitle("bbbc sumADCAmplEtaPhiLs \b");
  Rate3->Draw("");

  c1->cd(2);
  TH1F* Rate4 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_bbb1");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate4->SetXTitle("bbb1 sumADCAmplEtaPhiLs \b");
  Rate4->Draw("");

  c1->cd(3);
  TH1F* Rate5 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate5->SetXTitle("bbb3 sumADCAmplEtaPhiLs \b");
  Rate5->Draw("");

  c1->cd(4);
  TH1F* Rate6 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_ietaphi");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate6->SetXTitle("ietaphi sumADCAmplEtaPhiLs \b");
  Rate6->Draw("");

  c1->Update();
  //========================================================================================== 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 1);
  //  h_2DsumADCAmplEtaPhiLs00->Fill(float(lscounterM1), float(ietaphi), bbb1);//HB
  //     h_2D0sumADCAmplLS1->Fill(double(ieta), double(k3), bbb1);
  c1->cd(1);
  TH2F* Cefz2 = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs0");
  //TH2F *Cefz2= (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs00");
  //    TH2F *Cefz2= (TH2F*)hfile1->Get("h_2D0sumADCAmplLS1");
  gPad->SetGridy();
  gPad->SetGridx();
  //      gPad->SetLogz();
  Cefz2->SetMarkerStyle(20);
  Cefz2->SetMarkerSize(0.4);
  //    Cefz2->GetYaxis()->SetLabelSize(0.04);
  Cefz2->GetZaxis()->SetLabelSize(0.08);
  Cefz2->SetXTitle("nv0-overAllLSs test with  HB1  #eta \b");
  Cefz2->SetYTitle("#phi  \b");
  //    Cefz2->SetZTitle("<A>  - HB Depth1 \b");
  Cefz2->SetMarkerColor(2);
  Cefz2->SetLineColor(2);
  //      Cefz2->SetMaximum(1.000);
  //      Cefz2->SetMinimum(1.0);
  Cefz2->Draw("COLZ");

  c1->Update();
  //======================================================================
  //======================================================================
  //========================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 1);

  c1->cd(1);
  int maxbinx = 0;
  int maxbiny = 0;
  nx = Cefz2->GetXaxis()->GetNbins();
  nx = maxbins;
  int ny = Cefz2->GetYaxis()->GetNbins();
  cout << "444 HB1        nx=     " << nx << " ny=     " << ny << endl;
  //    TH1F* ADCAmplLSHB1 = new TH1F("ADCAmplLSHB1","", 1000, 0., 1000000.);
  //    TH1F* ADCAmplLSHB1 = new TH1F("ADCAmplLSHB1","", 1000, 0., 1000.);
  TH2F* ADCAmplLSHB1 = new TH2F("ADCAmplLSHB1", "", 610, 0., 610., 160, 120., 280.);
  for (int i = 1; i <= nx; i++) {
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz2->GetBinContent(i, j);
      //	  if(ccc1>0.) {
      if (j > 130 && j < 270) {
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	    if(ccc1 <= 0. )	    	    cout<<"HB1:  ibin=  "<< i <<"      jbin= "<< j <<"      A= "<< ccc1 <<endl;
        //	    ADCAmplLSHB1 ->Fill(ccc1);
        ADCAmplLSHB1->Fill(float(i), float(j), ccc1);
      }  // if
    }
  }
  ADCAmplLSHB1->SetMarkerStyle(20);
  ADCAmplLSHB1->SetMarkerSize(0.4);
  ADCAmplLSHB1->GetYaxis()->SetLabelSize(0.04);
  ADCAmplLSHB1->SetXTitle("nev0-overAllLSs test with ADCAmplLSHB1 \b");
  ADCAmplLSHB1->SetMarkerColor(2);
  ADCAmplLSHB1->SetLineColor(0);
  //          ADCAmplLSHB1 ->SetMaximum(30.0);
  //          ADCAmplLSHB1 ->SetMinimum(20.0);
  //    gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  ADCAmplLSHB1->Draw("COLZ");
  //          ADCAmplLSHB1 ->Draw("Error");
  //   ADCAmplLSHB1 ->Draw("");
  cout << "444 HB1 for h_2D0sumADCAmplLS1 maxbinx =  " << maxbinx << "     maxbiny=  " << maxbiny << endl;
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=72; nphi=82;

  c1->Update();

  //========================================================================================== 5
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 2);

  TH1F* ATIT1 = (TH1F*)hfile1->Get("h_bcnvsamplitude_HB");
  TH1F* ATIT2 = (TH1F*)hfile1->Get("h_bcnvsamplitude0_HB");
  int minbx = 999999;
  int maxbx = -1;
  nx = ATIT2->GetXaxis()->GetNbins();
  for (int i = 0; i <= nx; i++) {
    //	i-=1;
    double ccc1 = ATIT2->GetBinContent(i);
    if (ccc1 > 0.) {
      if (i > maxbx)
        maxbx = i;
      if (i < minbx)
        minbx = i;
      if (i >= 3440 && i <= 3570)
        cout << "Page5: i = =     " << i - 1 << " Ni=     " << ccc1 << endl;
    }
  }
  cout << "Page5: minbx=     " << minbx - 1 << " maxbx=     " << maxbx - 1 << endl;

  c1->cd(1);
  TH1F* ITIT1 = new TH1F("ITIT1", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  for (int i = 1; i <= nx; i++) {
    double ccc1 = ATIT1->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" bcnvsamplitude_HB ;  i = "<<i<<" ccc1= "<<ccc1<<endl;
    if (ccc1 > 0.)
      ITIT1->Fill(float(i), ccc1);
  }
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  ITIT1->SetXTitle("bcnvsamplitude_HB \b");
  ITIT1->SetMarkerStyle(20);
  ITIT1->SetMarkerSize(0.4);
  ITIT1->GetYaxis()->SetLabelSize(0.04);
  ITIT1->SetMarkerColor(2);
  ITIT1->SetLineColor(0);  // ITIT1->SetMaximum(30.0);// ITIT1->SetMinimum(20.0);
  ITIT1->Draw("Error");

  c1->cd(2);
  TH1F* ITIT2 = new TH1F("ITIT2", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  for (int i = 1; i <= nx; i++) {
    double ccc1 = ATIT2->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" bcnvsamplitude_HB ;  i = "<<i<<" ccc1= "<<ccc1<<endl;
    if (ccc1 > 0.)
      ITIT2->Fill(float(i), ccc1);
  }
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  ITIT2->SetXTitle("bcnvsamplitude0 HBs \b");
  ITIT2->SetMarkerStyle(20);
  ITIT2->SetMarkerSize(0.4);
  ITIT2->GetYaxis()->SetLabelSize(0.04);
  ITIT2->SetMarkerColor(2);
  ITIT2->SetLineColor(0);  // ITIT2->SetMaximum(30.0);// ITIT2->SetMinimum(20.0);
  ITIT2->Draw("Error");

  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////

  //======================================================================
  //========================================================================================== 6
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 4);

  c1->cd(1);
  TH1F* twrd3 = (TH1F*)hfile1->Get("h_bcnvsamplitude_HB");
  TH1F* twrd2 = (TH1F*)hfile1->Get("h_bcnvsamplitude0_HB");
  twrd3->Sumw2();
  twrd2->Sumw2();
  gPad->SetGridy();
  TH1F* Rase1 = (TH1F*)twrd3->Clone("Rase1");
  Rase1->Divide(twrd3, twrd2, 1, 1, "B");
  Rase1->Sumw2();
  TH1F* Rase1NNN = new TH1F("Rase1NNN", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  nx = Rase1->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Rase1->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HB i = "<<i<<" <A> per bx= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Rase1NNN->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Rase1NNN->SetMarkerStyle(20);
  Rase1NNN->SetMarkerSize(0.4);
  Rase1NNN->GetYaxis()->SetLabelSize(0.04);
  Rase1NNN->SetXTitle("<ADCAmpl> per bx HB \b");
  Rase1NNN->SetMarkerColor(2);
  Rase1NNN->SetLineColor(0);  //Rase1NNN->SetMaximum(30.0);//Rase1NNN->SetMinimum(20.0);
  Rase1NNN->Draw("Error");

  c1->cd(2);
  TH1F* twed3 = (TH1F*)hfile1->Get("h_bcnvsamplitude_HE");
  TH1F* twed2 = (TH1F*)hfile1->Get("h_bcnvsamplitude0_HE");
  twed3->Sumw2();
  twed2->Sumw2();
  gPad->SetGridy();
  TH1F* Rase2 = (TH1F*)twed3->Clone("Rase2");
  Rase2->Divide(twed3, twed2, 1, 1, "B");
  Rase2->Sumw2();
  TH1F* Rase2NNN = new TH1F("Rase2NNN", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  nx = Rase2->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Rase2->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HE i = "<<i<<" <A> per bx= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Rase2NNN->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Rase2NNN->SetMarkerStyle(20);
  Rase2NNN->SetMarkerSize(0.4);
  Rase2NNN->GetYaxis()->SetLabelSize(0.04);
  Rase2NNN->SetXTitle("<ADCAmpl> per bx HE \b");
  Rase2NNN->SetMarkerColor(2);
  Rase2NNN->SetLineColor(0);  //Rase2NNN->SetMaximum(30.0);//Rase2NNN->SetMinimum(20.0);
  Rase2NNN->Draw("Error");

  c1->cd(3);
  TH1F* twwd3 = (TH1F*)hfile1->Get("h_bcnvsamplitude_HF");
  TH1F* twwd2 = (TH1F*)hfile1->Get("h_bcnvsamplitude0_HF");
  twwd3->Sumw2();
  twwd2->Sumw2();
  gPad->SetGridy();
  TH1F* Rase3 = (TH1F*)twwd3->Clone("Rase3");
  Rase3->Divide(twwd3, twwd2, 1, 1, "B");
  Rase3->Sumw2();
  TH1F* Rase3NNN = new TH1F("Rase3NNN", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  nx = Rase3->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Rase3->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HF i = "<<i<<" <A> per bx= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Rase3NNN->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Rase3NNN->SetMarkerStyle(20);
  Rase3NNN->SetMarkerSize(0.4);
  Rase3NNN->GetYaxis()->SetLabelSize(0.04);
  Rase3NNN->SetXTitle("<ADCAmpl> per bx HF \b");
  Rase3NNN->SetMarkerColor(2);
  Rase3NNN->SetLineColor(0);  //Rase3NNN->SetMaximum(30.0);//Rase3NNN->SetMinimum(20.0);
  Rase3NNN->Draw("Error");

  c1->cd(4);
  TH1F* twqd3 = (TH1F*)hfile1->Get("h_bcnvsamplitude_HO");
  TH1F* twqd2 = (TH1F*)hfile1->Get("h_bcnvsamplitude0_HO");
  twqd3->Sumw2();
  twqd2->Sumw2();
  gPad->SetGridy();
  TH1F* Rase4 = (TH1F*)twqd3->Clone("Rase4");
  Rase4->Divide(twqd3, twqd2, 1, 1, "B");
  Rase4->Sumw2();
  TH1F* Rase4NNN = new TH1F("Rase4NNN", "", maxbx - minbx + 1, float(minbx), maxbx + 1.);
  nx = Rase4->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Rase4->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HO i = "<<i<<" <A> per bx= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Rase4NNN->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Rase4NNN->SetMarkerStyle(20);
  Rase4NNN->SetMarkerSize(0.4);
  Rase4NNN->GetYaxis()->SetLabelSize(0.04);
  Rase4NNN->SetXTitle("<ADCAmpl> per bx HO \b");
  Rase4NNN->SetMarkerColor(2);
  Rase4NNN->SetLineColor(0);  //Rase4NNN->SetMaximum(30.0);//Rase4NNN->SetMinimum(20.0);
  Rase4NNN->Draw("Error");

  c1->Update();
  //========================================================================================== 7
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  c1->cd(1);
  TH1F* Rate7 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_orbitNum");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate7->SetXTitle("h_sumADCAmplEtaPhiLs_orbitNum \b");
  Rate7->Draw("");

  c1->cd(2);
  TH1F* Rate8 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_lscounterM1");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate8->SetXTitle("h_sumADCAmplEtaPhiLs_lscounterM1 \b");
  Rate8->Draw("");

  c1->cd(3);
  TH1F* Rate9 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_lscounterM1orbitNum");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Rate9->SetXTitle("h_sumADCAmplEtaPhiLs_lscounterM1orbitNum \b");
  Rate9->Draw("");

  c1->Update();
  //======================================================================
  //========================================================================================== 8
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  TH1F* ASSS1 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_lscounterM1orbitNum");
  TH1F* ASSS2 = (TH1F*)hfile1->Get("h_sumADCAmplEtaPhiLs_lscounterM1");
  int minls = 999999;
  int maxls = -1;
  nx = ASSS2->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = ASSS2->GetBinContent(i);
    if (ccc1 > 0.) {
      if (i > maxls)
        maxls = i;
      if (i < minls)
        minls = i;
    }
    //	  if(ccc1>0.)	  cout<<" ASSS2 ;  i = "<<i<<" ccc1= "<<ccc1<<endl;
    //	if(ccc1>0.) {maxls = i; if(i>maxls) maxls = i;}
  }
  cout << "Page8: minls=     " << minls << " maxls=     " << maxls << endl;
  //////////////////////////////////////////////////////////////////////////////////////
  c1->cd(1);
  TH1F* ISSS1 = new TH1F("ISSS1", "", maxls - minls + 1, float(minls), maxls + 1.);
  for (int i = 0; i <= nx; i++) {
    double ccc1 = ASSS1->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" bcnvsamplitude_HB ;  i = "<<i<<" ccc1= "<<ccc1<<endl;
    if (ccc1 > 0.)
      ISSS1->Fill(float(i), ccc1);
  }
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  ISSS1->SetXTitle("lscounterM1 w = orbitNum*Nentries \b");
  ISSS1->SetMarkerStyle(20);
  ISSS1->SetMarkerSize(0.4);
  ISSS1->GetYaxis()->SetLabelSize(0.04);
  ISSS1->SetMarkerColor(2);
  ISSS1->SetLineColor(0);  // ISSS1->SetMaximum(30.0);// ISSS1->SetMinimum(20.0);
  ISSS1->Draw("Error");
  //////////////////////////////////////////////////////////////////////////////////////
  c1->cd(2);
  TH1F* ISSS2 = new TH1F("ISSS2", "", maxls - minls + 1, float(minls), maxls + 1.);
  for (int i = 0; i <= nx; i++) {
    double ccc1 = ASSS2->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" bcnvsamplitude_HB ;  i = "<<i<<" ccc1= "<<ccc1<<endl;
    if (ccc1 > 0.)
      ISSS2->Fill(float(i), ccc1);
  }
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  ISSS2->SetXTitle("lscounterM1 w = Nentries \b");
  ISSS2->SetMarkerStyle(20);
  ISSS2->SetMarkerSize(0.4);
  ISSS2->GetYaxis()->SetLabelSize(0.04);
  ISSS2->SetMarkerColor(2);
  ISSS2->SetLineColor(0);  // ISSS2->SetMaximum(30.0);// ISSS2->SetMinimum(20.0);
  ISSS2->Draw("Error");
  //////////////////////////////////////////////////////////////////////////////////////
  c1->cd(3);
  TH1F* Roze1 = (TH1F*)ASSS2->Clone("Roze1");
  Roze1->Divide(ASSS1, ASSS2, 1, 1, "B");
  Roze1->Sumw2();
  TH1F* Roze1NNN = new TH1F("Roze1NNN", "", maxls - minls + 1, float(minls), maxls + 1.);
  nx = Roze1->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Roze1->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HB i = "<<i<<" <A> per ls= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Roze1NNN->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Roze1NNN->SetMarkerStyle(20);
  Roze1NNN->SetMarkerSize(0.4);
  Roze1NNN->GetYaxis()->SetLabelSize(0.04);
  Roze1NNN->SetXTitle("lscounterM1 w = <orbitNum> \b");
  Roze1NNN->SetMarkerColor(2);
  Roze1NNN->SetLineColor(0);  //Roze1NNN->SetMaximum(30.0);//Roze1NNN->SetMinimum(20.0);
  Roze1NNN->Draw("Error");

  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////

  //========================================================================================== 9
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  c1->cd(1);
  TH1F* TEST7 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude_HB");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  TEST7->SetXTitle("h_orbitNumvsamplitude_HB \b");
  TEST7->Draw("");

  c1->cd(2);
  TH1F* TEST8 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude0_HB");
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  TEST8->SetXTitle("h_orbitNumvsamplitude0_HB \b");
  TEST8->Draw("");

  c1->cd(3);
  TH1F* TEST9 = (TH1F*)TEST8->Clone("TEST9");
  TEST9->Divide(TEST7, TEST8, 1, 1, "B");
  //    TH1F* TEST9 = new TH1F("TEST9","", zaP, zaR, zaR2);
  //      TH1F* TEST9 = new TH1F("TEST9","", maxorbitNum-minorbitNum+1, float(minorbitNum), maxorbitNum+1.);
  /*
      nx =TEST9->GetXaxis()->GetNbins();
      for (int i=1;i<=nx;i++) {
	double ccc1 =  TEST9->GetBinContent(i);
		if(ccc1>0.)	  cout<<" HB i = "<<i<<" <A> per orbitNum= "<<ccc1<<endl;
		//	  if(ccc1>0.) TEST9->Fill(float(i), ccc1);
      }
*/
  //      gPad->SetLogy();
  TEST9->SetMarkerStyle(20);
  TEST9->SetMarkerSize(0.4);
  TEST9->GetYaxis()->SetLabelSize(0.04);
  TEST9->SetXTitle("<ADCAmpl> per orbitNum HB \b");
  TEST9->SetMarkerColor(2);
  TEST9->SetLineColor(0);  //TEST9->SetMaximum(30.0);//TEST9->SetMinimum(20.0);
  TEST9->Draw("Error");

  c1->Update();
  //======================================================================
  //========================================================================================== 10
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 4);

  c1->cd(1);
  TH1F* kqrd3 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude_HB");
  TH1F* kqrd2 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude0_HB");
  kqrd3->Sumw2();
  kqrd2->Sumw2();
  gPad->SetGridy();
  TH1F* Posw1 = (TH1F*)kqrd3->Clone("Posw1");
  Posw1->Divide(kqrd3, kqrd2, 1, 1, "B");
  Posw1->Sumw2();
  //    TH1F* Posw1 = new TH1F("Posw1","", zaP, zaR, zaR2);
  //      TH1F* Posw1 = new TH1F("Posw1","", maxorbitNum-minorbitNum+1, float(minorbitNum), maxorbitNum+1.);
  nx = Posw1->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Posw1->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HB i = "<<i<<" <A> per orbitNum= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Posw1->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Posw1->SetMarkerStyle(20);
  Posw1->SetMarkerSize(0.4);
  Posw1->GetYaxis()->SetLabelSize(0.04);
  Posw1->SetXTitle("<ADCAmpl> per orbitNum HB \b");
  Posw1->SetMarkerColor(2);
  Posw1->SetLineColor(0);  //Posw1->SetMaximum(30.0);//Posw1->SetMinimum(20.0);
  Posw1->Draw("Error");

  c1->cd(2);
  TH1F* kqed3 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude_HE");
  TH1F* kqed2 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude0_HE");
  kqed3->Sumw2();
  kqed2->Sumw2();
  gPad->SetGridy();
  TH1F* Posw2 = (TH1F*)kqed3->Clone("Posw2");
  Posw2->Divide(kqed3, kqed2, 1, 1, "B");
  Posw2->Sumw2();
  //    TH1F* Posw2 = new TH1F("Posw2","", zaP, zaR, zaR2);
  //      TH1F* Posw2 = new TH1F("Posw2","", maxorbitNum-minorbitNum+1, float(minorbitNum), maxorbitNum+1.);
  nx = Posw2->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Posw2->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HE i = "<<i<<" <A> per orbitNum= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Posw2->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Posw2->SetMarkerStyle(20);
  Posw2->SetMarkerSize(0.4);
  Posw2->GetYaxis()->SetLabelSize(0.04);
  Posw2->SetXTitle("<ADCAmpl> per orbitNum HE \b");
  Posw2->SetMarkerColor(2);
  Posw2->SetLineColor(0);  //Posw2->SetMaximum(30.0);//Posw2->SetMinimum(20.0);
  Posw2->Draw("Error");

  c1->cd(3);
  TH1F* kqwd3 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude_HF");
  TH1F* kqwd2 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude0_HF");
  kqwd3->Sumw2();
  kqwd2->Sumw2();
  gPad->SetGridy();
  TH1F* Posw3 = (TH1F*)kqwd3->Clone("Posw3");
  Posw3->Divide(kqwd3, kqwd2, 1, 1, "B");
  Posw3->Sumw2();
  //    TH1F* Posw3 = new TH1F("Posw3","", zaP, zaR, zaR2);
  //      TH1F* Posw3 = new TH1F("Posw3","", maxorbitNum-minorbitNum+1, float(minorbitNum), maxorbitNum+1.);
  nx = Posw3->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Posw3->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HF i = "<<i<<" <A> per orbitNum= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Posw3->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Posw3->SetMarkerStyle(20);
  Posw3->SetMarkerSize(0.4);
  Posw3->GetYaxis()->SetLabelSize(0.04);
  Posw3->SetXTitle("<ADCAmpl> per orbitNum HF \b");
  Posw3->SetMarkerColor(2);
  Posw3->SetLineColor(0);  //Posw3->SetMaximum(30.0);//Posw3->SetMinimum(20.0);
  Posw3->Draw("Error");

  c1->cd(4);
  TH1F* kqqd3 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude_HO");
  TH1F* kqqd2 = (TH1F*)hfile1->Get("h_orbitNumvsamplitude0_HO");
  kqqd3->Sumw2();
  kqqd2->Sumw2();
  gPad->SetGridy();
  TH1F* Posw4 = (TH1F*)kqqd3->Clone("Posw4");
  Posw4->Divide(kqqd3, kqqd2, 1, 1, "B");
  Posw4->Sumw2();
  //    TH1F* Posw4 = new TH1F("Posw4","", zaP, zaR, zaR2);
  //    TH1F* Posw4 = new TH1F("Posw4","", maxorbitNum-minorbitNum+1, float(minorbitNum), maxorbitNum+1.);
  nx = Posw4->GetXaxis()->GetNbins();
  for (int i = 1; i <= nx; i++) {
    double ccc1 = Posw4->GetBinContent(i);
    //	if(ccc1>0.)	  cout<<" HO i = "<<i<<" <A> per orbitNum= "<<ccc1<<endl;
    if (ccc1 > 0.)
      Posw4->Fill(float(i), ccc1);
  }
  //      gPad->SetLogy();
  Posw4->SetMarkerStyle(20);
  Posw4->SetMarkerSize(0.4);
  Posw4->GetYaxis()->SetLabelSize(0.04);
  Posw4->SetXTitle("<ADCAmpl> per orbitNum HO \b");
  Posw4->SetMarkerColor(2);
  Posw4->SetLineColor(0);  //Posw4->SetMaximum(30.0);//Posw4->SetMinimum(20.0);
  Posw4->Draw("Error");

  c1->Update();

  //========================================================================================== 11    HB - "h_2DsumADCAmplEtaPhiLs0
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH2F* Cefz1KKK = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs0");
  TH2F* Cefz1LLL = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs00");
  TH2F* Cefz1 = (TH2F*)Cefz1LLL->Clone("Cefz1");
  Cefz1->Divide(Cefz1KKK, Cefz1LLL, 1, 1, "B");  // average A
  Cefz1->Sumw2();
  //  maxbins, 1., maxbins+1.);
  int sumijhb = 0;
  c1->cd(1);
  maxbinx = 0;
  maxbiny = 0;
  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "Page11: HB h_2DsumADCAmplEtaPhiLs0         nx=     " << nx << " ny=     " << ny << endl;
  TH1F* ADCAmplLS0 = new TH1F("ADCAmplLS0", "", 100, 0., 50.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        sumijhb++;
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"Page11: HB h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        ADCAmplLS0->Fill(ccc1);
      }
    }
  }
  cout << "Page11: HB maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << "     sumijhb=  " << sumijhb << endl;
  ADCAmplLS0->SetMarkerStyle(20);
  ADCAmplLS0->SetMarkerSize(0.4);
  ADCAmplLS0->GetYaxis()->SetLabelSize(0.04);
  ADCAmplLS0->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  ADCAmplLS0->SetYTitle("     HB \b");
  ADCAmplLS0->SetMarkerColor(2);
  ADCAmplLS0->SetLineColor(0);  //ADCAmplLS0->SetMinimum(10.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      ADCAmplLS0 ->Draw("L");
  ADCAmplLS0->Draw("Error");
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  maxbins, 1., maxbins+1.);
  c1->cd(2);
  TH1F* ADCAmplLS = new TH1F("ADCAmplLS", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        //	  cout<<"Page11: HB h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        //	  ADCAmplLS ->Fill(ccc1/maxbinx);
        ADCAmplLS->Fill(float(i), ccc1* maxbinx / sumijhb);
      }
    }
  }
  ADCAmplLS->SetMarkerStyle(20);
  ADCAmplLS->SetMarkerSize(0.4);
  ADCAmplLS->GetYaxis()->SetLabelSize(0.04);
  ADCAmplLS->SetMarkerColor(2);
  ADCAmplLS->SetLineColor(0);
  ADCAmplLS->SetXTitle("        iLS  \b");
  ADCAmplLS->SetYTitle("     <A>k \b");
  //ADCAmplLS->SetMinimum(0.8);ADCAmplLS->SetMaximum(500.);
  //      gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  ADCAmplLS->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(3);
  TH1F* ADCAmplLS1 = new TH1F("ADCAmplLS1", "", 200, 0., 100.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"Page11: HB h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        ADCAmplLS1->Fill(ccc1);
      }
    }
  }
  cout << "Page11: HB maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << endl;
  ADCAmplLS1->SetMarkerStyle(20);
  ADCAmplLS1->SetMarkerSize(0.4);
  ADCAmplLS1->GetYaxis()->SetLabelSize(0.04);
  ADCAmplLS1->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  ADCAmplLS1->SetMarkerColor(2);
  ADCAmplLS1->SetLineColor(0);
  ADCAmplLS1->SetMinimum(0.8);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      ADCAmplLS1 ->Draw("L");
  ADCAmplLS1->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(4);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1;
  //          neta=72;    nphi=82;
  //         zneta=18;   znphi=22;
  TH2F* Cefz4 = new TH2F("Cefz4", "", 22, -11., 11., 18, 0., 18.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz1->GetBinContent(i, j);
      //if(ccc1>0.) cout<<"Page11: HB h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1/maxbinx <<endl;
      //	if(ccc1>0. && ccc1/maxbinx < 2000) {
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //		  if(i==1) cout<<"Page11: HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta-11 <<" jphi= "<< jphi-1 <<"  A= "<< ccc1/maxbinx <<endl;
        //	  Cefz4 ->Fill(jeta-11,jphi-1,ccc1/maxbinx);

        Cefz4->Fill(jeta - 11, jphi - 1, ccc1 * maxbiny / sumijhb);
      }
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Cefz4->SetMarkerStyle(20);
  Cefz4->SetMarkerSize(0.4);
  Cefz4->GetZaxis()->SetLabelSize(0.04);
  Cefz4->SetXTitle("<A>ij         #eta  \b");
  Cefz4->SetYTitle("      #phi \b");
  Cefz4->SetZTitle("<A>ij  - All \b");
  Cefz4->SetMarkerColor(2);
  Cefz4->SetLineColor(2);  //      Cefz4->SetMaximum(1.000);  //      Cefz4->SetMinimum(1.0);
  Cefz4->Draw("COLZ");

  c1->Update();

  //======================================================================

  //========================================================================================== 12   HB
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  //    c1->Divide(1,3);
  double ccc0HB = 0.;
  gStyle->SetOptStat(1110000);
  c1->Divide(2, 3);

  c1->cd(1);
  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HB GainStability        nx=     " << nx << " ny=     " << ny << endl;
  TH1F* GainStability0 = new TH1F("GainStability0", "", maxbins, 1., maxbins + 1.);
  TH1F* GainStability1 = new TH1F("GainStability1", "", maxbins, 1., maxbins + 1.);
  TH1F* GainStability2 = new TH1F("GainStability2", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Cefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	    cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
        if (jeta == 8 && jphi == 11)
          GainStability0->Fill(i, ccc1);
        if (jeta == 10 && jphi == 11)
          GainStability1->Fill(i, ccc1);
        if (jeta == 12 && jphi == 11)
          GainStability2->Fill(i, ccc1);
      }
    }
  }
  GainStability0->SetMarkerStyle(20);
  GainStability0->SetMarkerSize(0.4);
  GainStability0->GetYaxis()->SetLabelSize(0.04);
  GainStability0->SetXTitle("GainStability0 \b");
  GainStability0->SetMarkerColor(2);
  GainStability0->SetLineColor(
      0);  // GainStability0 ->SetMaximum(30.0);// GainStability0 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  GainStability0->Draw("Error");

  //================
  c1->cd(2);
  GainStability1->SetMarkerStyle(20);
  GainStability1->SetMarkerSize(0.4);
  GainStability1->GetYaxis()->SetLabelSize(0.04);
  GainStability1->SetXTitle("GainStability1 \b");
  GainStability1->SetMarkerColor(2);
  GainStability1->SetLineColor(
      0);  // GainStability1 ->SetMaximum(30.0);// GainStability1 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  GainStability1->Draw("Error");

  //================
  c1->cd(3);
  GainStability2->SetMarkerStyle(20);
  GainStability2->SetMarkerSize(0.4);
  GainStability2->GetYaxis()->SetLabelSize(0.04);
  GainStability2->SetXTitle("GainStability2 \b");
  GainStability2->SetMarkerColor(2);
  GainStability2->SetLineColor(
      0);  // GainStability2 ->SetMaximum(30.0);// GainStability2 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  GainStability2->Draw("Error");

  //======================================================================
  //================
  c1->cd(4);
  TH1F* Ghb5 = new TH1F("Ghb5", "", nx, 1., nx + 1.);
  //    TH1F* Ghb51 = new TH1F("Ghb51","", nx, 1., nx+1.);
  //    TH1F* Ghb50= new TH1F("Ghb50","", nx, 1., nx+1.);
  //    TH1F* Ghb5 = (TH1F*)Ghb50->Clone("Ghb5");
  // j - etaphi index:
  for (int j = 1; j <= ny; j++) {
    ccc0HB = Cefz1->GetBinContent(1, j);
    //	if(ccc0HB <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Cefz1->GetBinContent(i,j);if(ccc2>0.){ccc0HB=ccc2;cout<<"!!! ccc0HB= "<<ccc0HB<<endl;break;} }
    if (ccc0HB <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Cefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0HB = ccc2;
          break;
        }
      }
    if (ccc0HB > 0.) {
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0HB;
          Ghb5->Fill(float(i), Rij);
          //	      Ghb51 ->Fill( float(i), Rij);
          //	      Ghb50->Fill( float(i), 1.);
        }
      }
    }
  }
  //    Ghb5->Divide(Ghb51,Ghb50, 1, 1, "B");// average A
  for (int i = 1; i <= nx; i++) {
    Ghb5->SetBinError(i, 0.0001);
  }
  Ghb5->SetMarkerStyle(20);
  Ghb5->SetMarkerSize(0.4);
  Ghb5->GetYaxis()->SetLabelSize(0.04);
  Ghb5->SetMarkerColor(2);
  Ghb5->SetLineColor(0);
  Ghb5->SetXTitle("        iLS  \b");
  Ghb5->SetYTitle("     <R> \b");
  Ghb5->SetTitle("<Ri> vs iLS \b");
  Ghb5->SetMinimum(0.);  //Ghb5->SetMaximum(2.5);
  //            gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Ghb5->SetStats(0);
  Ghb5->GetYaxis()->SetLabelSize(0.025);
  Ghb5->Draw("Error");
  //================
  c1->cd(5);
  TH2F* Ghb60 = new TH2F("Ghb60", "", 22, -11., 11., 18, 0., 18.);
  TH2F* Ghb61 = new TH2F("Ghb61", "", 22, -11., 11., 18, 0., 18.);
  TH2F* Ghb6 = new TH2F("Ghb6", "", 22, -11., 11., 18, 0., 18.);
  // j - etaphi index; i - # LSs;
  //
  // define mean and RMS:
  double sumjHB = 0.;
  int njHB = 0;
  double meanjHB = 0.;
  for (int j = 1; j <= ny; j++) {
    ccc0HB = Cefz1->GetBinContent(1, j);
    if (ccc0HB <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Cefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0HB = ccc2;
          break;
        }
      }
    if (ccc0HB > 0.) {
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j) / ccc0HB;
        if (ccc1 > 0.) {
          sumjHB += ccc1;
          njHB++;
        }
      }
      meanjHB = sumjHB / njHB;
    }
  }  // j

  double ssumjHB = 0.;
  njHB = 0;
  double sigmajHB = 0.;
  for (int j = 1; j <= ny; j++) {
    ccc0HB = Cefz1->GetBinContent(1, j);
    if (ccc0HB <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Cefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0HB = ccc2;
          break;
        }
      }
    if (ccc0HB > 0.) {
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j) / ccc0HB;
        if (ccc1 > 0.) {
          ssumjHB += (ccc1 - meanjHB) * (ccc1 - meanjHB);
          njHB++;
        }
      }
      sigmajHB = sqrt(ssumjHB / njHB);
    }
  }  // j

  double dif3rmsHBMIN = meanjHB - 3 * sigmajHB;
  if (dif3rmsHBMIN < 0.)
    dif3rmsHBMIN = 0.;
  double dif3rmsHBMAX = meanjHB + 3 * sigmajHB;
  cout << "22HB-2    meanjHB=  " << meanjHB << "  sigmajHB=  " << sigmajHB << "  dif3rmsHBMIN=  " << dif3rmsHBMIN
       << "  dif3rmsHBMAX=  " << dif3rmsHBMAX << endl;

  double MAXdif3rmsHBMIN = dif3rmsHBMIN;
  double MINdif3rmsHBMAX = dif3rmsHBMAX;
  if (MAXdif3rmsHBMIN < 0.95)
    MAXdif3rmsHBMIN = 0.95;
  if (MINdif3rmsHBMAX > 1.05)
    MINdif3rmsHBMAX = 1.05;
  cout << "22HB-2     MAXdif3rmsHBMIN=  " << MAXdif3rmsHBMIN << "     MINdif3rmsHBMAX=  " << MINdif3rmsHBMAX << endl;
  //
  for (int j = 1; j <= ny; j++) {
    ccc0HB = Cefz1->GetBinContent(1, j);
    if (ccc0HB <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Cefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0HB = ccc2;
          break;
        }
      }
    if (ccc0HB > 0.) {
      int jeta = (j - 1) / 18;         // jeta = 0-21
      int jphi = (j - 1) - 18 * jeta;  // jphi=0-17
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0HB;
          if (Rij < MAXdif3rmsHBMIN || Rij > MINdif3rmsHBMAX) {
            Ghb61->Fill(jeta - 11, jphi, Rij);
            Ghb60->Fill(jeta - 11, jphi, 1.);
          }
        }                                 //if(ccc1>0.
      }                                   // i
    }                                     //if(ccc0HB>0
  }                                       // j
  Ghb6->Divide(Ghb61, Ghb60, 1, 1, "B");  // average R
  //      Ghb6->SetLabelOffset (Float_t offset=0.005, Option_t *axis="X")//Set offset between axis and axis' labels
  //      Ghb6->GetZaxis()->SetLabelOffset(-0.05);
  Ghb6->GetZaxis()->SetLabelSize(0.025);

  Ghb6->SetXTitle("             #eta  \b");
  Ghb6->SetYTitle("      #phi \b");
  Ghb6->SetTitle(
      "<Rj> for |1-<R>| > 0.05 \b");  //      Ghb6->SetMaximum(1.000);  //      Ghb6->SetMinimum(1.0); //Ghb6->SetZTitle("Rij averaged over LSs \b"); //Ghb6->GetZaxis()->SetLabelSize(0.04); //Ghb6->SetMarkerStyle(20);// Ghb6->SetMarkerSize(0.4);//Ghb6->SetMarkerColor(2); //Ghb6->SetLineColor(2);
  //gStyle->SetOptStat(kFALSE);
  Ghb6->SetStats(0);
  Ghb6->Draw("COLZ");
  //================
  c1->cd(6);
  TH1F* Ghb7 = new TH1F("Ghb7", "", 120, 0.4, 1.6);
  // j - etaphi index:
  for (int j = 1; j <= ny; j++) {
    ccc0HB = Cefz1->GetBinContent(1, j);
    if (ccc0HB <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Cefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0HB = ccc2;
          break;
        }
      }
    if (ccc0HB > 0.) {
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0HB;
          Ghb7->Fill(Rij);
        }
      }
    }
  }
  Ghb7->SetMarkerStyle(20);
  Ghb7->SetMarkerSize(0.4);
  Ghb7->GetYaxis()->SetLabelSize(0.04);
  Ghb7->SetMarkerColor(2);
  Ghb7->SetLineColor(0);
  Ghb7->SetYTitle("        N  \b");
  Ghb7->SetXTitle("     Rij \b");
  Ghb7->SetTitle(" Rij \b");
  //Ghb7->SetMinimum(0.8);Ghb7->SetMaximum(500.);
  gPad->SetGridy();
  gPad->SetGridx();  //            gPad->SetLogy();
  //      Ghb7->SetStats(1110000);
  Ghb7->GetYaxis()->SetLabelSize(0.025);
  Ghb7->Draw("Error");
  Float_t ymaxHB = Ghb7->GetMaximum();
  cout << "22HB-3   ymaxHB=  " << ymaxHB << "       MAXdif3rmsHBMIN=  " << MAXdif3rmsHBMIN
       << "         MINdif3rmsHBMAX=  " << MINdif3rmsHBMAX << endl;
  TLine* lineHB = new TLine(MAXdif3rmsHBMIN, 0., MAXdif3rmsHBMIN, ymaxHB);
  lineHB->SetLineColor(kBlue);
  lineHB->Draw();
  TLine* line1HB = new TLine(MINdif3rmsHBMAX, 0., MINdif3rmsHBMAX, ymaxHB);
  line1HB->SetLineColor(kBlue);
  line1HB->Draw();

  //================
  //
  // gain stabilitY:
  // Rij = Aij / A1j , where i-over LSs, j-channels
  //

  /*
      double ccc0 = 0.;
//================      
      c1->cd(4);
      TH1F* Cefz51 = new TH1F("Cefz51","", maxbins, 1., maxbins+1.);
      TH1F* Cefz50= new TH1F("Cefz50","", maxbins, 1., maxbins+1.);
      TH1F* Cefz5 = (TH1F*)Cefz50->Clone("Cefz5");
      // j - etaphi index:
      for (int j=1;j<=ny;j++) {
	ccc0 =  Cefz1->GetBinContent(1,j);
	//	if(ccc0 <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Cefz1->GetBinContent(i,j);if(ccc2>0.){ccc0=ccc2;cout<<"!!! ccc0= "<<ccc0<<endl;break;} }
	if(ccc0 <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Cefz1->GetBinContent(i,j);if(ccc2>0.){ccc0=ccc2;break;} }
	if(ccc0>0.) { 
	  // i - # LSs:
	  for (int i=1;i<=nx;i++) {
	    double ccc1 =  Cefz1->GetBinContent(i,j);
	    if(ccc1>0.) {
	      double Rij = ccc1/ccc0;		  
	      Cefz51 ->Fill( float(i), Rij);
	      Cefz50->Fill( float(i), 1.);
	    }}}}
      Cefz5->Divide(Cefz51,Cefz50, 1, 1, "B");// average A
      for (int jeta=1;jeta<=maxbins;jeta++) {Cefz5->SetBinError(jeta,0.0001);}
      Cefz5 ->SetMarkerStyle(20);Cefz5 ->SetMarkerSize(0.4);Cefz5 ->GetYaxis()->SetLabelSize(0.04);Cefz5 ->SetMarkerColor(2);Cefz5 ->SetLineColor(0);
      Cefz5->SetXTitle("        iLS  \b");  Cefz5->SetYTitle("     Rij \b");
      Cefz5->SetMinimum(0.);//Cefz5->SetMaximum(2.5);
      //            gPad->SetLogy();
      gPad->SetGridy();gPad->SetGridx();     
      Cefz5 ->Draw("Error");
//================
      c1->cd(5);
      TH2F* Cefz6     = new TH2F("Cefz6","",  22, -11., 11., 18, 0., 18. );
      // j - etaphi index:
      double mincutR = 999999.;double maxcutR = -999999.;
      for (int j=1;j<=ny;j++) {
	ccc0 =  Cefz1->GetBinContent(1,j);
	if(ccc0 <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Cefz1->GetBinContent(i,j);if(ccc2>0.){ccc0=ccc2;break;} }
	if(ccc0>0.) { 
	  int jeta = (j-1)/18;// jeta = 0-21
	  int jphi = (j-1)-18*jeta;// jphi=0-17 
	  // define mean and RMS:
	  double sumj=0.; double ssumj=0.; int nj=0;double meanj=0.;double sigmaj=0.;
	  for (int i=1;i<=nx;i++) {double ccc1 =  Cefz1->GetBinContent(i,j)/ccc0;if(ccc1>0.){sumj += ccc1;nj++;} } meanj=sumj/nj;
	  for (int i=1;i<=nx;i++) {double ccc1 =  Cefz1->GetBinContent(i,j)/ccc0;if(ccc1>0.) {ssumj += (ccc1-meanj)*(ccc1-meanj);}} sigmaj = sqrt(ssumj/nj);
	  //          cout<<"12    j=     "<< j <<" meanj=     "<< meanj <<" sigmaj=     "<< sigmaj <<endl;
	  double dif3rmsMIN = meanj-3*sigmaj;if(dif3rmsMIN<0.) dif3rmsMIN = 0.;  double dif3rmsMAX = meanj+3*sigmaj;
	  //	  cout<<"12    j=     "<< j <<" dif3rmsMIN=     "<< dif3rmsMIN <<" dif3rmsMAX=     "<< dif3rmsMAX <<endl;
	  if(dif3rmsMIN<mincutR) mincutR=dif3rmsMIN;if(dif3rmsMAX>maxcutR) maxcutR=dif3rmsMAX;
	  // i - # LSs:
	  for (int i=1;i<=nx;i++) {
	    double ccc1 =  Cefz1->GetBinContent(i,j);
	    if(ccc1>0.) {
	      double Rij = ccc1/ccc0;		  
	      if(Rij<dif3rmsMIN || Rij>dif3rmsMAX) { Cefz6 ->Fill(jeta-11,jphi,Rij); }
		      //    if(Rij<dif3rmsMIN || Rij>dif3rmsMAX) Cefz6 ->Fill(jeta-11,jphi,1.);
	    }//if(ccc1>0.
	  }// i
	}//if(ccc0>0 
      }// j
      Cefz6->SetMarkerStyle(20); Cefz6->SetMarkerSize(0.4); Cefz6->GetZaxis()->SetLabelSize(0.04); Cefz6->SetXTitle("Rij out 3sigma       #eta  \b"); Cefz6->SetYTitle("      #phi \b"); Cefz6->SetZTitle("sum of bad Rij  \b"); Cefz6->SetMarkerColor(2); Cefz6->SetLineColor(2);      //      Cefz6->SetMaximum(1.000);  //      Cefz6->SetMinimum(1.0);
      Cefz6->Draw("COLZ");
//================
      c1->cd(6);
      TH1F* Cefz7 = new TH1F("Cefz7","", 100, 0.8, 1.2);
      // j - etaphi index:
      for (int j=1;j<=ny;j++) {
	ccc0 =  Cefz1->GetBinContent(1,j);
	if(ccc0 <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Cefz1->GetBinContent(i,j);if(ccc2>0.){ccc0=ccc2;break;} }
	if(ccc0>0.) { 
	  // i - # LSs:
	  for (int i=1;i<=nx;i++) {
	    double ccc1 =  Cefz1->GetBinContent(i,j);
	    if(ccc1>0.) {
	      double Rij = ccc1/ccc0;		  
	      Cefz7 ->Fill( Rij );
	    }}}}
      Cefz7 ->SetMarkerStyle(20);Cefz7 ->SetMarkerSize(0.4);Cefz7 ->GetYaxis()->SetLabelSize(0.04);Cefz7 ->SetMarkerColor(2);Cefz7 ->SetLineColor(0);
      Cefz7->SetXTitle("        iLS  \b");  Cefz7->SetYTitle("     Rij \b");
      //Cefz7->SetMinimum(0.8);Cefz7->SetMaximum(500.);
      //            gPad->SetLogy();
      gPad->SetGridy();gPad->SetGridx();     
      Cefz7 ->Draw("Error");
      Float_t ymax = Cefz7->GetMaximum();
      TLine *line = new TLine(mincutR,0.,mincutR,ymax);line->SetLineColor(4);line->Draw();
      TLine *line1= new TLine(maxcutR,0.,maxcutR,ymax);line1->SetLineColor(4);line1->Draw();      
      //================

*/
  ////////////////////////////////////////////////////////////////////////////////////

  c1->Update();
  gStyle->SetOptStat(0);

  //========================================================================================== 13  HB: j = 7,8,9,10            11,12,13,14      7
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta0k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  int kcount = 1;
  cout << "HB Gforhbjeta0k *********************************************************************       jeta == 7    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta0k0 = new TH1F("h2CeffGforhbjeta0k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 7) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta0k0 = new TH1F("Gforhbjeta0k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta0k0 = (TH1F*)h2CeffGforhbjeta0k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta0k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta0k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta0k0->SetMarkerStyle(20);
      Gforhbjeta0k0->SetMarkerSize(0.4);
      Gforhbjeta0k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta0k0->SetXTitle("Gforhbjeta0k0 \b");
      Gforhbjeta0k0->SetMarkerColor(2);
      Gforhbjeta0k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta0k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta0k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 14 HB: j = 7,8,9,10            11,12,13,14     8
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta1k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta1k *********************************************************************       jeta == 8    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta1k0 = new TH1F("h2CeffGforhbjeta1k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 8) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta1k0 = new TH1F("Gforhbjeta1k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta1k0 = (TH1F*)h2CeffGforhbjeta1k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta1k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta1k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta1k0->SetMarkerStyle(20);
      Gforhbjeta1k0->SetMarkerSize(0.4);
      Gforhbjeta1k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta1k0->SetXTitle("Gforhbjeta1k0 \b");
      Gforhbjeta1k0->SetMarkerColor(2);
      Gforhbjeta1k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta1k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta1k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 15 HB: j = 7,8,9,10            11,12,13,14      9
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta2k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta2k *********************************************************************       jeta == 9    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta2k0 = new TH1F("h2CeffGforhbjeta2k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 9) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta2k0 = new TH1F("Gforhbjeta2k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta2k0 = (TH1F*)h2CeffGforhbjeta2k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta2k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta2k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta2k0->SetMarkerStyle(20);
      Gforhbjeta2k0->SetMarkerSize(0.4);
      Gforhbjeta2k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta2k0->SetXTitle("Gforhbjeta2k0 \b");
      Gforhbjeta2k0->SetMarkerColor(2);
      Gforhbjeta2k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta2k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta2k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 16 HB: j = 7,8,9,10            11,12,13,14       10
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta3k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta3k *********************************************************************       jeta == 10   "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta3k0 = new TH1F("h2CeffGforhbjeta3k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 10) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta3k0 = new TH1F("Gforhbjeta3k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta3k0 = (TH1F*)h2CeffGforhbjeta3k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta3k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta3k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta3k0->SetMarkerStyle(20);
      Gforhbjeta3k0->SetMarkerSize(0.4);
      Gforhbjeta3k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta3k0->SetXTitle("Gforhbjeta3k0 \b");
      Gforhbjeta3k0->SetMarkerColor(2);
      Gforhbjeta3k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta3k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta3k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 17 HB: j = 7,8,9,10            11,12,13,14        11
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta18k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta18k *********************************************************************       jeta == 11    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta18k0 = new TH1F("h2CeffGforhbjeta18k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 11) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta18k0 = new TH1F("Gforhbjeta18k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta18k0 = (TH1F*)h2CeffGforhbjeta18k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta18k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta18k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta18k0->SetMarkerStyle(20);
      Gforhbjeta18k0->SetMarkerSize(0.4);
      Gforhbjeta18k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta18k0->SetXTitle("Gforhbjeta18k0 \b");
      Gforhbjeta18k0->SetMarkerColor(2);
      Gforhbjeta18k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta18k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta18k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 18 HB: j = 7,8,9,10            11,12,13,14       12
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta19k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta19k *********************************************************************       jeta == 12    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta19k0 = new TH1F("h2CeffGforhbjeta19k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 12) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta19k0 = new TH1F("Gforhbjeta19k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta19k0 = (TH1F*)h2CeffGforhbjeta19k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta19k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta19k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta19k0->SetMarkerStyle(20);
      Gforhbjeta19k0->SetMarkerSize(0.4);
      Gforhbjeta19k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta19k0->SetXTitle("Gforhbjeta19k0 \b");
      Gforhbjeta19k0->SetMarkerColor(2);
      Gforhbjeta19k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta19k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta19k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 19 HB: j = 7,8,9,10            11,12,13,14       13
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta20k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta20k *********************************************************************       jeta == 13    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta20k0 = new TH1F("h2CeffGforhbjeta20k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 13) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta20k0 = new TH1F("Gforhbjeta20k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta20k0 = (TH1F*)h2CeffGforhbjeta20k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta20k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta20k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta20k0->SetMarkerStyle(20);
      Gforhbjeta20k0->SetMarkerSize(0.4);
      Gforhbjeta20k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta20k0->SetXTitle("Gforhbjeta20k0 \b");
      Gforhbjeta20k0->SetMarkerColor(2);
      Gforhbjeta20k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta20k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta20k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 20 HB: j = 7,8,9,10            11,12,13,14        14
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Cefz1->GetXaxis()->GetNbins();
  ny = Cefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohb > 0.)      nhistohb /= nlsohb;
  //    cout<<"HB Gforhbjeta21k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohb=     "<< nhistohb <<endl;
  kcount = 1;
  cout << "HB Gforhbjeta21k *********************************************************************       jeta == 14    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhbjeta21k0 = new TH1F("h2CeffGforhbjeta21k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 14) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhbjeta21k0 = new TH1F("Gforhbjeta21k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhbjeta21k0 = (TH1F*)h2CeffGforhbjeta21k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Cefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HB  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhbjeta21k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HB Gforhbjeta21k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhbjeta21k0->SetMarkerStyle(20);
      Gforhbjeta21k0->SetMarkerSize(0.4);
      Gforhbjeta21k0->GetYaxis()->SetLabelSize(0.04);
      Gforhbjeta21k0->SetXTitle("Gforhbjeta21k0 \b");
      Gforhbjeta21k0->SetMarkerColor(2);
      Gforhbjeta21k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhbjeta21k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhbjeta21k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 21    HE - "h_2DsumADCAmplEtaPhiLs1
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH2F* Sefz1KKK = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs1");
  TH2F* Sefz1LLL = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs10");
  TH2F* Sefz1 = (TH2F*)Sefz1LLL->Clone("Sefz1");
  Sefz1->Divide(Sefz1KKK, Sefz1LLL, 1, 1, "B");  // average A
  Sefz1->Sumw2();

  c1->cd(1);
  maxbinx = 0;
  maxbiny = 0;
  int sumijhe = 0;
  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HE h_2DsumADCAmplEtaPhiLs0         nx=     " << nx << " ny=     " << ny << endl;
  // i - # LSs:
  TH1F* Sefw0 = new TH1F("Sefw0", "", 200, 0., 15000.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Sefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        sumijhe++;
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HE h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Sefw0->Fill(ccc1);
      }
    }
  }
  cout << "HE maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << "     sumijhe=  " << sumijhe << endl;
  Sefw0->SetMarkerStyle(20);
  Sefw0->SetMarkerSize(0.4);
  Sefw0->GetYaxis()->SetLabelSize(0.04);
  Sefw0->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Sefw0->SetYTitle("     HE \b");
  Sefw0->SetMarkerColor(2);
  Sefw0->SetLineColor(0);
  Sefw0->SetMinimum(10.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Sefw0 ->Draw("L");
  Sefw0->Draw("Error");
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(2);
  TH1F* Sefw = new TH1F("Sefw", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Sefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        //	  cout<<"HE h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        //	  Sefw ->Fill(ccc1/maxbinx);
        Sefw->Fill(float(i), ccc1* maxbinx / sumijhe);
      }
    }
  }
  Sefw->SetMarkerStyle(20);
  Sefw->SetMarkerSize(0.4);
  Sefw->GetYaxis()->SetLabelSize(0.04);
  Sefw->SetMarkerColor(2);
  Sefw->SetLineColor(0);
  Sefw->SetXTitle("        iLS  \b");
  Sefw->SetYTitle("     <A>k \b");
  //Sefw->SetMinimum(0.8);Sefw->SetMaximum(500.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Sefw->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(3);
  TH1F* Sefw1 = new TH1F("Sefw1", "", 100, 0., 9000.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Sefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HE h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Sefw1->Fill(ccc1);
      }
    }
  }
  cout << "HE maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << endl;
  Sefw1->SetMarkerStyle(20);
  Sefw1->SetMarkerSize(0.4);
  Sefw1->GetYaxis()->SetLabelSize(0.04);
  Sefw1->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Sefw1->SetMarkerColor(2);
  Sefw1->SetLineColor(0);
  //Sefw1->SetMinimum(0.8);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Sefw1 ->Draw("L");
  Sefw1->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(4);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;
  TH2F* Sefz4 = new TH2F("Sefz4", "", 22, -11., 11., 18, 0., 18.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Sefz1->GetBinContent(i, j);
      //if(ccc1>0.) cout<<"HE h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1/maxbinx <<endl;
      //	if(ccc1>0. && ccc1/maxbinx < 2000) {
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //		  if(i==1) cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta-11 <<" jphi= "<< jphi-1 <<"  A= "<< ccc1/maxbinx <<endl;
        //	  Sefz4 ->Fill(jeta-11,jphi-1,ccc1/maxbinx);
        Sefz4->Fill(jeta - 11, jphi - 1, ccc1 * maxbiny / sumijhe);
      }
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Sefz4->SetMarkerStyle(20);
  Sefz4->SetMarkerSize(0.4);
  Sefz4->GetZaxis()->SetLabelSize(0.08);
  Sefz4->SetXTitle("<A>ij         #eta  \b");
  Sefz4->SetYTitle("      #phi \b");
  Sefz4->SetZTitle("<A>ij  - All \b");
  Sefz4->SetMarkerColor(2);
  Sefz4->SetLineColor(2);  //      Sefz4->SetMaximum(1.000);  //      Sefz4->SetMinimum(1.0);
  Sefz4->Draw("COLZ");

  c1->Update();

  //======================================================================

  //========================================================================================== 22 HE
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 3);

  c1->cd(1);
  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HE Sefk        nx=     " << nx << " ny=     " << ny << endl;
  TH1F* Sefk0 = new TH1F("Sefk0", "", maxbins, 1., maxbins + 1.);
  TH1F* Sefk1 = new TH1F("Sefk1", "", maxbins, 1., maxbins + 1.);
  TH1F* Sefk2 = new TH1F("Sefk2", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Sefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	    cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
        if (jeta == 4 && jphi == 11)
          Sefk0->Fill(i, ccc1);
        if (jeta == 5 && jphi == 11)
          Sefk1->Fill(i, ccc1);
        if (jeta == 16 && jphi == 11)
          Sefk2->Fill(i, ccc1);
      }
    }
  }
  Sefk0->SetMarkerStyle(20);
  Sefk0->SetMarkerSize(0.4);
  Sefk0->GetYaxis()->SetLabelSize(0.04);
  Sefk0->SetXTitle("Sefk0 \b");
  Sefk0->SetMarkerColor(2);
  Sefk0->SetLineColor(
      0);  // Sefk0 ->SetMaximum(30.0);// Sefk0 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Sefk0->Draw("Error");

  //================
  c1->cd(2);
  Sefk1->SetMarkerStyle(20);
  Sefk1->SetMarkerSize(0.4);
  Sefk1->GetYaxis()->SetLabelSize(0.04);
  Sefk1->SetXTitle("Sefk1 \b");
  Sefk1->SetMarkerColor(2);
  Sefk1->SetLineColor(
      0);  // Sefk1 ->SetMaximum(30.0);// Sefk1 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Sefk1->Draw("Error");

  //================
  c1->cd(3);
  Sefk2->SetMarkerStyle(20);
  Sefk2->SetMarkerSize(0.4);
  Sefk2->GetYaxis()->SetLabelSize(0.04);
  Sefk2->SetXTitle("Sefk2 \b");
  Sefk2->SetMarkerColor(2);
  Sefk2->SetLineColor(
      0);  // Sefk2 ->SetMaximum(30.0);// Sefk2 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Sefk2->Draw("Error");

  //======================================================================
  //================
  //
  // gain stabilitY:
  // Rij = Aij / A1j , where i-over LSs, j-channels
  //
  double ccc0E = 0.;
  //================
  /*
      c1->cd(4);
      TH1F* Sefz51 = new TH1F("Sefz51","", maxbins, 1., maxbins+1.);
      TH1F* Sefz50= new TH1F("Sefz50","", maxbins, 1., maxbins+1.);
      TH1F* Sefz5 = (TH1F*)Sefz50->Clone("Sefz5");
      // j - etaphi index:
      for (int j=1;j<=ny;j++) {
	ccc0E =  Sefz1->GetBinContent(1,j);
	//	if(ccc0E <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Sefz1->GetBinContent(i,j);if(ccc2>0.){ccc0E=ccc2;cout<<"!!! ccc0E= "<<ccc0E<<endl;break;} }
	if(ccc0E <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Sefz1->GetBinContent(i,j);if(ccc2>0.){ccc0E=ccc2;break;} }
	if(ccc0E>0.) { 
	  int jeta = (j-1)/18;// jeta = 0-21
	  int jphi = (j-1)-18*jeta;// jphi=0-17 
	  // i - # LSs:
	  for (int i=1;i<=nx;i++) {
	    double ccc1 =  Sefz1->GetBinContent(i,j);
	    if(ccc1>0.) {
	      double Rij = ccc1/ccc0E;		  
	      Sefz51 ->Fill( float(i), Rij);
	      Sefz50->Fill( float(i), 1.);
	    }//if(ccc1>0.
	  }// i
	  //      }// j
	  Sefz5->Divide(Sefz51,Sefz50, 1, 1, "B");// average A
	  for (int i=1;i<=maxbins;i++) {Sefz5->SetBinError(i,0.0001);}
	  //
	  cout<< "j=  " <<j  << "    jeta=  " <<jeta  << "   jphi=  " <<jphi  << "  j/50+1=  " << j/50+1  << "  maxbins= " << maxbins  << "  BinContent(maxbins)=  " << Sefz5->GetBinContent(maxbins) <<endl;
	        Sefz5 ->SetMarkerColor(j/50+1);
	  //Sefz5 ->SetMarkerColor(jeta);
	  //Sefz5 ->SetMarkerColor(jphi);
	  //      Sefz5 ->SetMarkerColor(2);
	  Sefz5 ->SetMarkerStyle(20);Sefz5 ->SetMarkerSize(0.4);Sefz5 ->GetYaxis()->SetLabelSize(0.04);Sefz5 ->SetLineColor(0);Sefz5->SetXTitle("        iLS  \b");  Sefz5->SetYTitle("     Rij \b");
	  gPad->SetGridy();gPad->SetGridx();  //            gPad->SetLogy();
	  //      Sefz5 ->Draw("Error");
	  //Sefz5->SetMinimum(0.995);Sefz5->SetMaximum(1.005);
	  Sefz5 ->Draw("ErrorSame");
	}//if(ccc0E>0 
      }// j
*/
  //================

  c1->cd(4);
  //    TH1F* Sefz5 = new TH1F("Sefz5","", maxbins, 1., maxbins+1.);
  TH1F* Sefz51 = new TH1F("Sefz51", "", maxbins, 1., maxbins + 1.);
  TH1F* Sefz50 = new TH1F("Sefz50", "", maxbins, 1., maxbins + 1.);
  TH1F* Sefz5 = (TH1F*)Sefz50->Clone("Sefz5");
  // j - etaphi index:
  for (int j = 1; j <= ny; j++) {
    ccc0E = Sefz1->GetBinContent(1, j);
    //	if(ccc0E <=0.) for (int i=1;i<=nx;i++) {double ccc2 =  Sefz1->GetBinContent(i,j);if(ccc2>0.){ccc0E=ccc2;cout<<"!!! ccc0E= "<<ccc0E<<endl;break;} }
    if (ccc0E <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Sefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0E = ccc2;
          break;
        }
      }

    cout << "!!! ccc0E= " << ccc0E << endl;

    if (ccc0E > 0.) {
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0E;
          //    Sefz5 ->Fill( float(i), Rij);
          Sefz51->Fill(float(i), Rij);
          Sefz50->Fill(float(i), 1.);
        }
      }
    }
  }
  Sefz5->Divide(Sefz51, Sefz50, 1, 1, "B");  // average A
  for (int jeta = 1; jeta <= maxbins; jeta++) {
    Sefz5->SetBinError(jeta, 0.0001);
  }
  Sefz5->SetMarkerStyle(20);
  Sefz5->SetMarkerSize(0.4);
  Sefz5->GetYaxis()->SetLabelSize(0.04);
  Sefz5->SetMarkerColor(2);
  Sefz5->SetLineColor(0);
  Sefz5->SetXTitle("        iLS  \b");
  Sefz5->SetYTitle("     <R> \b");
  Sefz5->SetMinimum(0.);  //Sefz5->SetMaximum(2.5);
  //            gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Sefz5->Draw("Error");

  //================
  c1->cd(5);
  TH2F* Sefz60 = new TH2F("Sefz60", "", 22, -11., 11., 18, 0., 18.);
  TH2F* Sefz61 = new TH2F("Sefz61", "", 22, -11., 11., 18, 0., 18.);
  TH2F* Sefz6 = new TH2F("Sefz6", "", 22, -11., 11., 18, 0., 18.);
  // j - etaphi index; i - # LSs;
  //
  // define mean and RMS:
  double sumj = 0.;
  int nj = 0;
  double meanj = 0.;
  for (int j = 1; j <= ny; j++) {
    ccc0E = Sefz1->GetBinContent(1, j);
    if (ccc0E <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Sefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0E = ccc2;
          break;
        }
      }
    if (ccc0E > 0.) {
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j) / ccc0E;
        if (ccc1 > 0.) {
          sumj += ccc1;
          nj++;
        }
      }
      meanj = sumj / nj;
    }
  }  // j

  double ssumj = 0.;
  nj = 0;
  double sigmaj = 0.;
  for (int j = 1; j <= ny; j++) {
    ccc0E = Sefz1->GetBinContent(1, j);
    if (ccc0E <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Sefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0E = ccc2;
          break;
        }
      }
    if (ccc0E > 0.) {
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j) / ccc0E;
        if (ccc1 > 0.) {
          ssumj += (ccc1 - meanj) * (ccc1 - meanj);
          nj++;
        }
      }
      sigmaj = sqrt(ssumj / nj);
    }
  }  // j

  double dif3rmsMIN = meanj - 3 * sigmaj;
  if (dif3rmsMIN < 0.)
    dif3rmsMIN = 0.;
  double dif3rmsMAX = meanj + 3 * sigmaj;
  cout << "22-5    meanj=  " << meanj << "  sigmaj=  " << sigmaj << "  dif3rmsMIN=  " << dif3rmsMIN
       << "  dif3rmsMAX=  " << dif3rmsMAX << endl;

  double MAXdif3rmsMIN = dif3rmsMIN;
  double MINdif3rmsMAX = dif3rmsMAX;
  if (MAXdif3rmsMIN < 0.95)
    MAXdif3rmsMIN = 0.95;
  if (MINdif3rmsMAX > 1.05)
    MINdif3rmsMAX = 1.05;
  cout << "22-5     MAXdif3rmsMIN=  " << MAXdif3rmsMIN << "     MINdif3rmsMAX=  " << MINdif3rmsMAX << endl;
  //
  for (int j = 1; j <= ny; j++) {
    ccc0E = Sefz1->GetBinContent(1, j);
    if (ccc0E <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Sefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0E = ccc2;
          break;
        }
      }
    if (ccc0E > 0.) {
      int jeta = (j - 1) / 18;         // jeta = 0-21
      int jphi = (j - 1) - 18 * jeta;  // jphi=0-17
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0E;
          if (Rij < MAXdif3rmsMIN || Rij > MINdif3rmsMAX) {
            Sefz61->Fill(jeta - 11, jphi, Rij);
            Sefz60->Fill(jeta - 11, jphi, 1.);
          }
        }                                    //if(ccc1>0.
      }                                      // i
    }                                        //if(ccc0E>0
  }                                          // j
  Sefz6->Divide(Sefz61, Sefz60, 1, 1, "B");  // average R

  Sefz6->SetMarkerStyle(20);
  Sefz6->SetMarkerSize(0.4);
  Sefz6->GetZaxis()->SetLabelSize(0.04);
  Sefz6->SetXTitle("<Rj> outside_Cuts         #eta  \b");
  Sefz6->SetYTitle("      #phi \b");
  Sefz6->SetZTitle("Rij averaged over LSs \b");
  Sefz6->SetMarkerColor(2);
  Sefz6->SetLineColor(2);  //      Sefz6->SetMaximum(1.000);  //      Sefz6->SetMinimum(1.0);
  Sefz6->Draw("COLZ");
  //================
  c1->cd(6);
  TH1F* Sefz7 = new TH1F("Sefz7", "", 120, 0.4, 1.6);
  // j - etaphi index:
  for (int j = 1; j <= ny; j++) {
    ccc0E = Sefz1->GetBinContent(1, j);
    if (ccc0E <= 0.)
      for (int i = 1; i <= nx; i++) {
        double ccc2 = Sefz1->GetBinContent(i, j);
        if (ccc2 > 0.) {
          ccc0E = ccc2;
          break;
        }
      }
    if (ccc0E > 0.) {
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          double Rij = ccc1 / ccc0E;
          Sefz7->Fill(Rij);
        }
      }
    }
  }
  Sefz7->SetMarkerStyle(20);
  Sefz7->SetMarkerSize(0.4);
  Sefz7->GetYaxis()->SetLabelSize(0.04);
  Sefz7->SetMarkerColor(2);
  Sefz7->SetLineColor(0);
  Sefz7->SetYTitle("        N  \b");
  Sefz7->SetXTitle("     Rij \b");
  //Sefz7->SetMinimum(0.8);Sefz7->SetMaximum(500.);
  //            gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Sefz7->Draw("Error");
  double ymaxE = Sefz7->GetMaximum();
  //	  cout<< "ymaxE=  " <<ymaxE  <<endl;
  cout << "22-6   ymaxE=  " << ymaxE << "       MAXdif3rmsMIN=  " << MAXdif3rmsMIN
       << "         MINdif3rmsMAX=  " << MINdif3rmsMAX << endl;
  TLine* lineE = new TLine(MAXdif3rmsMIN, 0., MAXdif3rmsMIN, ymaxE);
  lineE->SetLineColor(kGreen);
  lineE->Draw();
  TLine* line1E = new TLine(MINdif3rmsMAX, 0., MINdif3rmsMAX, ymaxE);
  line1E->SetLineColor(kGreen);
  line1E->Draw();
  //================
  ////////////////////////////////////////////////////////////////////////////////////

  c1->Update();

  //========================================================================================== 23  HE: j = 3,4,5, 6, 7      14,15,16,17,18                 3
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta0k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta0k *********************************************************************       jeta == 3    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta0k0 = new TH1F("h2CeffGforhejeta0k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 3) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta0k0 = new TH1F("Gforhejeta0k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta0k0 = (TH1F*)h2CeffGforhejeta0k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta0k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta0k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta0k0->SetMarkerStyle(20);
      Gforhejeta0k0->SetMarkerSize(0.4);
      Gforhejeta0k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta0k0->SetXTitle("Gforhejeta0k0 \b");
      Gforhejeta0k0->SetMarkerColor(2);
      Gforhejeta0k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta0k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta0k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 24 HE: j = 3,4,5, 6, 7      14,15,16,17,18                     4
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta1k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta1k *********************************************************************       jeta == 4    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta1k0 = new TH1F("h2CeffGforhejeta1k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 4) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta1k0 = new TH1F("Gforhejeta1k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta1k0 = (TH1F*)h2CeffGforhejeta1k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta1k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta1k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta1k0->SetMarkerStyle(20);
      Gforhejeta1k0->SetMarkerSize(0.4);
      Gforhejeta1k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta1k0->SetXTitle("Gforhejeta1k0 \b");
      Gforhejeta1k0->SetMarkerColor(2);
      Gforhejeta1k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta1k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta1k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 25 HE: j = 3,4,5, 6, 7      14,15,16,17,18          5
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta2k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta2k *********************************************************************       jeta == 5    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta2k0 = new TH1F("h2CeffGforhejeta2k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 5) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta2k0 = new TH1F("Gforhejeta2k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta2k0 = (TH1F*)h2CeffGforhejeta2k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta2k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta2k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta2k0->SetMarkerStyle(20);
      Gforhejeta2k0->SetMarkerSize(0.4);
      Gforhejeta2k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta2k0->SetXTitle("Gforhejeta2k0 \b");
      Gforhejeta2k0->SetMarkerColor(2);
      Gforhejeta2k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta2k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta2k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 26 HE: j = 3,4,5, 6, 7      14,15,16,17,18            6
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta3k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta3k *********************************************************************       jeta ==   6  "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta3k0 = new TH1F("h2CeffGforhejeta3k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 6) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta3k0 = new TH1F("Gforhejeta3k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta3k0 = (TH1F*)h2CeffGforhejeta3k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta3k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta3k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta3k0->SetMarkerStyle(20);
      Gforhejeta3k0->SetMarkerSize(0.4);
      Gforhejeta3k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta3k0->SetXTitle("Gforhejeta3k0 \b");
      Gforhejeta3k0->SetMarkerColor(2);
      Gforhejeta3k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta3k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta3k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 27 HE: j = 3,4,5, 6, 7      14,15,16,17,18    7
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta18k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta18k *********************************************************************       jeta ==  7    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta18k0 = new TH1F("h2CeffGforhejeta18k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 7) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta18k0 = new TH1F("Gforhejeta18k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta18k0 = (TH1F*)h2CeffGforhejeta18k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta18k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta18k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta18k0->SetMarkerStyle(20);
      Gforhejeta18k0->SetMarkerSize(0.4);
      Gforhejeta18k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta18k0->SetXTitle("Gforhejeta18k0 \b");
      Gforhejeta18k0->SetMarkerColor(2);
      Gforhejeta18k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta18k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta18k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 28 HE: j = 3,4,5, 6, 7      14,15,16,17,18           14
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta19k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta19k *********************************************************************       jeta == 14    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta19k0 = new TH1F("h2CeffGforhejeta19k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 14) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta19k0 = new TH1F("Gforhejeta19k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta19k0 = (TH1F*)h2CeffGforhejeta19k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta19k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta19k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta19k0->SetMarkerStyle(20);
      Gforhejeta19k0->SetMarkerSize(0.4);
      Gforhejeta19k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta19k0->SetXTitle("Gforhejeta19k0 \b");
      Gforhejeta19k0->SetMarkerColor(2);
      Gforhejeta19k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta19k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta19k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 29 HE: j = 3,4,5, 6, 7      14,15,16,17,18     15
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta20k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta20k *********************************************************************       jeta == 15    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta20k0 = new TH1F("h2CeffGforhejeta20k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 15) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta20k0 = new TH1F("Gforhejeta20k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta20k0 = (TH1F*)h2CeffGforhejeta20k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta20k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta20k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta20k0->SetMarkerStyle(20);
      Gforhejeta20k0->SetMarkerSize(0.4);
      Gforhejeta20k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta20k0->SetXTitle("Gforhejeta20k0 \b");
      Gforhejeta20k0->SetMarkerColor(2);
      Gforhejeta20k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta20k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta20k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 30 HE: j = 3,4,5, 6, 7      14,15,16,17,18    16
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta21k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta21k *********************************************************************       jeta ==  16   "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta21k0 = new TH1F("h2CeffGforhejeta21k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 16) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta21k0 = new TH1F("Gforhejeta21k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta21k0 = (TH1F*)h2CeffGforhejeta21k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta21k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta21k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta21k0->SetMarkerStyle(20);
      Gforhejeta21k0->SetMarkerSize(0.4);
      Gforhejeta21k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta21k0->SetXTitle("Gforhejeta21k0 \b");
      Gforhejeta21k0->SetMarkerColor(2);
      Gforhejeta21k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta21k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta21k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 31 HE: j = 3,4,5, 6, 7      14,15,16,17,18         17
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta22k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta22k *********************************************************************       jeta == 17    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta22k0 = new TH1F("h2CeffGforhejeta22k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 17) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta22k0 = new TH1F("Gforhejeta22k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta22k0 = (TH1F*)h2CeffGforhejeta22k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta22k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta22k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta22k0->SetMarkerStyle(20);
      Gforhejeta22k0->SetMarkerSize(0.4);
      Gforhejeta22k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta22k0->SetXTitle("Gforhejeta22k0 \b");
      Gforhejeta22k0->SetMarkerColor(2);
      Gforhejeta22k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta22k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta22k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 32 HE: j = 3,4,5, 6, 7      14,15,16,17,18          18
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Sefz1->GetXaxis()->GetNbins();
  ny = Sefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsohe > 0.)      nhistohe /= nlsohe;
  //    cout<<"HE Gforhejeta23k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohe=     "<< nhistohe <<endl;
  kcount = 1;
  cout << "HE Gforhejeta23k *********************************************************************       jeta == 18    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhejeta23k0 = new TH1F("h2CeffGforhejeta23k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 18) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhejeta23k0 = new TH1F("Gforhejeta23k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhejeta23k0 = (TH1F*)h2CeffGforhejeta23k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Sefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HE  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhejeta23k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HE Gforhejeta23k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhejeta23k0->SetMarkerStyle(20);
      Gforhejeta23k0->SetMarkerSize(0.4);
      Gforhejeta23k0->GetYaxis()->SetLabelSize(0.04);
      Gforhejeta23k0->SetXTitle("Gforhejeta23k0 \b");
      Gforhejeta23k0->SetMarkerColor(2);
      Gforhejeta23k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhejeta23k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhejeta23k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 33    HO - "h_2DsumADCAmplEtaPhiLs2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH2F* Yefz1KKK = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs2");
  TH2F* Yefz1LLL = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs20");
  TH2F* Yefz1 = (TH2F*)Yefz1LLL->Clone("Yefz1");
  Yefz1->Divide(Yefz1KKK, Yefz1LLL, 1, 1, "B");  // average A
  Yefz1->Sumw2();

  c1->cd(1);
  maxbinx = 0;
  maxbiny = 0;
  int sumijho = 0;
  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HO h_2DsumADCAmplEtaPhiLs0         nx=     " << nx << " ny=     " << ny << endl;
  // i - # LSs:
  TH1F* Yefw0 = new TH1F("Yefw0", "", 200, 0., 1000.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Yefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        sumijho++;
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HO h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Yefw0->Fill(ccc1);
      }
    }
  }
  cout << "HO maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << "     sumijho=  " << sumijho << endl;
  Yefw0->SetMarkerStyle(20);
  Yefw0->SetMarkerSize(0.4);
  Yefw0->GetYaxis()->SetLabelSize(0.04);
  Yefw0->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Yefw0->SetYTitle("     HO \b");
  Yefw0->SetMarkerColor(2);
  Yefw0->SetLineColor(0);
  //  Yefw0->SetMinimum(10.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Yefw0 ->Draw("L");
  Yefw0->Draw("Error");
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(2);
  TH1F* Yefw = new TH1F("Yefw", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    double sumsum = 0.;
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Yefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        //	    if(ccc1> 2500.) cout<<"HO :  i=  "<< i <<"      j= "<< j <<"  ccc1 = "<< ccc1 <<"  ccc1*maxbinx/sumijho = "<< ccc1*maxbinx/sumijho <<endl;
        //	 	  Yefw ->Fill(ccc1/maxbinx);
        sumsum += ccc1 * maxbinx / sumijho;
        //	    Yefw ->Fill( float(i), ccc1*maxbinx/sumijho);
      }
    }
    //	cout<<"HO :  i=  "<< i <<"  sumsum = "<< sumsum <<endl;
    Yefw->Fill(float(i), sumsum);
  }
  Yefw->SetMarkerStyle(20);
  Yefw->SetMarkerSize(0.4);
  Yefw->GetYaxis()->SetLabelSize(0.04);
  Yefw->SetMarkerColor(2);
  Yefw->SetLineColor(0);
  Yefw->SetXTitle("        iLS  \b");
  Yefw->SetYTitle("     <A>k \b");
  //Yefw->SetMinimum(0.8);Yefw->SetMaximum(500.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Yefw->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(3);
  TH1F* Yefw1 = new TH1F("Yefw1", "", 100, 0., 200.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Yefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HO h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Yefw1->Fill(ccc1);
      }
    }
  }
  cout << "HO maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << endl;
  Yefw1->SetMarkerStyle(20);
  Yefw1->SetMarkerSize(0.4);
  Yefw1->GetYaxis()->SetLabelSize(0.04);
  Yefw1->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Yefw1->SetMarkerColor(2);
  Yefw1->SetLineColor(0);
  //  Yefw1->SetMinimum(0.8);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Yefw1 ->Draw("L");
  Yefw1->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(4);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;
  TH2F* Yefz4 = new TH2F("Yefz4", "", 22, -11., 11., 18, 0., 18.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Yefz1->GetBinContent(i, j);
      //if(ccc1>0.) cout<<"HO h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1/maxbinx <<endl;
      //	if(ccc1>0. && ccc1/maxbinx < 2000) {
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	    if(i==1) cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta-11 <<" jphi= "<< jphi-1 <<"  A= "<< ccc1/maxbinx <<endl;
        //	  Yefz4 ->Fill(jeta-11,jphi-1,ccc1/maxbinx);
        Yefz4->Fill(jeta - 11, jphi - 1, ccc1 * maxbiny / sumijho);
      }
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Yefz4->SetMarkerStyle(20);
  Yefz4->SetMarkerSize(0.4);
  Yefz4->GetZaxis()->SetLabelSize(0.08);
  Yefz4->SetXTitle("<A>ij         #eta  \b");
  Yefz4->SetYTitle("      #phi \b");
  Yefz4->SetZTitle("<A>ij  - All \b");
  Yefz4->SetMarkerColor(2);
  Yefz4->SetLineColor(2);  //    Yefz4->SetMaximum(180.0);        Yefz4->SetMinimum(80.0);
  Yefz4->Draw("COLZ");

  c1->Update();

  //======================================================================

  //========================================================================================== 34 HO
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  c1->cd(1);
  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HO Yefk        nx=     " << nx << " ny=     " << ny << endl;
  TH1F* Yefk0 = new TH1F("Yefk0", "", maxbins, 1., maxbins + 1.);
  TH1F* Yefk1 = new TH1F("Yefk1", "", maxbins, 1., maxbins + 1.);
  TH1F* Yefk2 = new TH1F("Yefk2", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Yefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	    cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
        if (jeta == 7 && jphi == 11)
          Yefk0->Fill(i, ccc1);
        if (jeta == 10 && jphi == 11)
          Yefk1->Fill(i, ccc1);
        if (jeta == 12 && jphi == 11)
          Yefk2->Fill(i, ccc1);
      }
    }
  }
  Yefk0->SetMarkerStyle(20);
  Yefk0->SetMarkerSize(0.4);
  Yefk0->GetYaxis()->SetLabelSize(0.04);
  Yefk0->SetXTitle("Yefk0 \b");
  Yefk0->SetMarkerColor(2);
  Yefk0->SetLineColor(
      0);  // Yefk0 ->SetMaximum(30.0);// Yefk0 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Yefk0->Draw("Error");

  //================
  c1->cd(2);
  Yefk1->SetMarkerStyle(20);
  Yefk1->SetMarkerSize(0.4);
  Yefk1->GetYaxis()->SetLabelSize(0.04);
  Yefk1->SetXTitle("Yefk1 \b");
  Yefk1->SetMarkerColor(2);
  Yefk1->SetLineColor(
      0);  // Yefk1 ->SetMaximum(30.0);// Yefk1 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Yefk1->Draw("Error");

  //================
  c1->cd(3);
  Yefk2->SetMarkerStyle(20);
  Yefk2->SetMarkerSize(0.4);
  Yefk2->GetYaxis()->SetLabelSize(0.04);
  Yefk2->SetXTitle("Yefk2 \b");
  Yefk2->SetMarkerColor(2);
  Yefk2->SetLineColor(
      0);  // Yefk2 ->SetMaximum(30.0);// Yefk2 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Yefk2->Draw("Error");

  ////////////////////////////////////////////////////////////////////////////////////

  c1->Update();

  //========================================================================================== 35 HO: j = 7,8,9,10            11,12,13,14      7
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta0k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta0k *********************************************************************       jeta == 7    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta0k0 = new TH1F("h2CeffGforhojeta0k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 7) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta0k0 = new TH1F("Gforhojeta0k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta0k0 = (TH1F*)h2CeffGforhojeta0k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta0k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta0k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta0k0->SetMarkerStyle(20);
      Gforhojeta0k0->SetMarkerSize(0.4);
      Gforhojeta0k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta0k0->SetXTitle("Gforhojeta0k0 \b");
      Gforhojeta0k0->SetMarkerColor(2);
      Gforhojeta0k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta0k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta0k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 36 HO: j = 7,8,9,10            11,12,13,14      8
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta1k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta1k *********************************************************************       jeta == 8    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta1k0 = new TH1F("h2CeffGforhojeta1k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 8) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta1k0 = new TH1F("Gforhojeta1k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta1k0 = (TH1F*)h2CeffGforhojeta1k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta1k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta1k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta1k0->SetMarkerStyle(20);
      Gforhojeta1k0->SetMarkerSize(0.4);
      Gforhojeta1k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta1k0->SetXTitle("Gforhojeta1k0 \b");
      Gforhojeta1k0->SetMarkerColor(2);
      Gforhojeta1k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta1k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta1k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 37  HO: j = 7,8,9,10            11,12,13,14      9
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta2k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta2k *********************************************************************       jeta == 9    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta2k0 = new TH1F("h2CeffGforhojeta2k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 9) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta2k0 = new TH1F("Gforhojeta2k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta2k0 = (TH1F*)h2CeffGforhojeta2k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta2k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta2k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta2k0->SetMarkerStyle(20);
      Gforhojeta2k0->SetMarkerSize(0.4);
      Gforhojeta2k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta2k0->SetXTitle("Gforhojeta2k0 \b");
      Gforhojeta2k0->SetMarkerColor(2);
      Gforhojeta2k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta2k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta2k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 38  HO: j = 7,8,9,10            11,12,13,14    10
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta3k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta3k *********************************************************************       jeta == 10   "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta3k0 = new TH1F("h2CeffGforhojeta3k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 10) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta3k0 = new TH1F("Gforhojeta3k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta3k0 = (TH1F*)h2CeffGforhojeta3k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta3k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta3k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta3k0->SetMarkerStyle(20);
      Gforhojeta3k0->SetMarkerSize(0.4);
      Gforhojeta3k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta3k0->SetXTitle("Gforhojeta3k0 \b");
      Gforhojeta3k0->SetMarkerColor(2);
      Gforhojeta3k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta3k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta3k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 39  HO: j = 7,8,9,10            11,12,13,14    11
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta18k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta18k *********************************************************************       jeta == 11    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta18k0 = new TH1F("h2CeffGforhojeta18k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 11) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta18k0 = new TH1F("Gforhojeta18k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta18k0 = (TH1F*)h2CeffGforhojeta18k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta18k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta18k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta18k0->SetMarkerStyle(20);
      Gforhojeta18k0->SetMarkerSize(0.4);
      Gforhojeta18k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta18k0->SetXTitle("Gforhojeta18k0 \b");
      Gforhojeta18k0->SetMarkerColor(2);
      Gforhojeta18k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta18k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta18k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 40  HO: j = 7,8,9,10            11,12,13,14   12
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta19k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta19k *********************************************************************       jeta == 12    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta19k0 = new TH1F("h2CeffGforhojeta19k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 12) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta19k0 = new TH1F("Gforhojeta19k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta19k0 = (TH1F*)h2CeffGforhojeta19k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta19k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta19k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta19k0->SetMarkerStyle(20);
      Gforhojeta19k0->SetMarkerSize(0.4);
      Gforhojeta19k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta19k0->SetXTitle("Gforhojeta19k0 \b");
      Gforhojeta19k0->SetMarkerColor(2);
      Gforhojeta19k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta19k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta19k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 41  HO: j = 7,8,9,10            11,12,13,14    13
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta20k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta20k *********************************************************************       jeta == 13    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta20k0 = new TH1F("h2CeffGforhojeta20k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 13) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta20k0 = new TH1F("Gforhojeta20k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta20k0 = (TH1F*)h2CeffGforhojeta20k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta20k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta20k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta20k0->SetMarkerStyle(20);
      Gforhojeta20k0->SetMarkerSize(0.4);
      Gforhojeta20k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta20k0->SetXTitle("Gforhojeta20k0 \b");
      Gforhojeta20k0->SetMarkerColor(2);
      Gforhojeta20k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta20k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta20k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 42  HO: j = 7,8,9,10            11,12,13,14    14
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Yefz1->GetXaxis()->GetNbins();
  ny = Yefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //    if( nlsoho > 0.)      nhistoho /= nlsoho;
  //    cout<<"HO Gforhojeta21k        nx=     "<< nx <<" ny=     "<< ny <<" nhistoho=     "<< nhistoho <<endl;
  kcount = 1;
  cout << "HO Gforhojeta21k *********************************************************************       jeta == 14    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhojeta21k0 = new TH1F("h2CeffGforhojeta21k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 14) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhojeta21k0 = new TH1F("Gforhojeta21k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhojeta21k0 = (TH1F*)h2CeffGforhojeta21k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Yefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HO  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhojeta21k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HO Gforhojeta21k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhojeta21k0->SetMarkerStyle(20);
      Gforhojeta21k0->SetMarkerSize(0.4);
      Gforhojeta21k0->GetYaxis()->SetLabelSize(0.04);
      Gforhojeta21k0->SetXTitle("Gforhojeta21k0 \b");
      Gforhojeta21k0->SetMarkerColor(2);
      Gforhojeta21k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhojeta21k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhojeta21k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 43    HF - "h_2DsumADCAmplEtaPhiLs3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH2F* Gefz1KKK = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs3");
  TH2F* Gefz1LLL = (TH2F*)hfile1->Get("h_2DsumADCAmplEtaPhiLs30");
  TH2F* Gefz1 = (TH2F*)Gefz1LLL->Clone("Gefz1");
  Gefz1->Divide(Gefz1KKK, Gefz1LLL, 1, 1, "B");  // average A
  Gefz1->Sumw2();

  c1->cd(1);
  maxbinx = 0;
  maxbiny = 0;
  int sumijhf = 0;
  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HF h_2DsumADCAmplEtaPhiLs0         nx=     " << nx << " ny=     " << ny << endl;
  // i - # LSs:
  TH1F* Gefw0 = new TH1F("Gefw0", "", 250, 0., 1500.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        sumijhf++;
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HF h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Gefw0->Fill(ccc1);
      }
    }
  }
  cout << "HF maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << "     sumijhf=  " << sumijhf << endl;
  Gefw0->SetMarkerStyle(20);
  Gefw0->SetMarkerSize(0.4);
  Gefw0->GetYaxis()->SetLabelSize(0.04);
  Gefw0->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Gefw0->SetYTitle("     HF \b");
  Gefw0->SetMarkerColor(2);
  Gefw0->SetLineColor(0);
  //  Gefw0->SetMinimum(10.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Gefw0 ->Draw("L");
  Gefw0->Draw("Error");
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(2);
  TH1F* Gefw = new TH1F("Gefw", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        //	  cout<<"HF h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        //	  Gefw ->Fill(ccc1/maxbinx);
        Gefw->Fill(float(i), ccc1* maxbinx / sumijhf);
      }
    }
  }
  Gefw->SetMarkerStyle(20);
  Gefw->SetMarkerSize(0.4);
  Gefw->GetYaxis()->SetLabelSize(0.04);
  Gefw->SetMarkerColor(2);
  Gefw->SetLineColor(0);
  Gefw->SetXTitle("        iLS  \b");
  Gefw->SetYTitle("     <A>k \b");
  //Gefw->SetMinimum(0.8);Gefw->SetMaximum(500.);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  Gefw->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(3);
  TH1F* Gefw1 = new TH1F("Gefw1", "", 150, 0., 500.);
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        maxbinx = i;
        if (i > maxbinx)
          maxbinx = i;
        maxbiny = j;
        if (j > maxbiny)
          maxbiny = j;
        //	  cout<<"HF h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1 <<endl;
        Gefw1->Fill(ccc1);
      }
    }
  }
  cout << "HF maxbinx=  " << maxbinx << "     maxbiny=  " << maxbiny << endl;
  Gefw1->SetMarkerStyle(20);
  Gefw1->SetMarkerSize(0.4);
  Gefw1->GetYaxis()->SetLabelSize(0.04);
  Gefw1->SetXTitle("<A>ijk = <A> averaged per events in k-th LS \b");
  Gefw1->SetMarkerColor(2);
  Gefw1->SetLineColor(0);
  // Gefw1->SetMinimum(0.8);
  gPad->SetLogy();
  gPad->SetGridy();
  gPad->SetGridx();
  //      Gefw1 ->Draw("L");
  Gefw1->Draw("Error");

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(4);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;
  TH2F* Gefz4 = new TH2F("Gefz4", "", 22, -11., 11., 18, 0., 18.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      //if(ccc1>0.) cout<<"HF h_2DsumADCAmplEtaPhiLs:  ibin=  "<< i <<"      jbin= "<< j <<"  A= "<< ccc1/maxbinx <<endl;
      //	if(ccc1>0. && ccc1/maxbinx < 2000) {
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	  	  if(i==1) cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta-11 <<" jphi= "<< jphi-1 <<"  A= "<< ccc1/maxbinx <<endl;
        Gefz4->Fill(jeta - 11, jphi - 1, ccc1 * maxbiny / sumijhf);
        //	  Gefz4 ->Fill(jeta-11,jphi-1,ccc1/maxbinx);
      }
    }
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gefz4->SetMarkerStyle(20);
  Gefz4->SetMarkerSize(0.4);
  Gefz4->GetZaxis()->SetLabelSize(0.08);
  Gefz4->SetXTitle("<A>_RBX         #eta  \b");
  Gefz4->SetYTitle("      #phi \b");
  Gefz4->SetZTitle("<A>_RBX  - All \b");
  Gefz4->SetMarkerColor(2);
  Gefz4->SetLineColor(2);  //      Gefz4->SetMaximum(1.000);  //      Gefz4->SetMinimum(1.0);
  Gefz4->Draw("COLZ");

  c1->Update();

  //======================================================================

  //========================================================================================== 44 HF
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  c1->cd(1);
  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HF Gefk        nx=     " << nx << " ny=     " << ny << endl;
  TH1F* Gefk0 = new TH1F("Gefk0", "", maxbins, 1., maxbins + 1.);
  TH1F* Gefk1 = new TH1F("Gefk1", "", maxbins, 1., maxbins + 1.);
  TH1F* Gefk2 = new TH1F("Gefk2", "", maxbins, 1., maxbins + 1.);
  // i - # LSs:
  for (int i = 1; i <= nx; i++) {
    // j - etaphi index:
    for (int j = 1; j <= ny; j++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        int jeta = (j - 1) / 18;             // jeta = 0-21
        int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
        //	  jeta += 1;// jeta = 1-22
        //	    cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
        if (jeta == 1 && jphi == 11)
          Gefk0->Fill(i, ccc1);
        if (jeta == 2 && jphi == 11)
          Gefk1->Fill(i, ccc1);
        if (jeta == 3 && jphi == 11)
          Gefk2->Fill(i, ccc1);
      }
    }
  }
  Gefk0->SetMarkerStyle(20);
  Gefk0->SetMarkerSize(0.4);
  Gefk0->GetYaxis()->SetLabelSize(0.04);
  Gefk0->SetXTitle("Gefk0 \b");
  Gefk0->SetMarkerColor(2);
  Gefk0->SetLineColor(
      0);  // Gefk0 ->SetMaximum(30.0);// Gefk0 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefk0->Draw("Error");

  //================
  c1->cd(2);
  Gefk1->SetMarkerStyle(20);
  Gefk1->SetMarkerSize(0.4);
  Gefk1->GetYaxis()->SetLabelSize(0.04);
  Gefk1->SetXTitle("Gefk1 \b");
  Gefk1->SetMarkerColor(2);
  Gefk1->SetLineColor(
      0);  // Gefk1 ->SetMaximum(30.0);// Gefk1 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefk1->Draw("Error");

  //================
  c1->cd(3);
  Gefk2->SetMarkerStyle(20);
  Gefk2->SetMarkerSize(0.4);
  Gefk2->GetYaxis()->SetLabelSize(0.04);
  Gefk2->SetXTitle("Gefk2 \b");
  Gefk2->SetMarkerColor(2);
  Gefk2->SetLineColor(
      0);  // Gefk2 ->SetMaximum(30.0);// Gefk2 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefk2->Draw("Error");

  ////////////////////////////////////////////////////////////////////////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 45 HF
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(1, 3);

  //	    cout<<"  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
  c1->cd(1);
  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;
  cout << "HF Gefh        nx=     " << nx << " ny=     " << ny << endl;
  TH1F* Gefh0 = new TH1F("Gefh0", "", maxbins, 1., maxbins + 1.);
  TH1F* Gefh1 = new TH1F("Gefh1", "", maxbins, 1., maxbins + 1.);
  TH1F* Gefh2 = new TH1F("Gefh2", "", maxbins, 1., maxbins + 1.);
  // j - etaphi index:
  int nhistohf = 0.;
  int nlsohf = -1.;
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;             // jeta = 0-21
    int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
    //	  jeta += 1;// jeta = 1-22
    double sumj = 0.;
    double ssumj = 0.;
    int nj = 0;
    double meanj = 0.;
    double sigmaj = 0.;
    // i - # LSs:
    for (int i = 1; i <= nx; i++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        sumj += ccc1;
        nj++;
        nhistohf++;
      }
    }
    meanj = sumj / nj;
    if (nj > nlsohf)
      nlsohf = nj;
    // i - # LSs:
    for (int i = 1; i <= nx; i++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        ssumj += (ccc1 - meanj) * (ccc1 - meanj);
      }
    }
    sigmaj = sqrt(ssumj / nj);
    // i - # LSs:
    for (int i = 1; i <= nx; i++) {
      double ccc1 = Gefz1->GetBinContent(i, j);
      if (ccc1 > 0.) {
        double dif3rmsMIN = meanj - 3 * sigmaj;
        if (dif3rmsMIN < 0.)
          dif3rmsMIN = 0.;
        double dif3rmsMAX = meanj + 3 * sigmaj;
        if (jeta == 1 && jphi == 11 && (ccc1 < dif3rmsMIN || ccc1 > dif3rmsMAX))
          Gefh0->Fill(i, ccc1);
        if (jeta == 2 && jphi == 11 && (ccc1 < dif3rmsMIN || ccc1 > dif3rmsMAX))
          Gefh1->Fill(i, ccc1);
        if (jeta == 3 && jphi == 11 && (ccc1 < dif3rmsMIN || ccc1 > dif3rmsMAX))
          Gefh2->Fill(i, ccc1);
      }
    }
  }
  cout << "HF 45        nhistohf =     " << nhistohf << "           nlsohf =     " << nlsohf << endl;

  Gefh0->SetMarkerStyle(20);
  Gefh0->SetMarkerSize(0.4);
  Gefh0->GetYaxis()->SetLabelSize(0.04);
  Gefh0->SetXTitle("Gefh0 \b");
  Gefh0->SetMarkerColor(2);
  Gefh0->SetLineColor(
      0);  // Gefh0 ->SetMaximum(30.0);// Gefh0 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefh0->Draw("Error");
  //================
  c1->cd(2);
  Gefh1->SetMarkerStyle(20);
  Gefh1->SetMarkerSize(0.4);
  Gefh1->GetYaxis()->SetLabelSize(0.04);
  Gefh1->SetXTitle("Gefh1 \b");
  Gefh1->SetMarkerColor(2);
  Gefh1->SetLineColor(
      0);  // Gefh1 ->SetMaximum(30.0);// Gefh1 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefh1->Draw("Error");
  //================
  c1->cd(3);
  Gefh2->SetMarkerStyle(20);
  Gefh2->SetMarkerSize(0.4);
  Gefh2->GetYaxis()->SetLabelSize(0.04);
  Gefh2->SetXTitle("Gefh2 \b");
  Gefh2->SetMarkerColor(2);
  Gefh2->SetLineColor(
      0);  // Gefh2 ->SetMaximum(30.0);// Gefh2 ->SetMinimum(20.0); // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
  Gefh2->Draw("Error");

  ////////////////////////////////////////////////////////////////////////////////////
  c1->Update();

  //========================================================================================== 46 HF: j = 0,1,2, 3            18,19,20,21          0
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  if (nlsohf > 0.)
    nhistohf /= nlsohf;
  cout << "HF Gforhfjeta0k        nx=     " << nx << " ny=     " << ny << " nhistohf=     " << nhistohf << endl;
  kcount = 1;
  cout << "HF Gforhfjeta0k *********************************************************************       jeta == 0    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta0k0 = new TH1F("h2CeffGforhfjeta0k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 0) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta0k0 = new TH1F("Gforhfjeta0k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta0k0 = (TH1F*)h2CeffGforhfjeta0k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta0k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta0k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta0k0->SetMarkerStyle(20);
      Gforhfjeta0k0->SetMarkerSize(0.4);
      Gforhfjeta0k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta0k0->SetXTitle("Gforhfjeta0k0 \b");
      Gforhfjeta0k0->SetMarkerColor(2);
      Gforhfjeta0k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta0k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta0k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 47 HF: j = 0,1,2, 3            18,19,20,21      1
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta1k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta1k *********************************************************************       jeta == 1    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta1k0 = new TH1F("h2CeffGforhfjeta1k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 1) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta1k0 = new TH1F("Gforhfjeta1k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta1k0 = (TH1F*)h2CeffGforhfjeta1k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta1k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta1k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta1k0->SetMarkerStyle(20);
      Gforhfjeta1k0->SetMarkerSize(0.4);
      Gforhfjeta1k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta1k0->SetXTitle("Gforhfjeta1k0 \b");
      Gforhfjeta1k0->SetMarkerColor(2);
      Gforhfjeta1k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta1k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta1k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 48   HF: j = 0,1,2, 3            18,19,20,21    2
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta2k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta2k *********************************************************************       jeta == 2    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta2k0 = new TH1F("h2CeffGforhfjeta2k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 2) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta2k0 = new TH1F("Gforhfjeta2k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta2k0 = (TH1F*)h2CeffGforhfjeta2k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta2k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta2k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta2k0->SetMarkerStyle(20);
      Gforhfjeta2k0->SetMarkerSize(0.4);
      Gforhfjeta2k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta2k0->SetXTitle("Gforhfjeta2k0 \b");
      Gforhfjeta2k0->SetMarkerColor(2);
      Gforhfjeta2k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta2k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta2k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 49 HF: j = 0,1,2, 3            18,19,20,21   3
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta3k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta3k *********************************************************************       jeta == 3    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta3k0 = new TH1F("h2CeffGforhfjeta3k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 3) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta3k0 = new TH1F("Gforhfjeta3k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta3k0 = (TH1F*)h2CeffGforhfjeta3k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta3k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta3k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta3k0->SetMarkerStyle(20);
      Gforhfjeta3k0->SetMarkerSize(0.4);
      Gforhfjeta3k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta3k0->SetXTitle("Gforhfjeta3k0 \b");
      Gforhfjeta3k0->SetMarkerColor(2);
      Gforhfjeta3k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta3k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta3k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 50   HF: j = 0,1,2, 3            18,19,20,21      18
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta18k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta18k *********************************************************************       jeta == 18    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta18k0 = new TH1F("h2CeffGforhfjeta18k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 18) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta18k0 = new TH1F("Gforhfjeta18k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta18k0 = (TH1F*)h2CeffGforhfjeta18k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta18k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta18k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta18k0->SetMarkerStyle(20);
      Gforhfjeta18k0->SetMarkerSize(0.4);
      Gforhfjeta18k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta18k0->SetXTitle("Gforhfjeta18k0 \b");
      Gforhfjeta18k0->SetMarkerColor(2);
      Gforhfjeta18k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta18k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta18k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 51   HF: j = 0,1,2, 3            18,19,20,21     19
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta19k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta19k *********************************************************************       jeta == 19    "
       << endl;
  // j - etaphi index:
  TH1F* h2CeffGforhfjeta19k0 = new TH1F("h2CeffGforhfjeta19k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 19) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta19k0 = new TH1F("Gforhfjeta19k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta19k0 = (TH1F*)h2CeffGforhfjeta19k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta19k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta19k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta19k0->SetMarkerStyle(20);
      Gforhfjeta19k0->SetMarkerSize(0.4);
      Gforhfjeta19k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta19k0->SetXTitle("Gforhfjeta19k0 \b");
      Gforhfjeta19k0->SetMarkerColor(2);
      Gforhfjeta19k0->SetLineColor(0);
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta19k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta19k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 52  HF: j = 0,1,2, 3            18,19,20,21   20
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);

  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta20k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta20k *********************************************************************       jeta == 20    "
       << endl;
  TH1F* h2CeffGforhfjeta20k0 = new TH1F("h2CeffGforhfjeta20k0", "", maxbins, 1., maxbins + 1.);
  // j - etaphi index:
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 20) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta20k0 = new TH1F("Gforhfjeta20k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta20k0 = (TH1F*)h2CeffGforhfjeta20k0->Clone("twod1");
      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta20k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta20k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta20k0->SetMarkerStyle(20);
      Gforhfjeta20k0->SetMarkerSize(0.4);
      Gforhfjeta20k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta20k0->SetXTitle("Gforhfjeta20k0 \b");
      Gforhfjeta20k0->SetMarkerColor(2);
      Gforhfjeta20k0->SetLineColor(0);
      gPad->SetGridx();
      // gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();
      Gforhfjeta20k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta20k0;
      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 53   HF: j = 0,1,2, 3            18,19,20,21    21
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  c1->Divide(3, 6);
  nx = Gefz1->GetXaxis()->GetNbins();
  ny = Gefz1->GetYaxis()->GetNbins();
  nx = maxbins;

  //      if( nlsohf > 0.)      nhistohf /= nlsohf;
  //      cout<<"HF Gforhfjeta21k        nx=     "<< nx <<" ny=     "<< ny <<" nhistohf=     "<< nhistohf <<endl;
  kcount = 1;
  cout << "HF Gforhfjeta21k *********************************************************************       jeta == 21    "
       << endl;
  // j - etaphi index:
  //    TH1F *Gforhfjeta21k0= NULL;
  TH1F* h2CeffGforhfjeta21k0 = new TH1F("h2CeffGforhfjeta21k0", "", maxbins, 1., maxbins + 1.);
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / 18;  // jeta = 0-21
    if (jeta == 21) {
      int jphi = (j - 1) - 18 * jeta + 1;  // jphi=1-18
                                           //	  jeta += 1;// jeta = 1-22
      //	  TH1F* Gforhfjeta21k0 = new TH1F("Gforhfjeta21k0","", maxbins, 1., maxbins+1.);
      TH1F* Gforhfjeta21k0 = (TH1F*)h2CeffGforhfjeta21k0->Clone("twod1");

      //    h2Ceff = (TH2F*)twod1->Clone(Form("Ceff_HF%d",depth));
      //    h2Ceff->SetTitle(Form("HF Depth %d. (No cut) \b",depth));
      //    h2Ceff->Divide(twod1,twod0, 1, 1, "B");

      // i - # LSs:
      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        if (ccc1 > 0.) {
          //	      if(i==1)cout<<"HF  i= "<< i <<" j= "<< j <<"  jeta= "<< jeta <<" jphi= "<< jphi <<"      A= "<< ccc1 <<endl;
          Gforhfjeta21k0->Fill(i, ccc1);
        }
      }
      c1->cd(kcount);
      //	  cout<<"HF Gforhfjeta21k        kcount=     "<< kcount <<"   jphi   =     "<<jphi  <<endl;
      Gforhfjeta21k0->SetMarkerStyle(20);
      Gforhfjeta21k0->SetMarkerSize(0.4);
      Gforhfjeta21k0->GetYaxis()->SetLabelSize(0.04);
      Gforhfjeta21k0->SetXTitle("Gforhfjeta21k0 \b");
      Gforhfjeta21k0->SetMarkerColor(2);
      Gforhfjeta21k0->SetLineColor(0);
      gPad->SetGridy();
      gPad->SetGridx();
      //	   gPad->SetLogy();
      Gforhfjeta21k0->Draw("Error");
      kcount++;
      //	c1->Update();
      //		delete Gforhfjeta21k0;
      //	  if (Gforhfjeta21k0) delete Gforhfjeta21k0;

      if (kcount > 18)
        break;
    }
  }

  /////////////////
  c1->Update();
  //////////////////////////////////////////////////////////////////////////////////// 533    HF:: jeta = 0,1,2, 3            18,19,20,21       // jphi = 0,1,2,3,4,5.... 17
  //======================================================================
  //======================================================================
  //======================================================================
  /*
      c1->Clear();
      c1->Divide(1,1);
      c1->cd(1);
      maxbinx = 0;
      maxbiny = 0;
      nx = Gefz1->GetXaxis()->GetNbins();
      ny = Gefz1->GetYaxis()->GetNbins();
      nx = maxbins;// ls
      cout<<"533 HF1     LS   nx=     "<< nx <<" ny=     "<< ny <<endl;
      TH2F* ADCAmplLSHF1 = new TH2F("ADCAmplLSHF1","", 610, 0., 610.,400,0., 400.);
      TH2F* ADCAmplLSHF10 = new TH2F("ADCAmplLSHF10","", 610, 0., 610.,400,0., 400.);
       TH2F* ADCAmplLSHF2 = (TH2F*)ADCAmplLSHF10->Clone("ADCAmplLSHF2");
      for (int i=1;i<=nx;i++) {
	for (int j=1;j<=ny;j++) {
	  double ccc1 =  Gefz1->GetBinContent(i,j);
	  if(ccc1>0.) {
	    maxbinx = i; if(i>maxbinx) maxbinx = i;
	    maxbiny = j; if(j>maxbiny) maxbiny = j;
	    //	    	    if(ccc1 <= 0.)	    	    cout<<"HF1:  ibin=  "<< i <<"      jbin= "<< j <<"      A= "<< ccc1 <<endl;
	    //	    ADCAmplLSHF1 ->Fill(ccc1);
	    ADCAmplLSHF1 ->Fill(float(i), float(j),ccc1);
	    ADCAmplLSHF10 ->Fill(float(i), float(j),1.);
	  }
	}}
       ADCAmplLSHF2->Divide(ADCAmplLSHF1,ADCAmplLSHF10, 1, 1, "B");// average A
      ADCAmplLSHF2 ->SetMarkerStyle(20);
      ADCAmplLSHF2 ->SetMarkerSize(0.4);
      ADCAmplLSHF2 ->GetYaxis()->SetLabelSize(0.04);
      ADCAmplLSHF2 ->SetXTitle("nev0-overAllLSs test with ADCAmplLSHF1 \b");
      ADCAmplLSHF2 ->SetMarkerColor(2);
      ADCAmplLSHF2 ->SetLineColor(0);
      //    gPad->SetLogy();gPad->SetGridy();gPad->SetGridx();     
          ADCAmplLSHF2 ->Draw("COLZ");
      cout<<"533 HF1 for h_2D0sumADCAmplLS1 maxbinx =  "<< maxbinx<<"     maxbiny=  "<< maxbiny <<endl;
       c1->Update();
*/
  /*
       for (int j=1;j<=ny;j++) {
	 for (int i=1;i<=nx;i++) {
	   double ccc1 =  Gefz1->GetBinContent(i,j);
	   if(ccc1 <= 0.) cout<<"HF==================================*:  ibin=  "<< i <<"      jbin= "<< j <<"      A= "<< ccc1 <<endl;
	 }//i
       }//j
*/
  //======================================================================HF:: jeta = 0,1,2, 3            18,19,20,21       // jphi = 0,1,2,3,4,5.... 17
  int njeta = 22;
  int njphi = 18;
  //       ny = Gefz1->GetYaxis()->GetNbins();// # etaphi indexe
  //       nx = Gefz1->GetXaxis()->GetNbins();// # LS
  //       	   	   cout<<"HF 111 54        ny=     "<< ny <<"   nx   =     "<<nx  <<endl;
  //       nx = maxbins;
  //       	   	   cout<<"HF 222 54        ny=     "<< ny <<"   nx   =     "<<nx  <<endl;
  double alexhf[njeta][njphi][nx];
  for (int i = 0; i < nx; i++) {
    for (int jeta = 0; jeta < njeta; jeta++) {
      for (int jphi = 0; jphi < njphi; jphi++) {
        alexhf[jeta][jphi][i] = 0.;
      }
    }
  }
  for (int j = 1; j <= ny; j++) {
    int jeta = (j - 1) / njphi;  // jeta = 0-21
    if (jeta < 4 || jeta > 17) {
      int jphi = (j - 1) - njphi * jeta;  // jphi=0-17
      //	   	   	   cout<<"HF 54        jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;

      for (int i = 1; i <= nx; i++) {
        double ccc1 = Gefz1->GetBinContent(i, j);
        //	    	    if(ccc1 <= 0.)	    	    cout<<"HF*****************:  ibin=  "<< i <<"      jbin= "<< j <<"      A= "<< ccc1 <<endl;
        alexhf[jeta][jphi][i - 1] = ccc1;
        //	     	     if( i == 1 ) cout<<"HF 54  for LS=1      ccc1=     "<< ccc1 <<endl;
        //	     	     if( alexhf[jeta][jphi][i-1] <= 0. ) cout<<"HF 54        jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<"  j   =     "<<j  <<"  i-1   =     "<<i-1  <<"  ccc1   =     "<<ccc1  <<endl;
      }  //i
    }    //if
  }      //j
  //------------------------

  //========================================================================================== 54   HF:: jeta = 0,1,2, 3            18,19,20,21       // jphi = 0,1,2,3,4,5.... 17
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFnegativedirection1 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFnegativedirection1 = new TH1F("h2CeffHFnegativedirection1", "", maxbins, 1., maxbins + 1.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirection:
    if (jeta < 4) {
      // jphi = 0,1,2,3,4,5
      for (int jphi = 0; jphi < 6; jphi++) {
        //	   for (int jphi=0;jphi<njphi;jphi++) {
        //	     cout<<"HF 54 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
        TH1F* HFnegativedirection1 = (TH1F*)h2CeffHFnegativedirection1->Clone("twod1");
        for (int i = 0; i < nx; i++) {
          double ccc1 = alexhf[jeta][jphi][i];
          if (ccc1 > 0.) {
            HFnegativedirection1->Fill(i, ccc1);
          }
          //	       if( i == 0 ) cout<<"HF 54 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
        }  // for i
        c1->cd(kcountHFnegativedirection1);
        HFnegativedirection1->SetMarkerStyle(20);
        HFnegativedirection1->SetMarkerSize(0.4);
        HFnegativedirection1->GetYaxis()->SetLabelSize(0.04);
        HFnegativedirection1->SetXTitle("HFnegativedirection1 \b");
        HFnegativedirection1->SetMarkerColor(2);
        HFnegativedirection1->SetLineColor(0);
        gPad->SetGridy();
        gPad->SetGridx();
        //	   gPad->SetLogy();
        HFnegativedirection1->Draw("Error");
        kcountHFnegativedirection1++;
        if (kcountHFnegativedirection1 > 24)
          break;  // 4x6 = 24
      }           // for jphi
    }             //if
  }               //for jeta
  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 55   HF:: jeta = 0,1,2, 3            18,19,20,21       jphi = 6,7,8,9,10,11
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFnegativedirection2 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFnegativedirection2 = new TH1F("h2CeffHFnegativedirection2", "", maxbins, 1., maxbins + 1.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirection:
    if (jeta < 4) {
      // jphi = 6,7,8,9,10,11
      for (int jphi = 6; jphi < 12; jphi++) {
        //	       	     cout<<"HF 55 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<"   kcountHFnegativedirection2   =     "<<kcountHFnegativedirection2  <<endl;
        TH1F* HFnegativedirection2 = (TH1F*)h2CeffHFnegativedirection2->Clone("twod1");
        for (int i = 0; i < nx; i++) {
          double ccc1 = alexhf[jeta][jphi][i];
          if (ccc1 > 0.) {
            HFnegativedirection2->Fill(i, ccc1);
          }
          //		 	       if( i == 0 ) cout<<"HF 55 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
        }  // for i
        c1->cd(kcountHFnegativedirection2);
        HFnegativedirection2->SetMarkerStyle(20);
        HFnegativedirection2->SetMarkerSize(0.4);
        HFnegativedirection2->GetYaxis()->SetLabelSize(0.04);
        HFnegativedirection2->SetXTitle("HFnegativedirection2 \b");
        HFnegativedirection2->SetMarkerColor(2);
        HFnegativedirection2->SetLineColor(0);
        gPad->SetGridy();
        gPad->SetGridx();
        //	   gPad->SetLogy();
        HFnegativedirection2->Draw("Error");
        kcountHFnegativedirection2++;
        if (kcountHFnegativedirection2 > 24)
          break;  // 4x6 = 24
      }           // for jphi
    }             //if
  }               //for jeta
  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 56   HF:: jeta = 0,1,2, 3            18,19,20,21       jphi =12,13,14,15,16,17
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFnegativedirection3 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFnegativedirection3 = new TH1F("h2CeffHFnegativedirection3", "", maxbins, 1., maxbins + 1.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // negativedirection:
    if (jeta < 4) {
      // jphi = 12,13,14,15,16,17
      for (int jphi = 12; jphi < 18; jphi++) {
        //	       	     cout<<"HF 55 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<"   kcountHFnegativedirection3   =     "<<kcountHFnegativedirection3  <<endl;
        TH1F* HFnegativedirection3 = (TH1F*)h2CeffHFnegativedirection3->Clone("twod1");
        for (int i = 0; i < nx; i++) {
          double ccc1 = alexhf[jeta][jphi][i];
          if (ccc1 > 0.) {
            HFnegativedirection3->Fill(i, ccc1);
          }
          //		 	       if( i == 0 ) cout<<"HF 55 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
        }  // for i
        c1->cd(kcountHFnegativedirection3);
        HFnegativedirection3->SetMarkerStyle(20);
        HFnegativedirection3->SetMarkerSize(0.4);
        HFnegativedirection3->GetYaxis()->SetLabelSize(0.04);
        HFnegativedirection3->SetXTitle("HFnegativedirection3 \b");
        HFnegativedirection3->SetMarkerColor(2);
        HFnegativedirection3->SetLineColor(0);
        gPad->SetGridy();
        gPad->SetGridx();
        //	   gPad->SetLogy();
        HFnegativedirection3->Draw("Error");
        kcountHFnegativedirection3++;
        if (kcountHFnegativedirection3 > 24)
          break;  // 4x6 = 24
      }           // for jphi
    }             //if
  }               //for jeta
  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 57   HF:: jeta = 0,1,2, 3            18,19,20,21       // jphi = 0,1,2,3,4,5
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFpositivedirection1 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFpositivedirection1 = new TH1F("h2CeffHFpositivedirection1", "", maxbins, 1., maxbins + 1.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta > 17) {
      // jphi = 0,1,2,3,4,5
      for (int jphi = 0; jphi < 6; jphi++) {
        //	   for (int jphi=0;jphi<njphi;jphi++) {
        //	     cout<<"HF 54 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<endl;
        TH1F* HFpositivedirection1 = (TH1F*)h2CeffHFpositivedirection1->Clone("twod1");
        for (int i = 0; i < nx; i++) {
          double ccc1 = alexhf[jeta][jphi][i];
          if (ccc1 > 0.) {
            HFpositivedirection1->Fill(i, ccc1);
          }
          //	       if( i == 0 ) cout<<"HF 54 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
        }  // for i
        c1->cd(kcountHFpositivedirection1);
        HFpositivedirection1->SetMarkerStyle(20);
        HFpositivedirection1->SetMarkerSize(0.4);
        HFpositivedirection1->GetYaxis()->SetLabelSize(0.04);
        HFpositivedirection1->SetXTitle("HFpositivedirection1 \b");
        HFpositivedirection1->SetMarkerColor(2);
        HFpositivedirection1->SetLineColor(0);
        gPad->SetGridy();
        gPad->SetGridx();
        //	   gPad->SetLogy();
        HFpositivedirection1->Draw("Error");
        kcountHFpositivedirection1++;
        if (kcountHFpositivedirection1 > 24)
          break;  // 4x6 = 24
      }           // for jphi
    }             //if
  }               //for jeta
  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 58   HF:: jeta = 0,1,2, 3            18,19,20,21       jphi = 6,7,8,9,10,11
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFpositivedirection2 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFpositivedirection2 = new TH1F("h2CeffHFpositivedirection2", "", maxbins, 1., maxbins + 1.);
  for (int jeta = 0; jeta < njeta; jeta++) {
    // positivedirection:
    if (jeta > 17) {
      // jphi = 6,7,8,9,10,11
      for (int jphi = 6; jphi < 12; jphi++) {
        //	       	     cout<<"HF 55 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<"   kcountHFpositivedirection2   =     "<<kcountHFpositivedirection2  <<endl;
        TH1F* HFpositivedirection2 = (TH1F*)h2CeffHFpositivedirection2->Clone("twod1");
        for (int i = 0; i < nx; i++) {
          double ccc1 = alexhf[jeta][jphi][i];
          if (ccc1 > 0.) {
            HFpositivedirection2->Fill(i, ccc1);
          }
          //		 	       if( i == 0 ) cout<<"HF 55 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
        }  // for i
        c1->cd(kcountHFpositivedirection2);
        HFpositivedirection2->SetMarkerStyle(20);
        HFpositivedirection2->SetMarkerSize(0.4);
        HFpositivedirection2->GetYaxis()->SetLabelSize(0.04);
        HFpositivedirection2->SetXTitle("HFpositivedirection2 \b");
        HFpositivedirection2->SetMarkerColor(2);
        HFpositivedirection2->SetLineColor(0);
        gPad->SetGridy();
        gPad->SetGridx();
        //	   gPad->SetLogy();
        HFpositivedirection2->Draw("Error");
        kcountHFpositivedirection2++;
        if (kcountHFpositivedirection2 > 24)
          break;  // 4x6 = 24
      }           // for jphi
    }             //if
  }               //for jeta
  /////////////////
  c1->Update();
  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 59   HF:: jeta = 0,1,2, 3            18,19,20,21       jphi =12,13,14,15,16,17
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(4, 6);
  int kcountHFpositivedirection3 = 1;
  // j - etaphi index:
  TH1F* h2CeffHFpositivedirection3 =
      new TH1F("h2CeffHFpositivedirection3", "", maxbins, 1., maxbins + 1.);  //  h2CeffHFpositivedirection3->Sumw2();
  for (int jphi = 12; jphi < 18; jphi++) {
    for (int jeta = 18; jeta < 22; jeta++) {
      //	       	     cout<<"HF 55 PLOTTING       jeta=     "<< jeta <<"   jphi   =     "<<jphi  <<"   kcountHFpositivedirection3   =     "<<kcountHFpositivedirection3  <<endl;
      TH1F* HFpositivedirection3 = (TH1F*)h2CeffHFpositivedirection3->Clone("twod1");
      for (int i = 0; i < nx; i++) {
        double ccc1 = alexhf[jeta][jphi][i];
        //	    if(ccc1 <= 0.)	    	    cout<<"59  HF:  ibin=  "<< i <<"      jphi= "<< jphi <<"      jeta= "<< jeta <<"      A= "<< ccc1 <<endl;

        if (ccc1 > 0.) {
          HFpositivedirection3->Fill(i, ccc1);
          HFpositivedirection3->SetBinError(i, 0.01);
        }
        //	       if(ccc1>0.) {HFpositivedirection3->AddBinContent(int(HFpositivedirection3->FindBin(i)), ccc1);HFpositivedirection3->SetBinError(i,0.);}

        //		 	       if( i == 0 ) cout<<"HF 55 PLOTTING  for LS=1      ccc1=     "<< ccc1 <<endl;
      }  // for i
      c1->cd(kcountHFpositivedirection3);
      HFpositivedirection3->SetMarkerStyle(20);
      HFpositivedirection3->SetMarkerSize(0.4);
      HFpositivedirection3->GetYaxis()->SetLabelSize(0.04);
      HFpositivedirection3->SetXTitle("HFpositivedirection3 \b");
      HFpositivedirection3->SetMarkerColor(2);
      HFpositivedirection3->SetLineColor(0);
      gPad->SetGridy();
      gPad->SetGridx();
      //	   gPad->SetLogy();
      if (kcountHFpositivedirection3 == 1)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 12 \b");
      if (kcountHFpositivedirection3 == 5)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 13 \b");
      if (kcountHFpositivedirection3 == 9)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 14 \b");
      if (kcountHFpositivedirection3 == 13)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 15 \b");
      if (kcountHFpositivedirection3 == 17)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 16 \b");
      if (kcountHFpositivedirection3 == 21)
        HFpositivedirection3->SetXTitle("HF jeta = 18; jphi = 17 \b");

      if (kcountHFpositivedirection3 == 2)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 12 \b");
      if (kcountHFpositivedirection3 == 6)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 13 \b");
      if (kcountHFpositivedirection3 == 10)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 14 \b");
      if (kcountHFpositivedirection3 == 14)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 15 \b");
      if (kcountHFpositivedirection3 == 18)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 16 \b");
      if (kcountHFpositivedirection3 == 22)
        HFpositivedirection3->SetXTitle("HF jeta = 19; jphi = 17 \b");

      if (kcountHFpositivedirection3 == 3)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 12 \b");
      if (kcountHFpositivedirection3 == 7)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 13 \b");
      if (kcountHFpositivedirection3 == 11)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 14 \b");
      if (kcountHFpositivedirection3 == 15)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 15 \b");
      if (kcountHFpositivedirection3 == 19)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 16 \b");
      if (kcountHFpositivedirection3 == 23)
        HFpositivedirection3->SetXTitle("HF jeta = 20; jphi = 17 \b");

      if (kcountHFpositivedirection3 == 4)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 12 \b");
      if (kcountHFpositivedirection3 == 8)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 13 \b");
      if (kcountHFpositivedirection3 == 12)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 14 \b");
      if (kcountHFpositivedirection3 == 16)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 15 \b");
      if (kcountHFpositivedirection3 == 20)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 16 \b");
      if (kcountHFpositivedirection3 == 24)
        HFpositivedirection3->SetXTitle("HF jeta = 21; jphi = 17 \b");

      HFpositivedirection3->Draw("Error");

      //int bin = HFpositivedirection3->FindBin(i);			//OPTION C This works fine.
      //HFpositivedirection3->AddBinContent(bin, ccc1);
      //HFpositivedirection3->AddBinContent(int(HFpositivedirection3->FindBin(i)), ccc1);

      //		     HFpositivedirection3->Draw("HIST");

      kcountHFpositivedirection3++;
      if (kcountHFpositivedirection3 > 24)
        break;  // 4x6 = 24
    }           // for jphi
  }             //for jeta
                /////////////////
  if (h2CeffHFpositivedirection3)
    delete h2CeffHFpositivedirection3;
  c1->Update();
  //////////////////////////////////////////////////////////////////////////////////// Gefz1

  ////////////////////////////////////////////////////////////////////////////////////
  //========================================================================================== 60   HF:: 2D  jeta = 0 - 21       jphi =0 - 17
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(1, 1);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(1);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;
  TH2F* Gefz42D = new TH2F("Gefz42D", "", 23, -11.5, 11.5, 18, 0., 18.);
  TH2F* Gefz42D0 = new TH2F("Gefz42D0", "", 23, -11.5, 11.5, 18, 0., 18.);
  //     TH2F* Gefz42D      = new TH2F("Gefz42D","",   22, -11., 11., 18, 0., 18. );
  //     TH2F* Gefz42D0     = new TH2F("Gefz42D0","",  22, -11., 11., 18, 0., 18. );
  //      TH2F* Gefz42D      = new TH2F("Gefz42D","",   24, -12., 12., 18, 0., 18. );
  //     TH2F* Gefz42D0     = new TH2F("Gefz42D0","",  24, -12., 12., 18, 0., 18. );
  TH2F* Gefz42DF = (TH2F*)Gefz42D0->Clone("Gefz42DF");
  for (int jphi = 0; jphi < 18; jphi++) {
    for (int jeta = 0; jeta < 22; jeta++) {
      for (int i = 0; i < nx; i++) {
        double ccc1 = alexhf[jeta][jphi][i];
        int neweta = jeta - 11 - 0.5;
        if (jeta >= 11)
          neweta = jeta - 11 + 1.5;
        if (ccc1 > 0.) {
          Gefz42D->Fill(neweta, jphi, ccc1);
          Gefz42D0->Fill(neweta, jphi, 1.);
        }
      }
    }
  }
  Gefz42DF->Divide(Gefz42D, Gefz42D0, 1, 1, "B");  // average A
  //    Gefz1->Sumw2();
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gefz42DF->SetMarkerStyle(20);
  Gefz42DF->SetMarkerSize(0.4);
  Gefz42DF->GetZaxis()->SetLabelSize(0.08);
  Gefz42DF->SetXTitle("<A>_RBX         #eta  \b");
  Gefz42DF->SetYTitle("      #phi \b");
  Gefz42DF->SetZTitle("<A>_RBX  - All \b");
  Gefz42DF->SetMarkerColor(2);
  Gefz42DF->SetLineColor(0);  //      Gefz42DF->SetMaximum(1.000);  //      Gefz42DF->SetMinimum(1.0);
  Gefz42DF->Draw("COLZ");

  c1->Update();

  //======================================================================
  //========================================================================================== 61   HF:: 2D  jeta = 0 - 21       jphi =0 - 17
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(1, 1);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(1);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;

  TH1F* Gefz41D = new TH1F("Gefz41D", "", 18, 0., 18.);
  TH1F* Gefz41D0 = new TH1F("Gefz41D0", "", 18, 0., 18.);
  TH1F* Gefz41DF = (TH1F*)Gefz41D0->Clone("Gefz41DF");
  for (int jphi = 0; jphi < 18; jphi++) {
    for (int jeta = 0; jeta < 22; jeta++) {
      for (int i = 0; i < nx; i++) {
        double ccc1 = alexhf[jeta][jphi][i];
        if (ccc1 > 0.) {
          Gefz41D->Fill(jphi, ccc1);
          Gefz41D0->Fill(jphi, 1.);
        }
      }
    }
  }
  //     Gefz41D->Sumw2();Gefz41D0->Sumw2();
  Gefz41DF->Divide(Gefz41D, Gefz41D0, 1, 1, "B");  // average A
                                                   //     Gefz41DF->Sumw2();
  for (int jphi = 1; jphi < 19; jphi++) {
    Gefz41DF->SetBinError(jphi, 0.01);
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gefz41DF->SetMarkerStyle(20);
  Gefz41DF->SetMarkerSize(1.4);
  Gefz41DF->GetZaxis()->SetLabelSize(0.08);
  Gefz41DF->SetXTitle("#phi  \b");
  Gefz41DF->SetYTitle("  <A> \b");
  Gefz41DF->SetZTitle("<A>_PHI  - All \b");
  Gefz41DF->SetMarkerColor(4);
  Gefz41DF->SetLineColor(4);
  Gefz41DF->SetMinimum(0.8);  //      Gefz41DF->SetMaximum(1.000);
  Gefz41DF->Draw("Error");

  c1->Update();

  //======================================================================
  //======================================================================
  //========================================================================================== 62   HF:: 2D  jeta = 0 - 21       jphi =0 - 17
  //======================================================================
  //======================================================================
  //======================================================================
  //======================================================================
  c1->Clear();
  /////////////////
  c1->Divide(1, 1);
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  c1->cd(1);
  // int ietaphi = 0; ietaphi = ((k2+1)-1)*nphi + (k3+1) ;  k2=0-neta-1; k3=0-nphi-1; neta=18; nphi=22;
  // jeta = 0 - 21       jphi = 0 - 17
  TH1F* Gjeta41D = new TH1F("Gjeta41D", "", 23, -11.5, 11.5);
  TH1F* Gjeta41D0 = new TH1F("Gjeta41D0", "", 23, -11.5, 11.5);
  TH1F* Gjeta41DF = (TH1F*)Gjeta41D0->Clone("Gjeta41DF");

  for (int jeta = 0; jeta < 22; jeta++) {
    for (int jphi = 0; jphi < 18; jphi++) {
      for (int i = 0; i < nx; i++) {
        double ccc1 = alexhf[jeta][jphi][i];
        int neweta = jeta - 11 - 0.5;
        if (jeta >= 11)
          neweta = jeta - 11 + 1.5;
        if (ccc1 > 0.) {
          Gjeta41D->Fill(neweta, ccc1);
          Gjeta41D0->Fill(neweta, 1.);
          //	       if( i == 0 ) cout<<"62  HF:  ibin=  "<< i <<"      jphi= "<< jphi <<"      jeta= "<< jeta <<"      A= "<< ccc1 <<endl;
        }
      }
    }
  }
  //     Gjeta41D->Sumw2();Gjeta41D0->Sumw2();
  Gjeta41DF->Divide(Gjeta41D, Gjeta41D0, 1, 1, "B");  // average A
                                                      //     Gjeta41DF->Sumw2();
  for (int jeta = 1; jeta < 24; jeta++) {
    Gjeta41DF->SetBinError(jeta, 0.01);
  }
  gPad->SetGridy();
  gPad->SetGridx();  //      gPad->SetLogz();
  Gjeta41DF->SetMarkerStyle(20);
  Gjeta41DF->SetMarkerSize(1.4);
  Gjeta41DF->GetZaxis()->SetLabelSize(0.08);
  Gjeta41DF->SetXTitle("#eta  \b");
  Gjeta41DF->SetYTitle("  <A> \b");
  Gjeta41DF->SetZTitle("<A>_ETA  - All \b");
  Gjeta41DF->SetMarkerColor(4);
  Gjeta41DF->SetLineColor(4);
  Gjeta41DF->SetMinimum(0.8);  //      Gjeta41DF->SetMaximum(1.000);
  Gjeta41DF->Draw("Error");

  c1->Update();

  //======================================================================
  //======================================================================
  //======================================================================
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////
  //======================================================================
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
