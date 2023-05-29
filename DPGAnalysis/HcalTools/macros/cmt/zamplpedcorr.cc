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
  TPostScript psfile("zamplpedcorr.ps", 111);

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
  TH1F *twod1 = (TH1F *)dir->FindObjectAny("h_pedvsampl_HB");
  TH1F *twod0 = (TH1F *)dir->FindObjectAny("h_pedvsampl0_HB");
  twod1->Sumw2();
  twod0->Sumw2();
  //      if(twod0->IsA()->InheritsFrom("TH1F")){
  TH1F *Cefz225 = (TH1F *)twod1->Clone("Cefz225");
  Cefz225->Divide(twod1, twod0, 1, 1, "B");
  Cefz225->Sumw2();
  //      }
  gPad->SetGridy();
  gPad->SetGridx();
  Cefz225->SetMarkerStyle(20);
  Cefz225->SetMarkerSize(0.4);
  Cefz225->SetXTitle("Pedestals \b");
  Cefz225->SetYTitle("<A> \b");
  Cefz225->SetMarkerColor(2);
  Cefz225->SetLineColor(2);
  //    Cefz225->SetMaximum(1.000);
  //    Cefz225->SetMinimum(0.0001);
  Cefz225->Draw("COLZ");

  c1->cd(2);
  TH1F *twod61 = (TH1F *)dir->FindObjectAny("h_pedwvsampl_HB");
  TH1F *twod60 = (TH1F *)dir->FindObjectAny("h_pedwvsampl0_HB");
  twod61->Sumw2();
  twod60->Sumw2();
  //      if(twod60->IsA()->InheritsFrom("TH1F")){
  TH1F *Cefz226 = (TH1F *)twod61->Clone("Cefz226");
  Cefz226->Divide(twod61, twod60, 1, 1, "B");
  Cefz226->Sumw2();
  //      }
  gPad->SetGridy();
  gPad->SetGridx();
  Cefz226->SetMarkerStyle(20);
  Cefz226->SetMarkerSize(0.4);
  Cefz226->SetXTitle("Width_Pedestals \b");
  Cefz226->SetYTitle("<A> \b");
  Cefz226->SetMarkerColor(2);
  Cefz226->SetLineColor(2);
  //    Cefz226->SetMaximum(1.000);
  //    Cefz226->SetMinimum(0.0001);
  Cefz226->Draw("COLZ");

  c1->cd(3);
  TH1F *twod71 = (TH1F *)dir->FindObjectAny("h_amplvsped_HB");
  TH1F *twod70 = (TH1F *)dir->FindObjectAny("h_amplvsped0_HB");
  twod71->Sumw2();
  twod70->Sumw2();
  //      if(twod70->IsA()->InheritsFrom("TH1F")){
  TH1F *Cefz227 = (TH1F *)twod71->Clone("Cefz227");
  Cefz227->Divide(twod71, twod70, 1, 1, "B");
  Cefz227->Sumw2();
  //      }
  gPad->SetGridy();
  gPad->SetGridx();
  Cefz227->SetMarkerStyle(20);
  Cefz227->SetMarkerSize(0.4);
  Cefz227->SetXTitle("Amplitude \b");
  Cefz227->SetYTitle("<Pedestals> \b");
  Cefz227->SetMarkerColor(2);
  Cefz227->SetLineColor(2);
  //    Cefz227->SetMaximum(1.000);
  //    Cefz227->SetMinimum(0.0001);
  Cefz227->Draw("COLZ");

  c1->cd(4);
  TH1F *twod81 = (TH1F *)dir->FindObjectAny("h_amplvspedw_HB");
  TH1F *twod80 = (TH1F *)dir->FindObjectAny("h_amplvsped0_HB");
  twod81->Sumw2();
  twod80->Sumw2();
  //      if(twod80->IsA()->InheritsFrom("TH1F")){
  TH1F *Cefz228 = (TH1F *)twod81->Clone("Cefz228");
  Cefz228->Divide(twod81, twod80, 1, 1, "B");
  Cefz228->Sumw2();
  //      }
  gPad->SetGridy();
  gPad->SetGridx();
  Cefz228->SetMarkerStyle(20);
  Cefz228->SetMarkerSize(0.4);
  Cefz228->SetXTitle("Amplitude \b");
  Cefz228->SetYTitle("<Width_Pedestals> \b");
  Cefz228->SetMarkerColor(2);
  Cefz228->SetLineColor(2);
  //    Cefz228->SetMaximum(1.000);
  //    Cefz228->SetMinimum(0.0001);
  Cefz228->Draw("COLZ");

  c1->Update();

  //========================================================================================= 2
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH2F *two11 = (TH2F *)dir->FindObjectAny("h2_pedvsampl_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two11->SetMarkerStyle(20);
  two11->SetMarkerSize(0.4);
  two11->SetXTitle("Pedestals HB \b");
  two11->SetYTitle("Amplitude HB\b");
  two11->SetMarkerColor(2);
  two11->SetLineColor(2);
  //         gPad->SetLogy();
  two11->Draw("COLZ");

  c1->cd(2);
  TH2F *two12 = (TH2F *)dir->FindObjectAny("h2_pedwvsampl_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two12->SetMarkerStyle(20);
  two12->SetMarkerSize(0.4);
  two12->SetXTitle("Width_Pedestals HB \b");
  two12->SetYTitle("Amplitude HB\b");
  two12->SetMarkerColor(2);
  two12->SetLineColor(2);
  //   gPad->SetLogy();
  two12->Draw("COLZ");

  c1->cd(3);
  TH2F *two22 = (TH2F *)dir->FindObjectAny("h2_amplvsped_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two22->SetMarkerStyle(20);
  two22->SetMarkerSize(0.4);
  two22->SetYTitle("Pedestals HB \b");
  two22->SetXTitle("Amplitude HB\b");
  two22->SetMarkerColor(2);
  two22->SetLineColor(2);
  two22->Draw("COLZ");

  c1->cd(4);
  TH2F *two23 = (TH2F *)dir->FindObjectAny("h2_amplvspedw_HB");
  gPad->SetGridy();
  gPad->SetGridx();
  two23->SetMarkerStyle(20);
  two23->SetMarkerSize(0.4);
  two23->SetYTitle("Width_Pedestals HB \b");
  two23->SetXTitle("Amplitude HB\b");
  two23->SetMarkerColor(2);
  two23->SetLineColor(2);
  two23->Draw("COLZ");

  c1->Update();

  //========================================================================================= 3
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);

  c1->cd(1);
  TH1F *aaaaaa1 = (TH1F *)dir->FindObjectAny("h_sumADCAmplLS1copy2");
  gPad->SetLogy();
  aaaaaa1->SetMarkerStyle(20);
  aaaaaa1->SetMarkerSize(0.8);
  aaaaaa1->GetYaxis()->SetLabelSize(0.04);
  aaaaaa1->SetXTitle("<A>(ev.in LS) in LSs & channels - HB depth1\b");
  aaaaaa1->SetMarkerColor(4);
  aaaaaa1->SetLineColor(0);
  aaaaaa1->Draw("Error");

  c1->cd(2);
  TH1F *aaaaaa2 = (TH1F *)dir->FindObjectAny("h_sumADCAmplLS1copy3");
  gPad->SetLogy();
  aaaaaa2->SetMarkerStyle(20);
  aaaaaa2->SetMarkerSize(0.8);
  aaaaaa2->GetYaxis()->SetLabelSize(0.04);
  aaaaaa2->SetXTitle("<A>(ev.in LS) in LSs & channels - HB depth1\b");
  aaaaaa2->SetMarkerColor(4);
  aaaaaa2->SetLineColor(0);
  aaaaaa2->Draw("Error");

  c1->cd(3);
  TH1F *aaaaaa3 = (TH1F *)dir->FindObjectAny("h_sumADCAmplLS1copy4");
  gPad->SetLogy();
  aaaaaa3->SetMarkerStyle(20);
  aaaaaa3->SetMarkerSize(0.8);
  aaaaaa3->GetYaxis()->SetLabelSize(0.04);
  aaaaaa3->SetXTitle("<A>(ev.in LS) in LSs & channels - HB depth1\b");
  aaaaaa3->SetMarkerColor(4);
  aaaaaa3->SetLineColor(0);
  aaaaaa3->Draw("Error");
  c1->cd(4);
  TH1F *aaaaaa4 = (TH1F *)dir->FindObjectAny("h_sumADCAmplLS1copy5");
  gPad->SetLogy();
  aaaaaa4->SetMarkerStyle(20);
  aaaaaa4->SetMarkerSize(0.8);
  aaaaaa4->GetYaxis()->SetLabelSize(0.04);
  aaaaaa4->SetXTitle("<A>(ev.in LS) in LSs & channels - HB depth1\b");
  aaaaaa4->SetMarkerColor(4);
  aaaaaa4->SetLineColor(0);
  aaaaaa4->Draw("Error");

  c1->Update();

  //=============================================================================================== 4
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);
  /*
      c1->cd(1);
      TH1F *twodhe1= (TH1F*)dir->FindObjectAny("h_pedvsampl_HE");
      TH1F *twodhe0= (TH1F*)dir->FindObjectAny("h_pedvsampl0_HE");
      twodhe1->Sumw2();
      twodhe0->Sumw2();
//      if(twodhe0->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhe225= (TH1F*)twodhe1->Clone("Cefzhe225");
	Cefzhe225->Divide(twodhe1,twodhe0, 1, 1, "B");
	Cefzhe225->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhe225->SetMarkerStyle(20);
      Cefzhe225->SetMarkerSize(0.4);
          Cefzhe225->SetXTitle("Pedestals \b");
          Cefzhe225->SetYTitle("<A> \b");
      Cefzhe225->SetMarkerColor(2);
      Cefzhe225->SetLineColor(2);
      //    Cefzhe225->SetMaximum(1.000);
      //    Cefzhe225->SetMinimum(0.0001);
      Cefzhe225->Draw("COLZ");
      
      c1->cd(2);
      TH1F *twodhe61= (TH1F*)dir->FindObjectAny("h_pedwvsampl_HE");
      TH1F *twodhe60= (TH1F*)dir->FindObjectAny("h_pedwvsampl0_HE");
      twodhe61->Sumw2();
      twodhe60->Sumw2();
//      if(twodhe60->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhe226= (TH1F*)twodhe61->Clone("Cefzhe226");
	Cefzhe226->Divide(twodhe61,twodhe60, 1, 1, "B");
	Cefzhe226->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhe226->SetMarkerStyle(20);
      Cefzhe226->SetMarkerSize(0.4);
          Cefzhe226->SetXTitle("Width_Pedestals \b");
          Cefzhe226->SetYTitle("<A> \b");
      Cefzhe226->SetMarkerColor(2);
      Cefzhe226->SetLineColor(2);
      //    Cefzhe226->SetMaximum(1.000);
      //    Cefzhe226->SetMinimum(0.0001);
      Cefzhe226->Draw("COLZ");
      
      c1->cd(3);
      TH1F *twodhe71= (TH1F*)dir->FindObjectAny("h_amplvsped_HE");
      TH1F *twodhe70= (TH1F*)dir->FindObjectAny("h_amplvsped0_HE");
      twodhe71->Sumw2();
      twodhe70->Sumw2();
//      if(twodhe70->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhe227= (TH1F*)twodhe71->Clone("Cefzhe227");
	Cefzhe227->Divide(twodhe71,twodhe70, 1, 1, "B");
	Cefzhe227->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhe227->SetMarkerStyle(20);
      Cefzhe227->SetMarkerSize(0.4);
          Cefzhe227->SetXTitle("Amplitude \b");
          Cefzhe227->SetYTitle("<Pedestals> \b");
      Cefzhe227->SetMarkerColor(2);
      Cefzhe227->SetLineColor(2);
      //    Cefzhe227->SetMaximum(1.000);
      //    Cefzhe227->SetMinimum(0.0001);
      Cefzhe227->Draw("COLZ");
      
      c1->cd(4);
      TH1F *twodhe81= (TH1F*)dir->FindObjectAny("h_amplvspedw_HE");
      TH1F *twodhe80= (TH1F*)dir->FindObjectAny("h_amplvsped0_HE");
      twodhe81->Sumw2();
      twodhe80->Sumw2();
//      if(twodhe80->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhe228= (TH1F*)twodhe81->Clone("Cefzhe228");
	Cefzhe228->Divide(twodhe81,twodhe80, 1, 1, "B");
	Cefzhe228->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhe228->SetMarkerStyle(20);
      Cefzhe228->SetMarkerSize(0.4);
          Cefzhe228->SetXTitle("Amplitude \b");
          Cefzhe228->SetYTitle("<Width_Pedestals> \b");
      Cefzhe228->SetMarkerColor(2);
      Cefzhe228->SetLineColor(2);
      //    Cefzhe228->SetMaximum(1.000);
      //    Cefzhe228->SetMinimum(0.0001);
      Cefzhe228->Draw("COLZ");
      
 */
  c1->Update();

  //=============================================================================================== 5
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);
  /*
      c1->cd(1);
      TH1F *twodhf1= (TH1F*)dir->FindObjectAny("h_pedvsampl_HF");
      TH1F *twodhf0= (TH1F*)dir->FindObjectAny("h_pedvsampl0_HF");
      twodhf1->Sumw2();
      twodhf0->Sumw2();
//      if(twodhf0->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhf225= (TH1F*)twodhf1->Clone("Cefzhf225");
	Cefzhf225->Divide(twodhf1,twodhf0, 1, 1, "B");
	Cefzhf225->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhf225->SetMarkerStyle(20);
      Cefzhf225->SetMarkerSize(0.4);
          Cefzhf225->SetXTitle("Pedestals \b");
          Cefzhf225->SetYTitle("<A> \b");
      Cefzhf225->SetMarkerColor(2);
      Cefzhf225->SetLineColor(2);
      //    Cefzhf225->SetMaximum(1.000);
      //    Cefzhf225->SetMinimum(0.0001);
      Cefzhf225->Draw("COLZ");
      
      c1->cd(2);
      TH1F *twodhf61= (TH1F*)dir->FindObjectAny("h_pedwvsampl_HF");
      TH1F *twodhf60= (TH1F*)dir->FindObjectAny("h_pedwvsampl0_HF");
      twodhf61->Sumw2();
      twodhf60->Sumw2();
//      if(twodhf60->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhf226= (TH1F*)twodhf61->Clone("Cefzhf226");
	Cefzhf226->Divide(twodhf61,twodhf60, 1, 1, "B");
	Cefzhf226->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhf226->SetMarkerStyle(20);
      Cefzhf226->SetMarkerSize(0.4);
          Cefzhf226->SetXTitle("Width_Pedestals \b");
          Cefzhf226->SetYTitle("<A> \b");
      Cefzhf226->SetMarkerColor(2);
      Cefzhf226->SetLineColor(2);
      //    Cefzhf226->SetMaximum(1.000);
      //    Cefzhf226->SetMinimum(0.0001);
      Cefzhf226->Draw("COLZ");
      
      c1->cd(3);
      TH1F *twodhf71= (TH1F*)dir->FindObjectAny("h_amplvsped_HF");
      TH1F *twodhf70= (TH1F*)dir->FindObjectAny("h_amplvsped0_HF");
      twodhf71->Sumw2();
      twodhf70->Sumw2();
//      if(twodhf70->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhf227= (TH1F*)twodhf71->Clone("Cefzhf227");
	Cefzhf227->Divide(twodhf71,twodhf70, 1, 1, "B");
	Cefzhf227->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhf227->SetMarkerStyle(20);
      Cefzhf227->SetMarkerSize(0.4);
          Cefzhf227->SetXTitle("Amplitude \b");
          Cefzhf227->SetYTitle("<Pedestals> \b");
      Cefzhf227->SetMarkerColor(2);
      Cefzhf227->SetLineColor(2);
      //    Cefzhf227->SetMaximum(1.000);
      //    Cefzhf227->SetMinimum(0.0001);
      Cefzhf227->Draw("COLZ");
      
      c1->cd(4);
      TH1F *twodhf81= (TH1F*)dir->FindObjectAny("h_amplvspedw_HF");
      TH1F *twodhf80= (TH1F*)dir->FindObjectAny("h_amplvsped0_HF");
      twodhf81->Sumw2();
      twodhf80->Sumw2();
//      if(twodhf80->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzhf228= (TH1F*)twodhf81->Clone("Cefzhf228");
	Cefzhf228->Divide(twodhf81,twodhf80, 1, 1, "B");
	Cefzhf228->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzhf228->SetMarkerStyle(20);
      Cefzhf228->SetMarkerSize(0.4);
          Cefzhf228->SetXTitle("Amplitude \b");
          Cefzhf228->SetYTitle("<Width_Pedestals> \b");
      Cefzhf228->SetMarkerColor(2);
      Cefzhf228->SetLineColor(2);
      //    Cefzhf228->SetMaximum(1.000);
      //    Cefzhf228->SetMinimum(0.0001);
      Cefzhf228->Draw("COLZ");
      
 */

  c1->Update();

  //=============================================================================================== 6
  //======================================================================
  //======================================================================
  //================
  //======================================================================
  c1->Clear();
  c1->Divide(2, 2);
  /*
      c1->cd(1);
      TH1F *twodho1= (TH1F*)dir->FindObjectAny("h_pedvsampl_HO");
      TH1F *twodho0= (TH1F*)dir->FindObjectAny("h_pedvsampl0_HO");
      twodho1->Sumw2();
      twodho0->Sumw2();
//      if(twodho0->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzho225= (TH1F*)twodho1->Clone("Cefzho225");
	Cefzho225->Divide(twodho1,twodho0, 1, 1, "B");
	Cefzho225->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzho225->SetMarkerStyle(20);
      Cefzho225->SetMarkerSize(0.4);
          Cefzho225->SetXTitle("Pedestals \b");
          Cefzho225->SetYTitle("<A> \b");
      Cefzho225->SetMarkerColor(2);
      Cefzho225->SetLineColor(2);
      //    Cefzho225->SetMaximum(1.000);
      //    Cefzho225->SetMinimum(0.0001);
      Cefzho225->Draw("COLZ");
      
      c1->cd(2);
      TH1F *twodho61= (TH1F*)dir->FindObjectAny("h_pedwvsampl_HO");
      TH1F *twodho60= (TH1F*)dir->FindObjectAny("h_pedwvsampl0_HO");
      twodho61->Sumw2();
      twodho60->Sumw2();
//      if(twodho60->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzho226= (TH1F*)twodho61->Clone("Cefzho226");
	Cefzho226->Divide(twodho61,twodho60, 1, 1, "B");
	Cefzho226->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzho226->SetMarkerStyle(20);
      Cefzho226->SetMarkerSize(0.4);
          Cefzho226->SetXTitle("Width_Pedestals \b");
          Cefzho226->SetYTitle("<A> \b");
      Cefzho226->SetMarkerColor(2);
      Cefzho226->SetLineColor(2);
      //    Cefzho226->SetMaximum(1.000);
      //    Cefzho226->SetMinimum(0.0001);
      Cefzho226->Draw("COLZ");
      
      c1->cd(3);
      TH1F *twodho71= (TH1F*)dir->FindObjectAny("h_amplvsped_HO");
      TH1F *twodho70= (TH1F*)dir->FindObjectAny("h_amplvsped0_HO");
      twodho71->Sumw2();
      twodho70->Sumw2();
//      if(twodho70->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzho227= (TH1F*)twodho71->Clone("Cefzho227");
	Cefzho227->Divide(twodho71,twodho70, 1, 1, "B");
	Cefzho227->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzho227->SetMarkerStyle(20);
      Cefzho227->SetMarkerSize(0.4);
          Cefzho227->SetXTitle("Amplitude \b");
          Cefzho227->SetYTitle("<Pedestals> \b");
      Cefzho227->SetMarkerColor(2);
      Cefzho227->SetLineColor(2);
      //    Cefzho227->SetMaximum(1.000);
      //    Cefzho227->SetMinimum(0.0001);
      Cefzho227->Draw("COLZ");
      
      c1->cd(4);
      TH1F *twodho81= (TH1F*)dir->FindObjectAny("h_amplvspedw_HO");
      TH1F *twodho80= (TH1F*)dir->FindObjectAny("h_amplvsped0_HO");
      twodho81->Sumw2();
      twodho80->Sumw2();
//      if(twodho80->IsA()->InheritsFrom("TH1F")){
	TH1F* Cefzho228= (TH1F*)twodho81->Clone("Cefzho228");
	Cefzho228->Divide(twodho81,twodho80, 1, 1, "B");
	Cefzho228->Sumw2();
//      }
      gPad->SetGridy();
      gPad->SetGridx();
      Cefzho228->SetMarkerStyle(20);
      Cefzho228->SetMarkerSize(0.4);
          Cefzho228->SetXTitle("Amplitude \b");
          Cefzho228->SetYTitle("<Width_Pedestals> \b");
      Cefzho228->SetMarkerColor(2);
      Cefzho228->SetLineColor(2);
      //    Cefzho228->SetMaximum(1.000);
      //    Cefzho228->SetMinimum(0.0001);
      Cefzho228->Draw("COLZ");
      
 */

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
