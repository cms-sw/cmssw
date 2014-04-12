#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <list>

#include <math.h>
#include <vector>

#include "Rtypes.h"
#include "TROOT.h"
#include "TRint.h"
#include "TObject.h"
#include "TFile.h"
// #include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TRefArray.h"
#include "TStyle.h"
#include "TGraph.h"
#include "TLegend.h"


// ttbar+jets at 7TeV (sample 5226, cern/eos)
// xqcut=20
//
const int NQCutTTbar = 23 ;
double QCutTTbar[23] = { 20., 25., 30., 
31., 32., 33., 34., 35., 36., 37., 38., 39., 
40.,  // "golden" qcut for Py6 
41., 42., 43., 44.,  45., 46., 47., 48., 49., 50. };
//
// Py6+Z2
//
double TTbarEffPy6[23] = { 24.1, 27.1, 28.6,
28.5, 28.6, 29.0, 29.3, 29.2, 29.5, 29.3, 29.5, 29.7, 
29.9, // total matching eff. at "golden" qcut=40
29.7, 30.0, 30.0, 30.1, 30.0, 30.0, 30.3, 30.2, 30.2, 30.3 };
//
// Py8+4C
//
double TTbarEffPy8[23] = { 19.4, 22.3, 24.1,
24.5, 24.7, 24.8, 24.9, 25.3, 25.7, 25.8, 25.9, 26.0, 
26.2, // total matching eff. at Py6's "golden" qcut=40
26.2, 
26.5, // total matching eff. at Py8's proposed qcut=42
26.6, 26.5, 26.7, 26.7, 26.9, 26.8, 27.1, 26.9 };  

// Wbb at 8TeV (sample 6673, cern/eos)
// xqcut=10
//
const int NQCutWbb = 25 ;
double QCutWbb[25] = { 10., 11., 12., 13., 14.,
15., // "golden" qcut for Py6
16., 17., 18., 19., 20, 21., 22., 23., 24., 25., 26., 27., 28., 29., 30.,
35., 40., 45., 50. };
//
// Py6+Z2star
//
double WbbEffPy6[25] = { 39.1, 40.4, 41.2, 41.5, 41.8,
42.3, // total matching eff. at "golden" qcut=15
42.4, 42.1, 42.0, 41.6, 41.3, 41.0, 40.6, 40.3, 40.0, 39.3, 39.3, 38.8, 38.5, 38.2, 37.5,
35.8, 33.9, 32.5, 31.3 };
//
// Py8+4C
//
double WbbEffPy8[25] = {  30.3, 31.7, 32.7, 33.7, 34.5,
34.8, // total matching eff. at Py6's "golden qcut=15
35.2, // total matching eff. at Py8's proposed qcut=16
35.7, 35.7, 35.8, 35.9, 35.8, 35.9, 35.7, 35.7, 35.4, 35.2, 35.2, 34.9, 34.5, 34.6,
33.3, 32.3, 31.1, 30.1 };

// W+jets at 7TeV (sample 2924, cern/eos)
// xqcut=10
//
const int NQCutW = 25;
double QCutW[25] = { 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
20., // "golden" qcut for Py6
21., 22., 23., 24., 25., 
26., // proposed for Py8 qcut=26
27., 28., 29., 30.,
35., 40., 45., 50. }; 
//
// Py6+Z2
//
double WEffPy6[25] = { 41.4, 43.7, 45.3, 46.3, 47.4, 48.0, 48.7, 48.8, 49.1, 49.5, 
49.6, // total matching eff. at Py6's "golden" qcut=20
49.8, 49.6, 49.8, 50.0, 49.9, 49.9, 49.9, 49.9, 50.0, 50.0,
49.9, 49.8, 49.4, 49.2 };
//
// Py8+4C
//
double WEffPy8[25] = { 33.7, 36.3, 37.8, 39.4, 40.6, 41.5, 42.2, 43.0, 43.3, 43.9,
44.1, // total matching eff. at Py6's "golden" qcut=20
44.5, 44.9, 45.1, 45.5, 45.4,
45.8, // total matching eff. at the proposed for Py8 qcut=26
45.6, 45.8, 45.9, 45.9,
46.1, 46.4, 46.4, 46.2 };

// Zinclusive at 7TeV (sample 2925, cern/eos)
// xqcut=10
//
const int NQCutZ = 25;
double QCutZ[25] = { 10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
20., // "golden" qcut for Py6
21., 22., 23., 24., 
25., // proposed for Py8 qcut=25
26., 27., 28., 29., 30.,
35., 40., 45., 50. }; 
//
// Py6+Z2
//
double ZEffPy6[25] = { 37.8, 39.9, 41.6, 42.4, 43.3, 43.7, 44.4, 44.5, 44.9, 45.0,
45.0, // total matching eff. for Py6's "golden" qcut=20
45.1, 45.3, 45.4, 45.3, 45.5, 45.4, 45.4, 45.4, 45.3, 45.5,
45.1, 44.9, 44.8, 44.7 };
//
// Py8+4C
//
double ZEffPy8[25] = { 30.9, 32.9, 34.9, 36.0, 37.1, 37.9, 38.6, 39.2, 39.7, 40.1,
40.1, // total matching eff. at Py6's "golden" qcut=20
40.7, 41.0, 41.0, 41.3, 
41.4, // total matching eff. at the proposed for Py8 qcut=25
41.5, 41.7, 41.7, 41.7, 41.8,  
42.1, 42.1, 42.1, 42.1 };


//--------------------------------------------


void plotTTbarEff()
{

   TCanvas* myc = new TCanvas("myc","", 800, 600);
      
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);

   TGraph* grTotalEffPy6 = new TGraph( NQCutTTbar, QCutTTbar, TTbarEffPy6 );
   grTotalEffPy6->SetTitle("ttbar+jets at 7TeV (MG+Py6/Py8, xqcut=20)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutTTbar, QCutTTbar, TTbarEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6/Z2", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8/4C", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
      
   myc->cd();

   return;

}

void plotWbbEff()
{

   TCanvas* myc = new TCanvas("myc","", 800, 600);
      
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   TGraph* grTotalEffPy6 = new TGraph( NQCutWbb, QCutWbb, WbbEffPy6 );
   grTotalEffPy6->SetTitle("Wbb at 8TeV (MG+Py6/Py8, xqcut=10)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutWbb, QCutWbb, WbbEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6/Z2star", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8/4C", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
      
   myc->cd();


   return;

}

void plotWjetsEff()
{

   TCanvas* myc = new TCanvas("myc","", 800, 600);
      
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   TGraph* grTotalEffPy6 = new TGraph( NQCutW, QCutW, WEffPy6 );
   grTotalEffPy6->SetTitle("W+jets at 7TeV (MG+Py6/Py8, xqcut=10)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutW, QCutW, WEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6/Z2", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8/4C", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
      
   myc->cd();


   return;

}

void plotZinclEff()
{

   TCanvas* myc = new TCanvas("myc","", 800, 600);
      
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   TGraph* grTotalEffPy6 = new TGraph( NQCutZ, QCutZ, ZEffPy6 );
   grTotalEffPy6->SetTitle("Z at 7TeV (MG+Py6/Py8, xqcut=10)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutZ, QCutZ, ZEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6/Z2", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8/4C", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
      
   myc->cd();


   return;

}
