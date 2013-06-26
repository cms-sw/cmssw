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

// Z-invisible
//
// These are for 5K stat, Py6 uses DT6 tune
//
const int NQCutZvv = 7;
double QCutZvv[7] = { 10., 15., 20, 25., 30., 35., 40. };

//                                        D=20
/* 5K stat, DT6 tune
double ZvvP1EffPy6[7] = {    11.0, 17.6,  18.7,  35.2, 33.0, 50.5, 52.7 };
double ZvvP2EffPy6[7] = {     5.9, 11.4,  15.1,  21.9, 23.5, 32.1, 33.7 };
double ZvvP3EffPy6[7] = {     4.0,  7.9,   9.8,  12.1, 11.5, 10.5, 11.6 };
double ZvvP4EffPy6[7] = {    14.0, 11.9,   9.4,  7.5,   5.0,  3.5,  3.1 };
double ZvvTotalEffPy6[7] = {  9.8, 10.7,  10.3,  11.1,  9.7, 10.0, 10.1 };
*/

/* 25K stat, Z2 tune */
//                                       D=20
double ZvvP1EffPy6[7] = {    12.9, 23.2, 27.3, 34.6, 41.7, 51.1, 55.1 };
double ZvvP2EffPy6[7] = {     8.3, 15.9, 23.5, 27.7, 33.3, 36.7, 39.1 };
double ZvvP3EffPy6[7] = {     6.3, 10.8, 13.4, 13.9, 13.0, 13.1, 12.4 };
double ZvvP4EffPy6[7] = {    18.3, 15.1, 11.4,  8.2,  6.1,  4.6,  3.0 };
double ZvvTotalEffPy6[7] = { 13.3, 13.9, 13.7, 12.7, 12.3, 11.7, 11.0 };  


//                                       D=20
double ZvvP1EffPy8[7] = {    6.6, 15.4,  24.2,  31.9, 34.1, 40.7, 58.2 };
double ZvvP2EffPy8[7] = {    4.3, 11.2,  19.6,  20.9, 27.4, 34.2, 35.6 };
double ZvvP3EffPy8[7] = {    3.8,  8.1,  11.2,  11.8, 12.1, 10.9, 11.6 };
double ZvvP4EffPy8[7] = {   13.7, 10.8,  10.2,   6.8,  6.0,  4.6,  3.2 };
double ZvvTotalEffPy8[7] = { 9.4, 10.0,  11.8,  10.4, 10.9, 10.6,  10.5 };


// Z -> b bar
//
const int NQCutZbb = 8;
double QCutZbb[8] = { 10., 13., 15., 20., 25., 30., 35., 40. };

/* 5K stat, DT5 tune
//                                  D=13
double ZbbP1EffPy6[8] = {    45.8,  53.2,  58.5, 67.3, 75.5, 79.5, 82.8, 85.7 };
double ZbbP2EffPy6[8] = {    26.1,  27.6,  28.1, 27.0, 23.1, 19.2, 16.8, 14.0 };
double ZbbP3EffPy6[8] = {    39.1,  33.7,  27.7, 19.7, 14.5, 11.0, 7.7, 7.2 };
double ZbbTotalEffPy6[8] = { 36.8,  38.3,  38.4, 38.6, 38.5, 37.4, 36.7, 36.6 };
*/

/* 25K stat, Z2 tune */
//                                  D=13
double ZbbP1EffPy6[8] = {    35.2,  44.8,  51.4, 61.8, 70.5, 76.6, 81.4, 84.8 };
double ZbbP2EffPy6[8] = {    19.0,  23.4,  24.2, 25.5, 22.9, 21.3, 18.5, 16.3 };
double ZbbP3EffPy6[8] = {    34.6,  30.7,  27.3, 21.0, 16.7, 11.7,  8.9,  7.4 };
double ZbbTotalEffPy6[8] = { 29.3,  33.0,  34.5, 36.6, 37.4, 37.4, 37.2, 37.1 };

//                                  D=13
double ZbbP1EffPy8[8] = {    27.3,  35.7,  39.9, 51.6, 60.1, 66.6, 72.6, 78.1 }; 
double ZbbP2EffPy8[8] = {    11.1,  18.6,  18.4, 20.6, 21.9, 19.5, 18.5, 16.9 };
double ZbbP3EffPy8[8] = {    26.0,  25.5,  24.8, 18.7, 16.6, 12.0,  9.2,  8.3 };
double ZbbTotalEffPy8[8] = { 21.2,  26.5,  27.8, 30.7, 33.5, 33.4, 34.3, 35.3 };

// ttbar+jets
//
const int NQCutTTbar = 7;
double QCutTTbar[7] = { 20., 25., 30., 35., 40., 45., 50. };

//                                                      D=40
/* 5K stat, DT6 tune
double TTbarP0EffPy6[7] = {    40.3, 49.8, 55.9, 60.4,  64.5,  69.7, 71.6 };
double TTbarP1EffPy6[7] = {    26.9, 28.7, 29.2, 28.2,  27.4,  27.6, 25.1 };
double TTbarP2EffPy6[7] = {    19.0, 18.2, 19.3, 16.1,  13.0,  10.2,  7.6 };
double TTbarP3EffPy6[7] = {    27.4, 19.6, 18.0, 12.5,  10.6,   5.4,  5.8 };
double TTbarTotalEffPy6[7] = { 28.9, 31.4, 33.2, 33.2,  33.2,  33.5, 32.7 }; 
*/

/* 25K stat, Z2 tune */
//                                                     D=40
double TTbarP0EffPy6[7] = {    31.0, 38.2, 45.0, 50.3, 55.6, 59.8, 64.5 };
double TTbarP1EffPy6[7] = {    22.1, 25.1, 25.8, 26.7, 25.7, 24.9, 24.6 };
double TTbarP2EffPy6[7] = {    16.4, 18.4, 17.6, 15.1, 13.2, 12.6, 10.2 };
double TTbarP3EffPy6[7] = {    27.1, 22.2, 16.9, 12.9, 10.1,  7.4,  5.6 };
double TTbarTotalEffPy6[7] = { 23.8, 27.0, 28.6, 29.4, 29.9, 30.5, 31.0 };

/* 5K stat
//                                                      D=40
double TTbarP0EffPy8[7] = {    23.7, 31.2, 35.2, 40.7,  45.3,  49.0, 53.5 };
double TTbarP1EffPy8[7] = {    18.2, 21.7, 23.1, 25.5,  24.6,  25.5, 22.2 };
double TTbarP2EffPy8[7] = {    14.2, 14.1, 13.0, 12.7,  10.5,  10.9,  9.8 };
double TTbarP3EffPy8[7] = {    24.0, 19.0, 15.2, 11.1,   8.8,   7.7,  5.5 };
double TTbarTotalEffPy8[7] = { 19.4, 22.3, 23.3, 25.3,  25.6,  27.0, 26.7 };
*/

/* 25K stat */
//                                                     D=40
double TTbarP0EffPy8[7] = {    24.3, 29.7, 36.0, 40.3, 46.0, 49.9, 54.0 };
double TTbarP1EffPy8[7] = {    18.3, 21.7, 24.7, 24.7, 24.3, 24.3, 23.1 };
double TTbarP2EffPy8[7] = {    14.0, 14.5, 14.9, 13.4, 12.8, 12.7, 10.5 };
double TTbarP3EffPy8[7] = {    24.4, 19.3, 15.0, 12.1,  9.9,  9.4,  5.7 };
double TTbarTotalEffPy8[7] = { 19.6, 22.0, 24.6, 25.2, 26.4, 26.8, 27.4 };


//  avjets
//
const int NQCutAVjets = 7;
double QCutAVjets[7] = { 10., 15., 20., 25., 30., 35., 40. };

//                                     D=15
double AVjetsP0EffPy6[7] = {    51.2,  66.3,  76.5, 83.8, 87.1, 89.6, 91.8 };
double AVjetsP1EffPy6[7] = {    68.1,  55.8,  41.8, 35.4, 25.3, 21.6, 14.7 };
double AVjetsP2EffPy6[7] = {    52.5,  66.0,  75.7, 82.1, 87.5, 90.6, 92.1 };
double AVjetsP3EffPy6[7] = {    70.3,  60.1,  51.3, 40.8, 35.2, 29.5, 26.5 };
double AVjetsP4EffPy6[7] = {    50.0,  60.6,  72.9, 78.0, 80.7, 86.7, 90.4 };
double AVjetsP5EffPy6[7] = {    70.5,  50.2,  43.5, 36.2, 28.0, 27.5, 20.8 };
double AVjetsTotalEffPy6[7] = { 60.3,  62.2,  63.0, 62.1, 61.4, 60.8, 59.9 };

//                                     D=15
double AVjetsP0EffPy8[7] = {    40.0,  55.1,  68.5, 75.7, 81.1, 83.0, 86.3 };
double AVjetsP1EffPy8[7] = {    65.7,  59.2,  47.9, 34.7, 29.7, 24.6, 17.6 };
double AVjetsP2EffPy8[7] = {    42.0,  55.9,  69.1, 76.6, 80.0, 86.3, 88.4 };
double AVjetsP3EffPy8[7] = {    69.6,  58.9,  51.0, 44.1, 38.4, 32.7, 27.4 };
double AVjetsP4EffPy8[7] = {    35.3,  50.9,  62.8, 78.0, 76.6, 81.7, 84.4 };
double AVjetsP5EffPy8[7] = {    62.3,  58.5,  43.0, 34.8, 35.3, 31.4, 20.8 };
double AVjetsTotalEffPy8[7] = { 53.7,  57.0,  59.6, 60.0, 59.5, 59.7, 58.2 };


//
// These are for 100K stat
//
/*
const int NQCutZvv = 7;
double QCutZvv[7] = { 10., 15., 20, 25., 30., 35., 40. };

double ZvvP1EffPy6[7] = { 11.0, 17.6, 18.7, 35.2, 33.0, 50.5, 52.7 };
double ZvvP2EffPy6[7] = {  5.9, 11.4, 15.1, 21.9, 23.5, 32.1, 33.7 };
double ZvvP3EffPy6[7] = {  4.0,  7.9,  9.8, 12.1, 11.5, 10.5, 11.6 };
double ZvvP4EffPy6[7] = { 14.0, 11.9,  9.4, 7.5,   5.0,  3.5,  3.1 };
double ZvvTotalEffPy6[7] = {  9.8, 10.7, 10.3, 11.1,  9.7, 10.0, 10.1 };

double ZvvP1EffPy8[7] = {  6.6, 15.4, 24.2, 31.9, 34.1, 40.7, 58.2 };
double ZvvP2EffPy8[7] = {  4.3, 11.2, 19.6, 20.9, 27.4, 34.2, 35.6 };
double ZvvP3EffPy8[7] = {  3.8,  8.1, 11.2, 11.8, 12.1, 10.9, 11.6 };
double ZvvP4EffPy8[7] = { 13.7, 10.8, 10.2,  6.8,  6.0,  4.6,  3.2 };
double ZvvTotalEffPy8[7] = { 9.4, 10.0, 11.8, 10.4, 10.9, 10.6,  10.5 };
*/

//--------------------------------------------


void plotZvvEff()
{

   TCanvas* myc = new TCanvas("myc","",600,1000);
   
   TPad* pad1 = new TPad("pad1","", 0.01, 0.51, 0.99, 0.99);
   TPad* pad2 = new TPad("pad2","", 0.01, 0.01, 0.49, 0.49);
   TPad* pad3 = new TPad("pad3","", 0.51, 0.01, 0.99, 0.49);
   
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   
   pad1->cd();
   
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);

   TGraph* grTotalEffPy6 = new TGraph( NQCutZvv, QCutZvv, ZvvTotalEffPy6 );
   grTotalEffPy6->SetTitle("Z #rightarrow nu nu (MG + Py6/Py8, xqcut=10)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutZvv, QCutZvv, ZvvTotalEffPy8 );
   //grTotalEffPy8->SetTitle("Z #rightarrow nu nu (MG + Py6/Py8, xqcut=10)" );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
   
   pad2->cd();
   plotZvvEffPy6();
   
   pad3->cd();
   plotZvvEffPy8();
   
   myc->cd();

   return;

}

void plotZvvEffPy6()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutZvv, QCutZvv, ZvvTotalEffPy6 );
   grTotalEff->SetTitle("Z #rightarrow nu nu (MG+Py6, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/
   TGraph* grP1Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP1EffPy6 );
   grP1Eff->SetTitle("Z #rightarrow nu nu (MG+Py6, xqcut=10)" );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->GetXaxis()->SetTitle("qcut");
   grP1Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP1Eff->GetYaxis()->SetTitleOffset(1.5);
   // grP1Eff->Draw("plsame");
   grP1Eff->Draw("apl");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP2EffPy6 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP3EffPy6 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");

   TGraph* grP4Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP4EffPy6 );
   grP4Eff->SetMarkerStyle(25);
   grP4Eff->SetMarkerSize(1.5);
   grP4Eff->SetMarkerColor(kBlack);
   grP4Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP4Eff->Draw("plsame");
   leg->AddEntry( grP4Eff, "Process 4, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/
   leg->Draw();
   leg->SetFillColor(kWhite);

   return;

}

void plotZvvEffPy8()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutZvv, QCutZvv, ZvvTotalEffPy8 );
   grTotalEff->SetTitle("Z #rightarrow nu nu (MG+Py8, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/
   TGraph* grP1Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP1EffPy8 );
   grP1Eff->SetTitle("Z #rightarrow nu nu (MG+Py8, xqcut=10)" );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->GetXaxis()->SetTitle("qcut");
   grP1Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP1Eff->GetYaxis()->SetTitleOffset(1.5);
   // grP1Eff->Draw("plsame");
   grP1Eff->Draw("apl");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP2EffPy8 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP3EffPy8 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");

   TGraph* grP4Eff = new TGraph( NQCutZvv, QCutZvv, ZvvP4EffPy8 );
   grP4Eff->SetMarkerStyle(25);
   grP4Eff->SetMarkerSize(1.5);
   grP4Eff->SetMarkerColor(kBlack);
   grP4Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP4Eff->Draw("plsame");
   leg->AddEntry( grP4Eff, "Process 4, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/
   leg->Draw();
   leg->SetFillColor(kWhite);

   return;

}

/*
void plotZvv()
{

   TCanvas *myc = new TCanvas("myc","",1000,600);
   myc->Divide(2,1);
   
   myc->cd(1);
   // gPad->SetLeftMargin(0.15);
   plotZvvEffPy6();
   
   myc->cd(2);
   plotZvvEffPy8();
   
   myc->cd();

   return;

}
*/

void plotTTbarEff()
{

   TCanvas* myc = new TCanvas("myc","",600,1000);
   
   TPad* pad1 = new TPad("pad1","", 0.01, 0.51, 0.99, 0.99);
   TPad* pad2 = new TPad("pad2","", 0.01, 0.01, 0.49, 0.49);
   TPad* pad3 = new TPad("pad3","", 0.51, 0.01, 0.99, 0.49);
   
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   
   pad1->cd();
   
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);

   TGraph* grTotalEffPy6 = new TGraph( NQCutTTbar, QCutTTbar, TTbarTotalEffPy6 );
   grTotalEffPy6->SetTitle("ttbar + jets (MG + Py6/Py8, xqcut=20)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutTTbar, QCutTTbar, TTbarTotalEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
   
   pad2->cd();
   plotTTbarEffPy6();
   
   pad3->cd();
   plotTTbarEffPy8();
   
   myc->cd();

   return;

}

void plotTTbarEffPy6()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutTTbar, QCutTTbar, TTbarTotalEffPy6 );
   grTotalEff->SetTitle("ttbar + jets (MG+Py6, xqcut=20)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/

   TGraph* grP0Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP0EffPy6 );
   grP0Eff->SetTitle("ttbar + jets (MG+Py6, xqcut=20)" );
   grP0Eff->SetMarkerStyle(26);
   grP0Eff->SetMarkerSize(1.5);
   grP0Eff->SetMarkerColor(kBlack); // 7 = very light sky-blue
   grP0Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP0Eff->GetXaxis()->SetTitle("qcut");
   grP0Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP0Eff->GetYaxis()->SetTitleOffset(1.5);
   // grP0Eff->Draw("plsame");
   grP0Eff->Draw("apl");
   leg->AddEntry( grP0Eff, "Process 0, Eff.", "p");

   TGraph* grP1Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP1EffPy6 );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP2EffPy6 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP3EffPy6 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/

   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

void plotTTbarEffPy8()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutTTbar, QCutTTbar, TTbarTotalEffPy8 );
   grTotalEff->SetTitle("ttbar + jets (MG+Py8, xqcut=20)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/

   TGraph* grP0Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP0EffPy8 );
   grP0Eff->SetTitle("ttbar + jets (MG+Py8, xqcut=20)" );
   grP0Eff->SetMarkerStyle(26);
   grP0Eff->SetMarkerSize(1.5);
   grP0Eff->SetMarkerColor(kBlack); // 7 = very light sky-blue
   grP0Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP0Eff->GetXaxis()->SetTitle("qcut");
   grP0Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP0Eff->GetYaxis()->SetTitleOffset(1.5);
   // grP0Eff->Draw("plsame");
   grP0Eff->Draw("apl");
   leg->AddEntry( grP0Eff, "Process 0, Eff.", "p");

   TGraph* grP1Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP1EffPy8 );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP2EffPy8 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutTTbar, QCutTTbar, TTbarP3EffPy8 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/
   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

/*
void plotTTbar()
{

   TCanvas *myc = new TCanvas("myc","",1000,600);
   myc->Divide(2,1);
   
   myc->cd(1);
   // gPad->SetLeftMargin(0.15);
   plotTTbarEffPy6();
   
   myc->cd(2);
   plotTTbarEffPy8();
   
   myc->cd();
   
   return;

}
*/

void plotZbbEff()
{

   TCanvas* myc = new TCanvas("myc","",600,1000);
   
   TPad* pad1 = new TPad("pad1","", 0.01, 0.51, 0.99, 0.99);
   TPad* pad2 = new TPad("pad2","", 0.01, 0.01, 0.49, 0.49);
   TPad* pad3 = new TPad("pad3","", 0.51, 0.01, 0.99, 0.49);
   
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   
   pad1->cd();
   
   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);

   TGraph* grTotalEffPy6 = new TGraph( NQCutZbb, QCutZbb, ZbbTotalEffPy6 );
   grTotalEffPy6->SetTitle("Z #rightarrow b bbar (MG + Py6/Py8, xqcut=10)" );
   grTotalEffPy6->SetMarkerStyle(21);
   grTotalEffPy6->SetMarkerSize(1.5);
   grTotalEffPy6->SetMarkerColor(kBlue);
   grTotalEffPy6->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEffPy6->GetXaxis()->SetTitle("qcut");
   grTotalEffPy6->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEffPy6->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy6->Draw("apl");

   TGraph* grTotalEffPy8 = new TGraph( NQCutZbb, QCutZbb, ZbbTotalEffPy8 );
   grTotalEffPy8->SetMarkerStyle(21);
   grTotalEffPy8->SetMarkerSize(1.5);
   grTotalEffPy8->SetMarkerColor(kRed);
   grTotalEffPy8->GetYaxis()->SetRangeUser(0.,100.);
   //grTotalEffPy8->GetXaxis()->SetTitle("qcut");
   //grTotalEffPy8->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   //grTotalEffPy8->GetYaxis()->SetTitleOffset(1.5);
   grTotalEffPy8->Draw("plsame");

   leg->AddEntry( grTotalEffPy6, "Total Eff., Pythia6", "p");
   leg->AddEntry( grTotalEffPy8, "Total Eff., Pythia8", "p");
   leg->SetFillColor(kWhite);
   leg->Draw();
   
   pad2->cd();
   plotZbbEffPy6();
   
   pad3->cd();
   plotZbbEffPy8();
   
   myc->cd();

   return;

}

void plotZbbEffPy6()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutZbb, QCutZbb, ZbbTotalEffPy6 );
   grTotalEff->SetTitle("Z #rightarrow b bbar (MG+Py6, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/
   TGraph* grP1Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP1EffPy6 );
   grP1Eff->SetTitle("Z #rightarrow b bbar (MG+Py6, xqcut=10)" );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->GetXaxis()->SetTitle("qcut");
   grP1Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP1Eff->GetYaxis()->SetTitleOffset(1.5);
   grP1Eff->Draw("apl");
   // grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP2EffPy6 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP3EffPy6 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/

   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

void plotZbbEffPy8()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
/*
   TGraph* grTotalEff = new TGraph( NQCutZbb, QCutZbb, ZbbTotalEffPy8 );
   grTotalEff->SetTitle("Z #rightarrow b bbar (MG+Py8, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");
*/
   TGraph* grP1Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP1EffPy8 );
   grP1Eff->SetTitle("Z #rightarrow b bbar (MG+Py8, xqcut=10)" );
   grP1Eff->SetMarkerStyle(22);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kBlack);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->GetXaxis()->SetTitle("qcut");
   grP1Eff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grP1Eff->GetYaxis()->SetTitleOffset(1.5);
   grP1Eff->Draw("apl");
   // grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP2EffPy8 );
   grP2Eff->SetMarkerStyle(23);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlack);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutZbb, QCutZbb, ZbbP3EffPy8 );
   grP3Eff->SetMarkerStyle(24);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kBlack);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");
   
/*
   grTotalEff->Draw("plsame");
*/
   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

/*
void plotZbb()
{

   TCanvas *myc = new TCanvas("myc","",1000,600);
   myc->Divide(2,1);
   
   myc->cd(1);
   // gPad->SetLeftMargin(0.15);
   plotZbbEffPy6();
   
   myc->cd(2);
   plotZbbEffPy8();
   
   myc->cd();
   
   return;

}
*/

void plotAVjetsEffPy6()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
   TGraph* grTotalEff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsTotalEffPy6 );
   grTotalEff->SetTitle("avjets (MG+Py6, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");

   TGraph* grP0Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP0EffPy6 );
   grP0Eff->SetMarkerStyle(21);
   grP0Eff->SetMarkerSize(1.5);
   grP0Eff->SetMarkerColor(7); // very light sky-blue
   grP0Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP0Eff->Draw("plsame");
   leg->AddEntry( grP0Eff, "Process 0, Eff.", "p");

   TGraph* grP1Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP1EffPy6 );
   grP1Eff->SetMarkerStyle(21);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kRed);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP2EffPy6 );
   grP2Eff->SetMarkerStyle(21);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlue);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP3EffPy6 );
   grP3Eff->SetMarkerStyle(21);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kGreen);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");

   TGraph* grP4Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP4EffPy6 );
   grP4Eff->SetMarkerStyle(21);
   grP4Eff->SetMarkerSize(1.5);
   grP4Eff->SetMarkerColor(kMagenta);
   grP4Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP4Eff->Draw("plsame");
   leg->AddEntry( grP4Eff, "Process 4, Eff.", "p");
   
   TGraph* grP5Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP5EffPy6 );
   grP5Eff->SetMarkerStyle(21);
   grP5Eff->SetMarkerSize(1.5);
   grP5Eff->SetMarkerColor(kMagenta+2);
   // grP5Eff->SetMarkerColor(kYellow+1);
   grP5Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP5Eff->Draw("plsame");
   leg->AddEntry( grP5Eff, "Process 5, Eff.", "p");

   grTotalEff->Draw("plsame");

   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

void plotAVjetsEffPy8()
{

   TLegend* leg = new TLegend(0.6, 0.70, 0.9, 0.9);
   
   TGraph* grTotalEff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsTotalEffPy8 );
   grTotalEff->SetTitle("avjets (MG+Py8, xqcut=10)" );
   grTotalEff->SetMarkerStyle(21);
   grTotalEff->SetMarkerSize(1.5);
   grTotalEff->SetMarkerColor(kBlack);
   grTotalEff->GetYaxis()->SetRangeUser(0.,100.);
   grTotalEff->GetXaxis()->SetTitle("qcut");
   grTotalEff->GetYaxis()->SetTitle("PS Matching Efficiencvy (%)");
   grTotalEff->GetYaxis()->SetTitleOffset(1.5);
   grTotalEff->Draw("apl");
   leg->AddEntry( grTotalEff, "Total Eff.", "p");

   TGraph* grP0Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP0EffPy8 );
   grP0Eff->SetMarkerStyle(21);
   grP0Eff->SetMarkerSize(1.5);
   grP0Eff->SetMarkerColor(7); // very light sky-blue
   grP0Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP0Eff->Draw("plsame");
   leg->AddEntry( grP0Eff, "Process 0, Eff.", "p");

   TGraph* grP1Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP1EffPy8 );
   grP1Eff->SetMarkerStyle(21);
   grP1Eff->SetMarkerSize(1.5);
   grP1Eff->SetMarkerColor(kRed);
   grP1Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP1Eff->Draw("plsame");
   leg->AddEntry( grP1Eff, "Process 1, Eff.", "p");

   TGraph* grP2Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP2EffPy8 );
   grP2Eff->SetMarkerStyle(21);
   grP2Eff->SetMarkerSize(1.5);
   grP2Eff->SetMarkerColor(kBlue);
   grP2Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP2Eff->Draw("plsame");
   leg->AddEntry( grP2Eff, "Process 2, Eff.", "p");
   
   TGraph* grP3Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP3EffPy8 );
   grP3Eff->SetMarkerStyle(21);
   grP3Eff->SetMarkerSize(1.5);
   grP3Eff->SetMarkerColor(kGreen);
   grP3Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP3Eff->Draw("plsame");
   leg->AddEntry( grP3Eff, "Process 3, Eff.", "p");

   TGraph* grP4Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP4EffPy8 );
   grP4Eff->SetMarkerStyle(21);
   grP4Eff->SetMarkerSize(1.5);
   grP4Eff->SetMarkerColor(kMagenta);
   grP4Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP4Eff->Draw("plsame");
   leg->AddEntry( grP4Eff, "Process 4, Eff.", "p");
   
   TGraph* grP5Eff = new TGraph( NQCutAVjets, QCutAVjets, AVjetsP5EffPy8 );
   grP5Eff->SetMarkerStyle(21);
   grP5Eff->SetMarkerSize(1.5);
   grP5Eff->SetMarkerColor(kMagenta+2);
   // grP5Eff->SetMarkerColor(kYellow+1);
   grP5Eff->GetYaxis()->SetRangeUser(0.,100.);
   grP5Eff->Draw("plsame");
   leg->AddEntry( grP5Eff, "Process 5, Eff.", "p");

   grTotalEff->Draw("plsame");

   leg->Draw();
   leg->SetFillColor(kWhite);
      
   return;

}

void plotAVjets()
{

   TCanvas *myc = new TCanvas("myc","",1000,600);
   myc->Divide(2,1);
   
   myc->cd(1);
   // gPad->SetLeftMargin(0.15);
   plotAVjetsEffPy6();
   
   myc->cd(2);
   plotAVjetsEffPy8();
   
   myc->cd();
   
   return;

}
