#include "TFile.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TH1D.h"
#include "TLegend.h"

#include "TROOT.h"
#include <map>
#include <iostream>


void getHistograms(const TString canvasName, TH1F * & histo1, TH1D * & histo2, const TString & resonance);

/**
 * This macro creates the histograms to compare mass before and after
 * the correction with mass prob in the same conditions. <br>
 * It reads from plotMassOutput.root created by Plot_Mass.C.
 */
void ShowMassComparison(const TString & resonance = "Z")
{
  gROOT->SetBatch(true);
  TString canvasName("Allres");
  if( resonance == "Psis" || resonance == "Upsilons" || resonance == "LowPtResonances" || resonance == "AllResonances" ) {
    canvasName += "Together";
  }
  TH1F * histo1 = 0;
  TH1D * histo2 = 0;
  getHistograms(canvasName, histo1, histo2, resonance);

  TH1F * histo3 = 0;
  TH1D * histo4 = 0;
  getHistograms(canvasName+"2", histo3, histo4, resonance);

TString option("width");
double integral = histo1->Integral(option);
histo2->Scale(integral/histo2->Integral(option));
histo3->Scale(integral/histo3->Integral(option));
histo4->Scale(integral/histo4->Integral(option));

//   histo1->Scale(1./histo1->GetEntries());
//   histo2->Scale(1./histo2->GetEntries());
//   histo3->Scale(1./histo3->GetEntries());
//   histo4->Scale(1./histo4->GetEntries());

//   double integral = histo1->Integral(histo1->GetXaxis()->FindBin(9.1), histo1->GetXaxis()->FindBin(10.8));
//   histo1->Scale(1./integral);
//   integral = histo2->Integral(histo2->GetXaxis()->FindBin(9.1), histo2->GetXaxis()->FindBin(10.8));
//   histo2->Scale(1./integral);
//   integral = histo3->Integral(histo3->GetXaxis()->FindBin(9.1), histo3->GetXaxis()->FindBin(10.8));
//   histo3->Scale(1./integral);
//   integral = histo4->Integral(histo4->GetXaxis()->FindBin(9.1), histo4->GetXaxis()->FindBin(10.8));
//   histo4->Scale(1./integral);


//   histo1->Scale(1./histo1->Integral());
//   histo2->Scale(1./histo2->Integral());
//   histo3->Scale(1./histo3->Integral());
//   histo4->Scale(1./histo4->Integral());

  std::map<double, TH1*, greater<double> > histoMap;
  histoMap.insert(make_pair(histo1->GetMaximum(), histo1));
  histoMap.insert(make_pair(histo2->GetMaximum(), histo2));
  histoMap.insert(make_pair(histo3->GetMaximum(), histo3));
  histoMap.insert(make_pair(histo4->GetMaximum(), histo4));

  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  histo4->SetLineColor(kBlue);
  // histo4->SetLineStyle(2);
  histo4->SetMarkerColor(kBlue);
  histo2->SetMarkerColor(kRed);
  histo2->SetLineColor(kRed);
  // histo2->SetLineStyle(2);
  histo1->SetLineColor(kBlack);
  histo3->SetLineColor(kGreen);

  std::map<double, TH1*, greater<double> >::const_iterator it = histoMap.begin();
  it->second->Draw();
  for( ; it != histoMap.end(); ++it ) it->second->Draw("SAME");

  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(histo1, "mass before correction");
  legend->AddEntry(histo2, "mass prob before correction");
  legend->AddEntry(histo3, "mass after correction");
  legend->AddEntry(histo4, "mass prob after correction");
  legend->Draw("SAME");

  TFile * outputFile = new TFile("ShowMassComparison.root", "RECREATE");
  newCanvas->Write();
  outputFile->Close();
}


// In Progress:
// Histograms normalized in each region before putting all together
// ----------------------------------------------------------------
void fillMapAndLegend( const TString & canvasName, const TString & resonance, std::map<double, TH1*, greater<double> > & histoMap, TLegend * legend = 0 )
{
  TH1F * histo1 = 0;
  TH1D * histo2 = 0;
  getHistograms(canvasName, histo1, histo2, resonance);
  TH1F * histo3 = 0;
  TH1D * histo4 = 0;
  getHistograms(canvasName+"2", histo3, histo4, resonance);

  histo2->Scale(histo1->GetEntries()/histo2->GetEntries());
  histo3->Scale(histo1->GetEntries()/histo3->GetEntries());
  histo4->Scale(histo1->GetEntries()/histo4->GetEntries());

  histoMap.insert(make_pair(histo1->GetMaximum(), histo1));
  histoMap.insert(make_pair(histo2->GetMaximum(), histo2));
  histoMap.insert(make_pair(histo3->GetMaximum(), histo3));
  histoMap.insert(make_pair(histo4->GetMaximum(), histo4));

  histo4->SetLineColor(kBlue);
  histo4->SetMarkerColor(kBlue);
  histo2->SetMarkerColor(kRed);
  histo2->SetLineColor(kRed);
  histo1->SetLineColor(kBlack);
  histo3->SetLineColor(kGreen);

  if( legend != 0 ) {
    legend->SetTextSize(0.02);
    legend->SetFillColor(0); // Have a white background
    legend->AddEntry(histo1, "mass before correction");
    legend->AddEntry(histo2, "mass prob before correction");
    legend->AddEntry(histo3, "mass after correction");
    legend->AddEntry(histo4, "mass prob after correction");
  }
}

void ShowMassesComparison(const TString & resonance = "Z")
{
  TString canvasName("Allres");

  std::map<double, TH1*, greater<double> > histoMap;
  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);

  fillMapAndLegend(canvasName, "Upsilon", histoMap, legend);
  fillMapAndLegend(canvasName, "Upsilon2S", histoMap);
  fillMapAndLegend(canvasName, "Upsilon3S", histoMap);

  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  std::cout << "size = " << histoMap.size() << std::endl;
  std::map<double, TH1*, greater<double> >::const_iterator it = histoMap.begin();
  it->second->Draw();
  it->second->SetAxisRange(9,11);
  for( ; it != histoMap.end(); ++it ) it->second->Draw("SAME");
  legend->Draw("SAME");

  TFile * outputFile = new TFile("ShowMassComparison.root", "RECREATE");
  newCanvas->Write();
  outputFile->Close();
}
// ----------------------------------------------------------------------------

/**
 * Helper function to extract the histograms from the canvas file. <br>
 * Takes references to pointers in order to fill them.
 */
void getHistograms(const TString canvasName, TH1F * & histo1, TH1D * & histo2, const TString & resonance)
{
  std::cout << "canvasName = " << canvasName << std::endl;
  TFile * inputFile = new TFile("plotMassOutput.root");
  TCanvas * canvas = (TCanvas*)inputFile->Get(canvasName);
  TString resonanceNum("_1");
  if( resonance == "Upsilon3S" ) resonanceNum = "_2";
  if( resonance == "Upsilon2S" ) resonanceNum = "_3";
  if( resonance == "Upsilon" ) resonanceNum = "_4";
  if( resonance == "Psi2S" ) resonanceNum = "_5";
  if( resonance == "JPsi" ) resonanceNum = "_6";

  if( resonance == "Psis" ) resonanceNum = "_1";
  if( resonance == "Upsilons" ) resonanceNum = "_2";
  if( resonance == "LowPtResonances" ) resonanceNum = "_3";
  if( resonance == "AllResonances" ) resonanceNum = "_4";

  TPad * pad = (TPad*)canvas->GetPrimitive(canvasName+resonanceNum);
  histo1 = (TH1F*)pad->GetPrimitive("hRecBestResAllEvents_Mass");
  if( resonance == "Z" || resonance == "AllResonances" ) histo2 = (TH1D*)pad->GetPrimitive("Mass_PProf");
  else histo2 = (TH1D*)pad->GetPrimitive("Mass_fine_PProf");
  // if( resonance == "Z" || resonance == "AllResonances" ) histo2 = (TH1D*)pad->GetPrimitive("Mass_Probability");
  // else histo2 = (TH1D*)pad->GetPrimitive("Mass_fine_Probability");

  // std::cout << "histo1 = " << histo1 << ", histo2 = " << histo2 << std::endl;
  // std::cout << "histo1 = " << histo1->GetEntries() << ", histo2 = " << histo2->GetEntries() << std::endl;
}

