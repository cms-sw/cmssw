#include "TFile.h"
#include "TCanvas.h"
#include "TPad.h"
#include "TH1F.h"
#include "TProfile.h"
#include "TLegend.h"

#include <map>
#include <iostream>

using namespace std;

void getHistograms(const TString canvasName, TH1F * & histo1, TProfile * & histo2, const TString & resonance);


/**
 * This macro creates the histograms to compare mass before and after
 * the correction with mass prob in the same conditions. <br>
 * It reads from plotMassOutput.root created by Plot_Mass.C.
 */
void ShowMassComparison(const TString & resonance = "Z")
{
  TH1F * histo1 = 0;
  TProfile * histo2 = 0;
  getHistograms("Allres", histo1, histo2, resonance);

  TH1F * histo3 = 0;
  TProfile * histo4 = 0;
  getHistograms("Allres2", histo3, histo4, resonance);

  map<double, TH1*, greater<double> > histoMap;
  histoMap.insert(make_pair(histo1->GetMaximum(), histo1));
  histoMap.insert(make_pair(histo2->GetMaximum(), histo2));
  histoMap.insert(make_pair(histo3->GetMaximum(), histo3));
  histoMap.insert(make_pair(histo4->GetMaximum(), histo4));

  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  histo4->SetLineColor(kBlue);
  histo4->SetMarkerColor(kBlue);
  histo2->SetMarkerColor(kRed);
  histo2->SetLineColor(kRed);
  histo1->SetLineColor(kBlack);
  histo3->SetLineColor(kGreen);

  map<double, TH1*, greater<double> >::const_iterator it = histoMap.begin();
  it->second->Draw();
  for( ; it != histoMap.end(); ++it ) it->second->Draw("SAME");

//   if( histo2->GetMaximum() > histo4->GetMaximum() ) {
//     histo2->Draw();
//     histo4->Draw("SAME");
//   }
//   else {
//     histo4->Draw();
//     histo2->Draw("SAME");
//   }
//   histo1->Draw("SAME");
//   histo3->Draw("SAME");
//   histo2->Draw("SAME");
//   histo4->Draw("SAME");

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

/**
 * Helper function to extract the histograms from the canvas file. <br>
 * Takes references to pointers in order to fill them.
 */
void getHistograms(const TString canvasName, TH1F * & histo1, TProfile * & histo2, const TString & resonance)
{
  TFile * inputFile = new TFile("plotMassOutput.root");
  TCanvas * canvas = (TCanvas*)inputFile->Get(canvasName);
  TString resonanceNum("_1");
  if( resonance == "Upsilon3S" ) resonanceNum = "_2";
  if( resonance == "Upsilon2S" ) resonanceNum = "_3";
  if( resonance == "Upsilon" ) resonanceNum = "_4";
  if( resonance == "Psi2S" ) resonanceNum = "_5";
  if( resonance == "JPsi" ) resonanceNum = "_6";

  TPad * pad = (TPad*)canvas->GetPrimitive(canvasName+resonanceNum);
  histo1 = (TH1F*)pad->GetPrimitive("hRecBestRes_Mass");
  histo2 = (TProfile*)pad->GetPrimitive("Mass_P");
  // cout << "histo1 = " << histo1 << ", histo2 = " << histo2 << endl;
  // cout << "histo1 = " << histo1->GetEntries() << ", histo2 = " << histo2->GetEntries() << endl;
}
