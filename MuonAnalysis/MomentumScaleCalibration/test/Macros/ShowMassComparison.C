// #include "TFile.h"
// #include "TCanvas.h"
// #include "TPad.h"
// #include "TH1F.h"
// #include "TProfile.h"

// #include <iostream>

using namespace std;

void getHistograms(const TString canvasName, TH1F * & histo1, TProfile * & histo2);


/**
 * This macro creates the histograms to compare mass before and after
 * the correction with mass prob in the same conditions. <br>
 * It reads from plotMassOutput.root created by Plot_Mass.C.
 */
void ShowMassComparison()
{

  TH1F * histo1 = 0;
  TProfile * histo2 = 0;
  getHistograms("Allres", histo1, histo2);

  TH1F * histo3 = 0;
  TProfile * histo4 = 0;
  getHistograms("Allres2", histo3, histo4);

  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  histo4->SetLineColor(kGreen);
  histo4->Draw();
  histo1->Draw("SAME");
  histo2->Draw("SAME");
  histo3->SetLineColor(kBlack);
  histo3->Draw("SAME");
  histo4->SetMarkerColor(kGreen);

  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(histo1, "mass before correction");
  legend->AddEntry(histo2, "mass prob before correction");
  legend->AddEntry(histo3, "mass after correction");
  legend->AddEntry(histo4, "mass prob after correction");
  legend->Draw("SAME");
}

/**
 * Helper function to extract the histograms from the canvas file. <br>
 * Takes references to pointers in order to fill them.
 */
void getHistograms(const TString canvasName, TH1F * & histo1, TProfile * & histo2)
{
  TFile * inputFile = new TFile("plotMassOutput.root");
  TCanvas * canvas = (TCanvas*)inputFile->Get(canvasName);
  TPad * pad = (TPad*)canvas->GetPrimitive(canvasName+"_1");
  histo1 = (TH1F*)pad->GetPrimitive("hRecBestRes_Mass");
  histo2 = (TProfile*)pad->GetPrimitive("Mass_P");
  // cout << "histo1 = " << histo1 << ", histo2 = " << histo2 << endl;
}
