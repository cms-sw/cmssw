#include "TProfile.h"
#include "TCanvas.h"
#include "TFile.h"
#include <iostream>

void CompareProb()
{
  TFile * inputFile = new TFile("2_MuScleFit_DATA_triggercut.root");
  TProfile * probHisto = (TProfile*)inputFile->FindObjectAny("Mass_fine_PProf");
  TH1F * massHisto = (TH1F*)inputFile->FindObjectAny("hRecBestRes_Mass");

  TCanvas * newCanvas = new TCanvas();
  newCanvas->Draw();

  double xMin = 2.85;
  double xMax = 3.3;

  std::cout << probHisto << ", " << massHisto << std::endl;

  probHisto->Scale(1/probHisto->Integral(probHisto->FindBin(xMin), probHisto->FindBin(xMax), "width"));
  probHisto->SetLineColor(2);
  probHisto->SetMarkerColor(2);
  probHisto->SetMarkerStyle(2);
  probHisto->SetMarkerSize(0.2);

  massHisto->Scale(1/massHisto->Integral(massHisto->FindBin(xMin), massHisto->FindBin(xMax), "width"));

  massHisto->Draw();
  probHisto->Draw("same");
  // probHisto->Draw();

}
