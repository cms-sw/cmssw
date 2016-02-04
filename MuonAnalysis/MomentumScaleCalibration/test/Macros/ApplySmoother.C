#include "Smoother.C"
#include "TFile.h"
#include "TDirectory.h"

void ApplySmoother()
{
  Smoother smoother(0.2);

  TFile* inputFile = new TFile("Sherpa_Zeta.root");
  TDirectory* inputDir = (TDirectory*)inputFile->Get("TestHepMCEvt");
  TH1F* inputHisto = (TH1F*)inputDir->Get("HistZMass");

  const int iterations = 4;
  bool single[iterations] = { false, true, false, true };
  TH1F* smoothedHisto = smoother.smooth(inputHisto, iterations, single);

  TFile* outputFile = new TFile("SmoothedHisto.root", "RECREATE");
  outputFile->cd();
  smoothedHisto->Write();
  outputFile->Close();
}
