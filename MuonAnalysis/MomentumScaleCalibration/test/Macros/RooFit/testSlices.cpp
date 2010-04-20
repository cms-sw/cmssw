#include "FitXslices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * Edit the input and output files and eventually the rebins. <br>
 * To run this macro: root -l testSlices.cpp+
 */

void testSlices()
{
  gROOT->SetBatch(kTRUE);

  TFile * inputFile = new TFile("0_MuScleFit.root", "READ");

  TFile * outputFile = new TFile("BiasCheck_Psis_0.root", "RECREATE");

  FitXslices fitXslices;

  // Mass vs Pt
  TH2F * histoPt = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPt");
  histoPt->RebinX(4);
  histoPt->RebinY(2);
  fitXslices.fitter()->initMean(3.1, 2.9, 3.3);
  outputFile->mkdir("MassVsPt");
  outputFile->cd("MassVsPt");
  fitXslices(histoPt, 2.9, 3.3, "doubleGaussian", "exponential");
  // fitXslices(histoPt, 2.5, 3.7);
  // Mass vs Eta
  TH2F * histoEta = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSEta");
  histoEta->RebinX(2);
  histoEta->RebinY(4);
  outputFile->mkdir("MassVsEta");
  outputFile->cd("MassVsEta");
  // fitXslices(histoEta, 2.9, 3.3, "doubleGaussian", "exponential");
  fitXslices(histoEta, 2.5, 3.5, "doubleGaussian", "exponential");
  // Mass vs PhiPlus
  TH2F * histoPhiPlus = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPhiPlus");
  histoPhiPlus->RebinX(4);
  histoPhiPlus->RebinY(4);
  outputFile->mkdir("MassVsPhiPlus");
  outputFile->cd("MassVsPhiPlus");
  fitXslices(histoPhiPlus, 2.9, 3.3, "doubleGaussian", "exponential");
  // Mass vs PhiMinus
  TH2F * histoPhiMinus = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPhiMinus");
  histoPhiMinus->RebinX(4);
  histoPhiMinus->RebinY(4);
  outputFile->mkdir("MassVsPhiMinus");
  outputFile->cd("MassVsPhiMinus");
  fitXslices(histoPhiMinus, 2.9, 3.3, "doubleGaussian", "exponential");

  outputFile->Write();
  outputFile->Close();
}
