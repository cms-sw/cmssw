#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitXslices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * Edit the input and output files and eventually the rebins. <br>
 * To run this macro: root -l testSlices.cpp+
 */

void testSlices(const TString & inputFileName = "3_MuScleFit.root", const TString & outputFileName = "BiasCheck_3.root")
{
  gROOT->SetBatch(kTRUE);

  TFile * inputFile = new TFile(inputFileName, "READ");

  TFile * outputFile = new TFile(outputFileName, "RECREATE");

  FitXslices fitXslices;

  // Mass vs Pt
  TH2F * histoPt = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPt");
  histoPt->RebinX(4);
  histoPt->RebinY(2);
  fitXslices.fitter()->initMean(3.1, 2.9, 3.3);
  outputFile->mkdir("MassVsPt");
  outputFile->cd("MassVsPt");
  // fitXslices(histoPt, 3., 3.2, "doubleGaussian", "exponential");
  fitXslices(histoPt, 3., 3.2, "gaussian", "exponential");
  // fitXslices(histoPt, 2.5, 3.7);
  // Mass vs Eta
  TH2F * histoEta = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSEta");
  histoEta->RebinX(1);
  histoEta->RebinY(1);
  outputFile->mkdir("MassVsEta");
  outputFile->cd("MassVsEta");
  // fitXslices(histoEta, 3., 3.2, "gaussian", "");
  fitXslices(histoEta, 3., 3.2, "doubleGaussian", "exponential");
  // Mass vs PhiPlus
  TH2F * histoPhiPlus = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPhiPlus");
  histoPhiPlus->RebinX(2);
  histoPhiPlus->RebinY(2);
  outputFile->mkdir("MassVsPhiPlus");
  outputFile->cd("MassVsPhiPlus");
  fitXslices(histoPhiPlus, 3., 3.2, "gaussian", "exponential");
  // Mass vs PhiMinus
  TH2F * histoPhiMinus = (TH2F*)inputFile->Get("hRecBestResVSMu_MassVSPhiMinus");
  histoPhiMinus->RebinX(2);
  histoPhiMinus->RebinY(2);
  outputFile->mkdir("MassVsPhiMinus");
  outputFile->cd("MassVsPhiMinus");
  fitXslices(histoPhiMinus, 3., 3.2, "gaussian", "exponential");

  outputFile->Write();
  outputFile->Close();
}
