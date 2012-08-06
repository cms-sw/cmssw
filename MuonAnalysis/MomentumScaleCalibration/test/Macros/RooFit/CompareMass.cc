#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "RooHist.h"
#include "RooCurve.h"
#include "TPaveText.h"
#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitMass.cc"

void CompareMass()
{
  gROOT->SetStyle("Plain");

  TString fileNum1 = "0";
  TString fileNum2 = "3";

  TString fitType1("gaussianPlusCrystalBall");
  TString fitType2("gaussianPlusCrystalBall");
  TString backgroundType("exponential");

  FitMass fitMass1;
  fitMass1.rebinX = 2;
  fitMass1.fitter()->initMean(3.1, 3., 3.2);
  fitMass1.fitter()->initSigma(0.03, 0., 10.);
  fitMass1.fitter()->initSigma2(1., 0., 10.);
  fitMass1.fitter()->initAlpha(1.9, 0., 10.);
  fitMass1.fitter()->initN(0.4, 0., 10.);
  fitMass1.fitter()->initFGCB(0.4, 0., 1.);
  fitMass1.fit(2.95, 3.25, fileNum1, fitType1, backgroundType);
  FitMass fitMass2;
  fitMass2.rebinX = 2;
  fitMass2.fitter()->initMean(3.1, 3., 3.2);
  fitMass2.fitter()->initSigma(0.03, 0., 10.);
  fitMass2.fitter()->initSigma2(1., 0., 10.);
  fitMass2.fitter()->initAlpha(1.9, 0., 10.);
  fitMass2.fitter()->initN(0.4, 0., 10.);
  fitMass2.fitter()->initFGCB(0.4, 0., 1.);
  fitMass2.fit(2.95, 3.25, fileNum2, fitType2, backgroundType);

  TString fileName1("MassFit_"+fileNum1+".root");
  TString fileName2("MassFit_"+fileNum2+".root");  

  TFile * file1 = new TFile(fileName1, "READ");
  RooPlot * rooPlot1 = (RooPlot*)file1->Get("hRecBestResAllEvents_Mass_frame");

  TFile * file2 = new TFile(fileName2, "READ");
  RooPlot * rooPlot2 = (RooPlot*)file2->Get("hRecBestResAllEvents_Mass_frame");

  TFile * outputFile = new TFile("CompareMass.root", "RECREATE");
  outputFile->cd();
  TCanvas * canvas = new TCanvas("canvas", "canvas", 1000, 800);
  canvas->Draw();
  rooPlot1->Draw();
  rooPlot1->GetXaxis()->SetTitle("");
  rooPlot1->SetTitle("");
  rooPlot2->Draw("same");
  rooPlot2->getHist("h_dh")->SetLineColor(kRed);
  rooPlot2->getHist("h_dh")->SetMarkerColor(kRed);
  rooPlot2->GetXaxis()->SetTitle("Mass (GeV)");
  rooPlot2->GetYaxis()->SetTitle("");
  rooPlot2->SetTitle("");
  RooCurve * curve = 0;
  TPaveText * paveText = 0;
  if( backgroundType != "" ) {
    curve = (RooCurve*)rooPlot2->findObject("model_Norm[x]");
    paveText = (TPaveText*)rooPlot2->findObject("model_paramBox");
  }
  else if( fitType2 == "gaussian" ) {
    curve = (RooCurve*)rooPlot2->findObject("gaussian_Norm[x]");
    paveText = (TPaveText*)rooPlot2->findObject("gaussian_paramBox");
  }
  else if( fitType2 == "doubleGaussian" ) {
    paveText = (TPaveText*)rooPlot2->findObject("doubleGaussian_paramBox");
    curve = (RooCurve*)rooPlot2->findObject("doubleGaussian_Norm[x]");
  }
  else if( fitType2 == "voigtian" ) {
    paveText = (TPaveText*)rooPlot2->findObject("voigt_paramBox");
    curve = (RooCurve*)rooPlot2->findObject("voigt_Norm[x]");
  }
  if( curve != 0 ) curve->SetLineColor(kRed);
  if( paveText != 0 ) paveText->SetTextColor(kRed);
  canvas->Write();
  outputFile->Write();
  outputFile->Close();
}
