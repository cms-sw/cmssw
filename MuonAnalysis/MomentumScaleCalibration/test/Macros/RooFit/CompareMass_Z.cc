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
  // TString fileName1("../FSRmodel/MassFit_3.root");
  TString fileNum1 = "0";
  TString fileNum2 = "2";

  TString fitType1("voigtian");
  TString fitType2("voigtian");
  TString backgroundType("exponential");

  FitMass fitMass1;
  fitMass1.fitter()->initMean(91., 80., 100.);
  fitMass1.fitter()->initSigma(2.5, 0., 10.);
  fitMass1.fit(80., 100., fileNum1, fitType1, backgroundType);
  FitMass fitMass2;
  fitMass2.fitter()->initMean(91., 80., 100.);
  fitMass2.fitter()->initSigma(2.5, 0., 10.);
  fitMass2.fit(80., 100., fileNum2, fitType2, backgroundType);

  TString fileName1("MassFit_"+fileNum1+".root");
  TString fileName2("MassFit_"+fileNum2+".root");  

  TFile * file1 = new TFile(fileName1, "READ");
  RooPlot * rooPlot1 = (RooPlot*)file1->Get("hRecBestRes_Mass_frame");

  TFile * file2 = new TFile(fileName2, "READ");
  RooPlot * rooPlot2 = (RooPlot*)file2->Get("hRecBestRes_Mass_frame");

  TFile * outputFile = new TFile("CompareMass.root", "RECREATE");
  outputFile->cd();
  TCanvas * canvas = new TCanvas("canvas", "canvas", 1000, 800);
  canvas->Draw();
  rooPlot1->Draw();
  rooPlot2->Draw("same");
  rooPlot2->getHist("h_dh")->SetLineColor(kRed);
  rooPlot2->getHist("h_dh")->SetMarkerColor(kRed);
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
    curve = (RooCurve*)rooPlot2->findObject("doubleGaussian_Norm[x]");
    paveText = (TPaveText*)rooPlot2->findObject("doubleGaussian_paramBox");
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
