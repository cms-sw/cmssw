#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussModel.h"
#include "RooAddModel.h"
#include "RooTruthModel.h"
#include "RooDecay.h"
#include "RooPlot.h"
#include "RooLandau.h"
#include "RooGaussian.h"
#include "RooNumConvPdf.h"
#include "RooExponential.h"
#include "RooCBShape.h"
#include "TCanvas.h"
#include "TH1.h"
#include "RooGenericPdf.h"
#include "RooAddPdf.h"
#include <sstream>
#include "TFile.h"
#include "RooDataHist.h"
using namespace RooFit;

void GaussianFitOnData()
{
  TFile inputFile("0_MuScleFit.root", "READ");
  TH1F * inputHisto = (TH1F*)inputFile.Get("hRecBestResAllEvents_Mass");
  inputHisto->Rebin(10);

  // RooRealVar x("x", "Mass (GeV)", 2., 4.);
  // double xMin = 3.3;
  // double xMax = 4.4;
  double xMin = 2.;
  double xMax = 4.;
  RooRealVar x("x", "Mass (GeV)", xMin, xMax);

  RooDataHist histo("dh", "dh", x, Import(*inputHisto));

  // Build the CB
  RooRealVar peak("peak","peak", 3.1, 2.8, 3.8);
  // RooRealVar peak("peak","peak", 3.695, 3.6, 3.8);
  RooRealVar sigma("sigma","sigma", 0.06, 0.001, 0.1);
  RooGaussModel signal("signal", "signal", x, peak, sigma);

  // Build the exponential background pdf
  RooRealVar expCoeff("expCoeff", "exponential coefficient", -1., -10., 10.);
  RooExponential background("exponential", "exponential", x, expCoeff);

  TFile outputFile("gaussianFitOnData.root", "RECREATE"); 

  // // Fit the background in the sidebands
  // x.setRange("sb_lo", 2., 2.6);
  // x.setRange("sb_hi", 3.8, 4.4);
  // background.fitTo(histo, Range("sb_lo,sb_hi"));

  // TCanvas canvasBackground("canvasBackgroundFit", "canvasBackgroundFit", 1000, 800);
  // canvasBackground.cd();
  // canvasBackground.Draw();
  // RooPlot* frameBackground = x.frame();
  // histo.plotOn(frameBackground);
  // background.plotOn(frameBackground);
  // background.paramOn(frameBackground);

  // frameBackground->Draw();
  // frameBackground->Write();
  // canvasBackground.Write();

  // canvasBackground.Write();

  // // Fix the background parameter
  // expCoeff.setConstant(kTRUE);

  // Build the model adding the exponential background
  RooRealVar fSig("fSig", "signal fraction", 0.4, 0., 1.);
  RooAddPdf model("model", "model", RooArgList(signal, background), fSig);

  model.fitTo(histo, "hs");

  TCanvas canvas("canvas", "canvas", 1000, 800);
  canvas.cd();
  canvas.Draw();
  RooPlot* frame = x.frame();
  histo.plotOn(frame);
  model.plotOn(frame);
  model.paramOn(frame);

  model.plotOn(frame, Components("exponential"), LineStyle(kDashed));


  int binXmin = inputHisto->FindBin( xMin );
  int binXmax = inputHisto->FindBin( xMax );
  Double_t fullIntegral = inputHisto->Integral(binXmin, binXmax);
  std::cout << "Events in fit interval = " << fullIntegral << std::endl;

  RooAbsReal * backgroundFullIntegral = background.createIntegral(x); 
  Double_t backgroundFullIntegralValue = backgroundFullIntegral->getVal();
  std::cout << "backgroundFullIntegralValue = " << backgroundFullIntegralValue << std::endl;

  // Compute the integral and write the signal and background events in +-2.5sigma
  Double_t sbMin = peak.getVal() - 2.5*sigma.getVal();
  Double_t sbMax = peak.getVal() + 2.5*sigma.getVal();
  int minBin = inputHisto->FindBin( sbMin );
  int maxBin = inputHisto->FindBin( sbMax );
  std::cout << "minBin("<<peak.getVal() - 2.5*sigma.getVal()<<") = " << minBin << std::endl;
  std::cout << "maxBin("<<peak.getVal() + 2.5*sigma.getVal()<<") = " << maxBin << std::endl;

  Double_t integral = inputHisto->Integral(minBin, maxBin);

  x.setRange("small", sbMin, sbMax);
  RooAbsReal * backgroundSmallIntegral = background.createIntegral(x, "small"); 
  Double_t backgroundSmallIntegralValue = backgroundSmallIntegral->getVal();
  std::cout << "backgroundSmallIntegralValue = " << backgroundSmallIntegralValue << std::endl;

  double fSigSmall = 1 - fullIntegral*(1-fSig.getVal())/integral*backgroundSmallIntegralValue/backgroundFullIntegralValue;


  std::cout << "Events in ["<< xMin << ","<<xMax<<"]:" << std::endl;
  std::cout << "Signal events = " << (fSig.getVal())*fullIntegral << std::endl;
  std::cout << "Background events = " << (1-fSig.getVal())*fullIntegral << std::endl;

  std::cout << "Events in peak +- 2.5sigma:" << std::endl;
  std::cout << "Signal events = " << fSigSmall*integral << std::endl;
  std::cout << "Background events = " << (1-fSigSmall)*integral << std::endl;

  frame->Draw();
  frame->Write();
  canvas.Write();
  outputFile.Write();
};
