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
#include "RooExtendPdf.h"
using namespace RooFit;

void GaussianFitOnData_Psi2S()
{
  TFile inputFile("0_MuScleFit.root", "READ");
  TH1F * inputHisto = (TH1F*)inputFile.Get("hRecBestResAllEvents_Mass");
  inputHisto->Rebin(10);

  double xMin = 3.3;
  double xMax = 4.4;
  RooRealVar x("x", "Mass (GeV)", xMin, xMax);

  RooDataHist histo("dh", "dh", x, Import(*inputHisto));

  // Build the Gaussian
  RooRealVar peak("peak","peak", 3.695, 3.6, 3.8);
  RooRealVar sigma("sigma","sigma", 0.06, 0.001, 0.1);
  // RooRealVar sigma("sigma","sigma", 0.044);
  RooGaussian signal("signal", "signal", x, peak, sigma);

  // Build the exponential background pdf
  RooRealVar expCoeff("expCoeff", "exponential coefficient", -1., -10., 10.);
  RooExponential background("exponential", "exponential", x, expCoeff);

  TFile outputFile("gaussianFitOnData_Psi2S.root", "RECREATE"); 

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


  // // Prefit to determine peak and sigma
  // RooRealVar fSig("fSig", "signal fraction", 0.4, 0., 1.);
  // RooAddPdf prefitModel("prefitModel", "prefitModel", RooArgList(signal, background), fSig);
  // prefitModel.fitTo(histo, "hs");
  // prefitModel.fitTo(histo, "hs");
  // std::cout << "prefit peak = " << peak.getVal() << "+-" << peak.getError() << std::endl; 
  // std::cout << "prefit sigma = " << sigma.getVal() << "+-" << sigma.getError() << std::endl; 

  // Now we have peak and sigma to the actual values, build an extended model to compute
  // also the yelds of signal and background in the peak+-3sigma window

  int binXmin = inputHisto->FindBin( xMin );
  int binXmax = inputHisto->FindBin( xMax );
  Double_t fullIntegral = inputHisto->Integral(binXmin, binXmax);
  std::cout << "Events in fit interval = " << fullIntegral << std::endl;

  // Build the model adding the exponential background
  RooRealVar nSig("nSig", "signal fraction", 30, 0., fullIntegral);
  RooRealVar nBkg("nBkg", "background fraction", 20, 0., fullIntegral);

  Double_t sbSigma = 3.;
  // Do this twice: the first time to find the peak and sigma and the second time
  // to get the correct yelds in peak+-3sigma
  for( int i=0; i<2; ++i ) {
    // Compute the integral and write the signal and background events in +-sbSigma
    Double_t sbMin = peak.getVal() - sbSigma*sigma.getVal();
    Double_t sbMax = peak.getVal() + sbSigma*sigma.getVal();
    x.setRange("small", sbMin, sbMax);
    RooExtendPdf eSig("eSig", "eSig", signal, nSig, "small");
    RooExtendPdf eBkg("eBkg", "eBkg", background, nBkg, "small");

    // RooAddPdf model("model", "model", RooArgList(signal, background), RooArgList(nSig, nBkg));
    RooAddPdf model("model", "model", RooArgList(eSig, eBkg));
    // model.fitTo(histo, "hs");
    model.fitTo(histo);

    std::cout << "peak("<<i<<") = " << peak.getVal() << "+-" << peak.getError() << std::endl;
    std::cout << "sigma("<<i<<") = " << sigma.getVal() << "+-" << sigma.getError() << std::endl;
    if( i==1 ) {
      TCanvas canvas("canvas", "canvas", 1000, 800);
      canvas.cd();
      canvas.Draw();
      RooPlot* frame = x.frame();
      histo.plotOn(frame);
      model.plotOn(frame);
      model.paramOn(frame);

      model.plotOn(frame, Components("exponential"), LineStyle(kDashed));

      frame->Draw();
      frame->Write();
      canvas.Write();
      outputFile.Write();
    }
  }


  // RooAbsReal * backgroundFullIntegral = background.createIntegral(x); 
  // Double_t backgroundFullIntegralValue = backgroundFullIntegral->getVal();
  // std::cout << "backgroundFullIntegralValue = " << backgroundFullIntegralValue << std::endl;

  // int minBin = inputHisto->FindBin( sbMin );
  // int maxBin = inputHisto->FindBin( sbMax );
  // std::cout << "minBin("<<peak.getVal() - sbSigma*sigma.getVal()<<") = " << minBin << std::endl;
  // std::cout << "maxBin("<<peak.getVal() + sbSigma*sigma.getVal()<<") = " << maxBin << std::endl;

  // Double_t integral = inputHisto->Integral(minBin, maxBin);



  // RooAbsReal* fracSigRange = sig.createIntegral(x, x, "small");
  // Double_t nSigSmall = nSig.getVal()*fracSigRange->getVal();

  // std::cout << "esig = " << esig.getVal() << "+-" << std::endl;


  // RooAbsReal * backgroundSmallIntegral = background.createIntegral(x, "small"); 
  // Double_t backgroundSmallIntegralValue = backgroundSmallIntegral->getVal();
  // std::cout << "backgroundSmallIntegralValue = " << backgroundSmallIntegralValue << std::endl;


  // double fSigSmall = 1 - fullIntegral*(1-fSig.getVal())/integral*backgroundSmallIntegralValue/backgroundFullIntegralValue;

  // std::cout << "Events in ["<< xMin << ","<<xMax<<"] = " << fullIntegral << std::endl;
  // std::cout << "Signal events = " << (fSig.getVal())*fullIntegral << std::endl;
  // std::cout << "Background events = " << (1-fSig.getVal())*fullIntegral << std::endl;
  // std::cout << "S/B = " << fSig.getVal()/(1-fSig.getVal()) << std::endl;
  // std::cout << std::endl;
  // std::cout << "Events in peak +- "<<sbSigma<<"sigma = " << integral << std::endl;
  // std::cout << "Signal events = " << fSigSmall*integral << std::endl;
  // std::cout << "Background events = " << (1-fSigSmall)*integral << std::endl;
  // std::cout << "S/B = " << fSigSmall/(1-fSigSmall) << std::endl;

};
