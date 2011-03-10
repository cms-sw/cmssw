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

void macro()
{
  TFile inputFile("INPUTFILENAME", "READ");
  TH1F * inputHisto = (TH1F*)inputFile.Get("hRecBestResAllEvents_Mass");
  inputHisto->Rebin(4);

  double xMin = 2.6;
  double xMax = 3.4;
  RooRealVar x("x", "Mass (GeV)", xMin, xMax);

  RooDataHist histo("dh", "dh", x, Import(*inputHisto));

  // Build the CB
  RooRealVar peak("peak","peak", 3.095, 2.5, 4.);
  RooRealVar sigma("sigma","sigma", 0.04, 0.001, 0.1);
  RooRealVar alpha("alpha", "alpha", 1., 0., 10.);
  RooRealVar n("n", "n", 1., 0., 100.);
  RooCBShape signal("signal", "signal", x, peak, sigma, alpha, n);

  // Build the exponential background pdf
  RooRealVar expCoeff("expCoeff", "exponential coefficient", -1., -10., 10.);
  RooExponential background("exponential", "exponential", x, expCoeff);

  TFile outputFile("FITRESULT", "RECREATE"); 

  // Compute the total number of events in the fit window
  int binXmin = inputHisto->FindBin( xMin );
  int binXmax = inputHisto->FindBin( xMax );
  Double_t fullIntegral = inputHisto->Integral(binXmin, binXmax);

  // Build the model adding the exponential background
  RooRealVar nSig("nSig", "signal fraction", 3000, 0., fullIntegral);
  RooRealVar nBkg("nBkg", "background fraction", 2000, 0., fullIntegral);

  Double_t sbSigma = 3.;
  // Do this twice: the first time to find the peak and sigma and the second time
  // to get the correct yelds in peak+-3sigma
  for( int i=0; i<2; ++i ) {
    // Compute the integral and write the signal and background events in +-sbSigma
    Double_t sbMin = 2.6;
    Double_t sbMax = 3.4;
    if( i == 1 ) {
      sbMin = 3.09697 - 0.2;
      sbMax = 3.09697 + 0.2;
    }
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

      float S = nSig.getVal();
      float sS = nSig.getError();
      float B = nBkg.getVal();
      float sB = nBkg.getError();
      std::cout << "Signal fraction parameter = " << S << "+-" << sS << std::endl;
      std::cout << "Background fraction parameter = " << B << "+-" << sB << std::endl;
      std::cout << "Fraction parameter = " << S/(S+B) << "+-"
		<< sqrt(pow(B/(pow(S+B,2)),2)*sS*sS + pow(S/(pow(S+B,2)),2)*sB*sB) << std::endl;
      std::cout << "Exponential parameter = " << expCoeff.getVal() << "+-" << expCoeff.getError() << std::endl;
    }
  }
};
