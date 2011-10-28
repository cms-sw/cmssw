#ifndef FitSlices_cc
#define FitSlices_cc

#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitXslices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * This class can be used to fit the X slices of a TH1 histogram using RooFit.
 * It uses the FitXslices class to do the fitting.
 */
class FitSlices
{
public:
  FitSlices() :
    rebinX(2), rebinY(2), sigma2(0.1), sigma2Min(0.), sigma2Max(10.)
  {}

  // virtual void fit(const TString & inputFileName = "0_MuScleFit.root", const TString & outputFileName = "BiasCheck_0.root",
  // 		   const TString & signalType = "gaussian", const TString & backgroundType = "exponential",
  // 		   const double & xMean = 3.1, const double & xMin = 3., const double & xMax = 3.2,
  // 		   const double & sigma = 0.03, const double & sigmaMin = 0., const double & sigmaMax = 0.1,
  // 		   const TString & histoBaseName = "hRecBestResVSMu", const TString & histoBaseTitle = "MassVs") = 0;

  void fitSlice(const TString & histoName, const TString & dirName,
		const double & xMean, const double & xMin, const double & xMax,
		const double & sigma, const double & sigmaMin, const double & sigmaMax,
		const TString & signalType, const TString & backgroundType,
		TFile * inputFile, TDirectory * outputFile)
  {
    FitXslices fitXslices;
    fitXslices.fitter()->initMean( xMean, xMin, xMax );
    fitXslices.fitter()->initSigma( sigma, sigmaMin, sigmaMax );
    fitXslices.fitter()->initSigma2( sigma2, sigma2Min, sigma2Max );

    fitXslices.fitter()->initAlpha(1.6, 0., 10.);
    fitXslices.fitter()->initN(2, 0., 10.);
    fitXslices.fitter()->initFGCB(0.4, 0., 1.);

    std::cout << "Fit slices: initialization complete" << std::endl;
    TH2 * histoPt = (TH2*)inputFile->FindObjectAny(histoName);
    // TH2 * histoPt = 0;
    // inputFile->GetObject(histoName, histoPt);
    histoPt->RebinX(rebinX);
    histoPt->RebinY(rebinY);
    outputFile->mkdir(dirName);
    outputFile->cd(dirName);
    fitXslices(histoPt, xMin, xMax, signalType, backgroundType);
  }
  unsigned int rebinX;
  unsigned int rebinY;
  double sigma2, sigma2Min, sigma2Max;
};

#endif
