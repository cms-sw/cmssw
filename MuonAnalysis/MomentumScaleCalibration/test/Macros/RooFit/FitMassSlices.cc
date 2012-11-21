#ifndef FitMassSlices_cc
#define FitMassSlices_cc

#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitSlices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * This class can be used to fit the X slices of a TH1 histogram using RooFit.
 * It uses the FitXslices class to do the fitting.
 */
class FitMassSlices : public FitSlices
{
 public:
  void fit(const TString & inputFileName = "0_MuScleFit.root", const TString & outputFileName = "BiasCheck_0.root",
	   const TString & signalType = "gaussian", const TString & backgroundType = "exponential",
	   const double & xMean = 3.1, const double & xMin = 3., const double & xMax = 3.2,
	   const double & sigma = 0.03, const double & sigmaMin = 0., const double & sigmaMax = 0.1,
	   TDirectory * externalDir = 0,
	   const TString & histoBaseName = "hRecBestResVSMu", const TString & histoBaseTitle = "MassVs")
  {
    gROOT->SetBatch(kTRUE);

    TFile * inputFile = new TFile(inputFileName, "READ");

    TFile * outputFile = 0;
    TDirectory * dir = externalDir;
    if( dir == 0 ) {
      outputFile = new TFile(outputFileName, "RECREATE");
      dir = outputFile->GetDirectory("");
    }

    fitSlice(histoBaseName+"_MassVSPt", histoBaseTitle+"Pt",
    	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
    	     signalType, backgroundType,
    	     inputFile, dir);

    fitSlice(histoBaseName+"_MassVSEta", histoBaseTitle+"Eta",
	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
	     signalType, backgroundType,
	     inputFile, dir);

    fitSlice(histoBaseName+"_MassVSPhiPlus", histoBaseTitle+"PhiPlus",
    	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
    	     signalType, backgroundType,
    	     inputFile, dir);

    fitSlice(histoBaseName+"_MassVSPhiMinus", histoBaseTitle+"PhiMinus",
    	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
    	     signalType, backgroundType,
    	     inputFile, dir);

    if( outputFile != 0 ) {
      outputFile->Write();
      outputFile->Close();
    }
  }
};

#endif
