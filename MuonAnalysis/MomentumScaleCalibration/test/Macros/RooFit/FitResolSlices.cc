#ifndef FitResolSlices_cc
#define FitResolSlices_cc

#include "/home/destroyar/Desktop/MuScleFit/RooFitTest/Macros/FitSlices.cc"
#include "TFile.h"
#include "TH1F.h"
#include "TROOT.h"

/**
 * This class can be used to fit the X slices of a TH1 histogram using RooFit.
 * It uses the FitXslices class to do the fitting.
 */
class FitResolSlices : public FitSlices
{
 public:
  void fit(const TString & inputFileName = "0_MuScleFit.root", const TString & outputFileName = "ResolCheck_0.root",
	   const TString & signalType = "gaussian",
	   const double & xMean = 0., const double & xMin = -1., const double & xMax = 1.,
	   const double & sigma = 0.03, const double & sigmaMin = 0., const double & sigmaMax = 0.1,
	   const TString & histoBaseName = "hResolPtGenVSMu_ResoVS", const TString & histoBaseTitle = "ResolPtVs",
	   TFile * externalOutputFile = 0)
  {
    gROOT->SetBatch(kTRUE);

    TFile * inputFile = new TFile(inputFileName, "READ");

    TFile * outputFile = externalOutputFile;
    if( outputFile == 0 ) {
      outputFile = new TFile(outputFileName, "RECREATE");
    }
    outputFile->mkdir(histoBaseName);
    outputFile->cd(histoBaseName);
    TDirectory * dir = (TDirectory*)outputFile->Get(histoBaseName);

    fitSlice(histoBaseName+"Pt", histoBaseTitle+"Pt",
    	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
    	     signalType, "",
    	     inputFile, dir);

    fitSlice(histoBaseName+"Eta", histoBaseTitle+"Eta",
	     xMean, xMin, xMax, sigma, sigmaMin, sigmaMax,
	     signalType, "",
	     inputFile, dir);

    if( externalOutputFile == 0 ) {
      outputFile->Write();
      outputFile->Close();
    }
  }
};

#endif
