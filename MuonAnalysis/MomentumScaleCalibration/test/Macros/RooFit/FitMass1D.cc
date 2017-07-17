#ifndef FitMass1D_cc
#define FitMass1D_cc

#include <iostream>
#include <sstream>
#include "TFile.h"
#include "TH2F.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "FitWithRooFit.cc"

/**
 * Fit the mass distribution with the given function using RooFit.
 */

class FitMass1D
{
public:
  FitMass1D() : rebinX(1)
  {
  };

  void fit(const TString & inputFileName, const TString & outputFileName, const TString & outputFileOption, 
	   const double & xMin, const double & xMax, 
	   const TString & signalType = "doubleGaussian", const TString & backgroundType = "exponential")
  {
    //    TFile * inputFile1 = new TFile(fileNum+"_MuScleFit.root", "READ");
    TFile* inputFile = TFile::Open(inputFileName, "READ" );
    TH1F * histo = (TH1F*)inputFile->Get("hRecBestResAllEvents_Mass");
    histo->Rebin(rebinX);

    TFile * outputFile = new TFile(outputFileName,outputFileOption);
    outputFile->cd();

    fitter_.fit(histo, signalType, backgroundType, xMin, xMax);

    outputFile->Write();
    outputFile->Close();
  }

  FitWithRooFit * fitter()
  {
    return( &fitter_ );
  }
protected:
  int rebinX;
  FitWithRooFit fitter_;
};

#endif
