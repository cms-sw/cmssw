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

class FitMass
{
public:
  FitMass() : rebinX(1)
  {
    fitter_.initMean( 3.1, 2.9, 3.3 );
    fitter_.initSigma(  0.03, 0., 0.1 );
    fitter_.initSigma2( 0.1,  0., 1. );

    fitter_.initGamma( 2.4952, 0., 10.);
    fitter_.gamma()->setConstant(kTRUE);

    fitter_.initGaussFrac( 0.5, 0., 1. );
    fitter_.initExpCoeffa1( -1., -10., 0. );
    fitter_.initFsig(0.5, 0., 1.);
  };

  void fit(const double & xMin, const double & xMax, const TString & fileNum = "0",
	   const TString & signalType = "doubleGaussian", const TString & backgroundType = "exponential")
  {
    TFile * inputFile1 = new TFile(fileNum+"_MuScleFit.root", "READ");
    TH1F * histo1 = (TH1F*)inputFile1->Get("hRecBestResAllEvents_Mass");
    histo1->Rebin(rebinX);

    TFile * outputFile = new TFile("MassFit_"+fileNum+".root", "RECREATE");
    outputFile->cd();

    fitter_.fit(histo1, signalType, backgroundType, xMin, xMax);
    // fitter_.fit(histo1, "doubleGaussian", "exponential", xMin, xMax);
    // fitter_.fit(it->second, "gaussian", "exponential", xMin, xMax);

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
