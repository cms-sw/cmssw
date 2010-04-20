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
  FitMass()
  {
    fitter_.initMean( 3.1, 2.9, 3.3 );
    fitter_.initSigma( 0.03, 0., 0.1 );
    fitter_.initSigma2( 0.1, 0., 1. );
    fitter_.initGaussFrac( 0.5, 0., 1. );
    fitter_.initExpCoeff( -1., -10., 0. );
    fitter_.initFsig(0.5, 0., 1.);
  };

  void fit(const double & xMin, const double & xMax)
  {
    TFile * inputFile1 = new TFile("0_MuScleFit.root", "READ");
    TH1F * histo1 = (TH1F*)inputFile1->Get("hRecBestRes_Mass");
    histo1->Rebin(4);

    TFile * outputFile = new TFile("MassFit.root", "RECREATE");
    outputFile->cd();

    fitter_.fit(histo1, "doubleGaussian", "exponential", xMin, xMax);
    // fitter_.fit(it->second, "gaussian", "exponential", xMin, xMax);

    outputFile->Write();
    outputFile->Close();
  }
protected:
  FitWithRooFit fitter_;
};
