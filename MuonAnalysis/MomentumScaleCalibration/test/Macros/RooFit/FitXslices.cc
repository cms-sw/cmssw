#include <iostream>
#include <sstream>
#include "TH2F.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "FitWithRooFit.cc"

/**
 * This class performs the following actions: <br>
 * - take a TH2* as input and fit slices of it in the x coordinate (one slice per bin) <br>
 * - store the result of each slice fit and draw them on a canvas <br>
 * - draw plots of each parameter result with the corresponding error <br>
 * It uses RooFit for the fitting.
 */

class FitXslices
{
public:
  FitXslices()
  {
    fitter_.initMean( 3.1, 2.9, 3.3 );
    fitter_.initSigma( 0.03, 0., 0.1 );
    fitter_.initSigma2( 0.1, 0., 1. );
    fitter_.initGaussFrac( 0.5, 0., 1. );
    fitter_.initExpCoeff( -1., -10., 0. );
    fitter_.initFsig(0.5, 0., 1.);
  };

  FitWithRooFit * fitter()
  {
    return( &fitter_ );
  }

  void operator()(TH2 * histo, const double & xMin, const double & xMax, const TString & signalType, const TString & backgroundType)
  {
    // Create and move in a subdir
    gDirectory->mkdir("allHistos");
    gDirectory->cd("allHistos");

    std::vector<Parameter> means;

    // Loop on all X bins, project on Y and fit the resulting TH1
    TString name = histo->GetName();
    unsigned int binsX = histo->GetNbinsX();

    // Store all the non-empty slices
    std::map<unsigned int, TH1 *> slices;
    for( unsigned int x=1; x<=binsX; ++x ) {
      std::stringstream ss;
      ss << x;
      TH1 * sliceHisto = histo->ProjectionY(name+ss.str(), x, x);
      if( sliceHisto->GetEntries() != 0 ) {
	// std::cout << "filling for x = " << x << endl;
	slices.insert(std::make_pair(x, sliceHisto));
      }
    }
    // Create the canvas for all the fits
    TCanvas * fitsCanvas = new TCanvas("fitsCanvas", "fits canvas", 1000, 800);
    // cout << "slices.size = " << slices.size() << endl;
    unsigned int x = sqrt(slices.size());
    unsigned int y = x;
    if( x*y < slices.size() ) {
      x += 1;
      y += 1;
    }
    fitsCanvas->Divide(x, y);
    // The canvas for the results of the fit (the mean values for the gaussians +- errors)
    TCanvas * meanCanvas = new TCanvas("meanCanvas", "meanCanvas", 1000, 800);
    TH1D * meanHisto = new TH1D("meanHisto", "meanHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
    TCanvas * backgroundCanvas = new TCanvas("backgroundCanvas", "backgroundCanvas", 1000, 800);
    TH1D * backgroundHisto = new TH1D("backgroundHisto", "backgroundHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
    TCanvas * signalFractionCanvas = new TCanvas("signalFractionCanvas", "signalFractionCanvas", 1000, 800);
    TH1D * signalFractionHisto = new TH1D("signalFractionHisto", "signalFractionHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    // Loop on the saved slices and fit
    std::map<unsigned int, TH1 *>::iterator it = slices.begin();
    unsigned int i=1;
    for( ; it != slices.end(); ++it, ++i ) {
      fitsCanvas->cd(i);
      fitter_.fit(it->second, signalType, backgroundType, xMin, xMax);
      // fitter_.fit(it->second, "doubleGaussian", "exponential", xMin, xMax);
      // fitter_.fit(it->second, "gaussian", "exponential", xMin, xMax);
      // fitter_.fit(it->second, "gaussian", "exponential", xMin, xMax);
      RooRealVar * mean = fitter_.mean();
      meanHisto->SetBinContent(it->first, mean->getVal());
      meanHisto->SetBinError(it->first, mean->getError());
      RooRealVar * expCoeff = fitter_.expCoeff();
      backgroundHisto->SetBinContent(it->first, expCoeff->getVal());
      backgroundHisto->SetBinError(it->first, expCoeff->getError());
      RooRealVar * fsig = fitter_.fsig();
      signalFractionHisto->SetBinContent(it->first, fsig->getVal());
      signalFractionHisto->SetBinError(it->first, fsig->getError());
    }
    // Go back to the main dir before saving the canvases
    gDirectory->GetMotherDir()->cd();
    meanCanvas->cd();
    meanHisto->Draw();
    backgroundCanvas->cd();
    backgroundHisto->Draw();
    signalFractionCanvas->cd();
    signalFractionHisto->Draw();

    fitsCanvas->Write();
    meanCanvas->Write();
    backgroundCanvas->Write();
    signalFractionCanvas->Write();
  }
protected:
  struct Parameter
  {
    Parameter(const double & inputValue, const double & inputError) :
      value(inputValue), error(inputError)
    {}

    double value;
    double error;
  };

  FitWithRooFit fitter_;
};
