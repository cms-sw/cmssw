#ifndef FitXslices_cc
#define FitXslices_cc

#include <iostream>
#include <sstream>
#include "TColor.h"
#include "TH2F.h"
#include "TH3F.h"
#include "TDirectory.h"
#include "TROOT.h"
#include "TStyle.h"
#include "FitWithRooFit.cc"

/**
 * This class performs the following actions: <br>
 * - take a TH2* as input and fit slices of it in the x coordinate (one slice per bin) <br>
 * - store the result of each slice fit and draw them on a canvas <br>
 * - draw plots of each parameter result with the corresponding error <br>
 * It uses RooFit for the fitting.
 */

class FitXslices {
public:
  FitXslices() {
    /// Initialize suitable parameter values for Z fitting
    /// N.B.: mean and sigma of gaussian, as well as mean and range of the peak are defined in the CompareBiasZValidation class

    /// Fraction of signal
    fitter_.initFsig(0.9, 0., 1.);

    /// CB
    fitter_.initMean2(0., -20., 20.);
    fitter_.mean2()->setConstant(kTRUE);
    fitter_.initSigma(1.5, 0.5, 3.);
    fitter_.initSigma2(1.2, 0., 5.);
    fitter_.initAlpha(1.5, 0.05, 10.);
    fitter_.initN(1, 0.01, 100.);

    // BW Fix the gamma for the Z
    fitter_.initGamma(2.4952, 0., 10.);
    fitter_.gamma()->setConstant(kTRUE);  // This can be overriden as well.

    fitter_.initGaussFrac(0.5, 0., 1.);

    /// Exponential with pol2 argument
    fitter_.initExpCoeffA0(-1., -10., 10.);
    fitter_.initExpCoeffA1(0., -10., 10.);
    fitter_.initExpCoeffA2(0., -2., 2.);

    /// Polynomial
    fitter_.initA0(0., -10., 10.);
    fitter_.initA1(0., -10., 10.);
    fitter_.initA2(0., -10., 10.);
    fitter_.initA3(0., -10., 10.);
    fitter_.initA4(0., -10., 10.);
    fitter_.initA5(0., -10., 10.);
    fitter_.initA6(0., -10., 10.);

    fitter_.useChi2_ = false;
  }

  FitWithRooFit* fitter() { return (&fitter_); }

  //  void fitSlices( std::map<unsigned int, TH1*> & slices, const double & xMin, const double & xMax, const TString & signalType, const TString & backgroundType, const bool twoD ){    }

  void operator()(TH2* histo,
                  const double& xMin,
                  const double& xMax,
                  const TString& signalType,
                  const TString& backgroundType,
                  unsigned int rebinY) {
    // Create and move in a subdir
    gDirectory->mkdir("allHistos");
    gDirectory->cd("allHistos");

    gStyle->SetPalette(1);
    // Loop on all X bins, project on Y and fit the resulting TH1
    TString name = histo->GetName();
    unsigned int binsX = histo->GetNbinsX();

    // The canvas for the results of the fit (the mean values for the gaussians +- errors)
    TCanvas* meanCanvas = new TCanvas("meanCanvas", "meanCanvas", 1000, 800);
    TH1D* meanHisto =
        new TH1D("meanHisto", "meanHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* sigmaCanvas = new TCanvas("sigmaCanvas", "sigmaCanvas", 1000, 800);
    TH1D* sigmaHisto =
        new TH1D("sigmaHisto", "sigmaHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* backgroundCanvas = new TCanvas("backgroundCanvas", "backgroundCanvas", 1000, 800);
    TH1D* backgroundHisto = new TH1D(
        "backgroundHisto", "backgroundHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* backgroundCanvas2 = new TCanvas("backgroundCanvas2", "backgroundCanvas2", 1000, 800);
    TH1D* backgroundHisto2 =
        new TH1D("backgroundHisto2", "exp a1", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* backgroundCanvas3 = new TCanvas("backgroundCanvas3", "backgroundCanvas3", 1000, 800);
    TH1D* backgroundHisto3 =
        new TH1D("backgroundHisto3", "exp a2", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* signalFractionCanvas = new TCanvas("signalFractionCanvas", "signalFractionCanvas", 1000, 800);
    TH1D* signalFractionHisto = new TH1D(
        "signalFractionHisto", "signalFractionHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas* probChi2Canvas = new TCanvas("probChi2Canvas", "probChi2Canvas", 1000, 800);
    TH1D* probChi2Histo =
        new TH1D("probChi2Histo", "probChi2Histo", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    // Store all the non-empty slices
    std::map<unsigned int, TH1*> slices;
    for (unsigned int x = 1; x <= binsX; ++x) {
      std::stringstream ss;
      ss << x;
      TH1* sliceHisto = histo->ProjectionY((std::string{name.Data()} + ss.str()).c_str(), x, x);
      if (sliceHisto->GetEntries() != 0) {
        // std::cout << "filling for x = " << x << endl;
        slices.insert(std::make_pair(x, sliceHisto));
        sliceHisto->Rebin(rebinY);
      }
    }

    // Create the canvas for all the fits
    TCanvas* fitsCanvas = new TCanvas("fitsCanvas", "fits canvas", 1000, 800);
    // cout << "slices.size = " << slices.size() << endl;
    unsigned int x = sqrt(slices.size());
    unsigned int y = x;
    if (x * y < slices.size()) {
      x += 1;
      y += 1;
    }
    fitsCanvas->Divide(x, y);

    // Loop on the saved slices and fit
    std::map<unsigned int, TH1*>::iterator it = slices.begin();
    unsigned int i = 1;
    for (; it != slices.end(); ++it, ++i) {
      fitsCanvas->cd(i);
      fitter_.fit(it->second, signalType, backgroundType, xMin, xMax);
      //      fitsCanvas->GetPad(i)->SetLogy();
      // FIXME: prob(chi2) needs to be computed properly inside FitWithRooFit.cc
      // probChi2Histo->SetBinContent(it->first, mean->getVal());

      RooRealVar* mean = fitter_.mean();

      meanHisto->SetBinContent(it->first, mean->getVal());
      meanHisto->SetBinError(it->first, mean->getError());

      RooRealVar* sigma = fitter_.sigma();
      sigmaHisto->SetBinContent(it->first, sigma->getVal());
      sigmaHisto->SetBinError(it->first, sigma->getError());

      std::cout << "backgroundType = " << backgroundType << std::endl;
      if (backgroundType == "exponential") {
        RooRealVar* expCoeff = fitter_.expCoeffa1();
        backgroundHisto->SetBinContent(it->first, expCoeff->getVal());
        backgroundHisto->SetBinError(it->first, expCoeff->getError());
      } else if (backgroundType == "exponentialpol") {
        RooRealVar* expCoeffa0 = fitter_.expCoeffa0();
        backgroundHisto->SetBinContent(it->first, expCoeffa0->getVal());
        backgroundHisto->SetBinError(it->first, expCoeffa0->getError());

        RooRealVar* expCoeffa1 = fitter_.expCoeffa1();
        backgroundHisto2->SetBinContent(it->first, expCoeffa1->getVal());
        backgroundHisto2->SetBinError(it->first, expCoeffa1->getError());

        RooRealVar* expCoeffa2 = fitter_.expCoeffa2();
        backgroundHisto3->SetBinContent(it->first, expCoeffa2->getVal());
        backgroundHisto3->SetBinError(it->first, expCoeffa2->getError());
      }

      else if (backgroundType == "linear") {
        RooRealVar* linearTerm = fitter_.a1();
        backgroundHisto->SetBinContent(it->first, linearTerm->getVal());
        backgroundHisto->SetBinError(it->first, linearTerm->getError());

        RooRealVar* constant = fitter_.a0();
        backgroundHisto2->SetBinContent(it->first, constant->getVal());
        backgroundHisto2->SetBinError(it->first, constant->getError());
      }
      RooRealVar* fsig = fitter_.fsig();
      signalFractionHisto->SetBinContent(it->first, fsig->getVal());
      signalFractionHisto->SetBinError(it->first, fsig->getError());
    }
    // Go back to the main dir before saving the canvases
    gDirectory->GetMotherDir()->cd();
    meanCanvas->cd();
    meanHisto->Draw();
    sigmaCanvas->cd();
    sigmaHisto->Draw();
    backgroundCanvas->cd();
    backgroundHisto->Draw();
    if (backgroundType == "linear") {
      backgroundCanvas2->cd();
      backgroundHisto2->Draw();
    }
    if (backgroundType == "exponentialpol") {
      backgroundCanvas2->cd();
      backgroundHisto2->Draw();
      backgroundCanvas3->cd();
      backgroundHisto3->Draw();
    }
    signalFractionCanvas->cd();
    signalFractionHisto->Draw();
    probChi2Canvas->cd();
    probChi2Histo->Draw();

    fitsCanvas->Write();
    meanCanvas->Write();
    sigmaCanvas->Write();
    backgroundCanvas->Write();
    signalFractionCanvas->Write();
    if (backgroundType == "linear") {
      backgroundCanvas2->Write();
    }
    probChi2Canvas->Write();

    fitsCanvas->Close();
    probChi2Canvas->Close();
    signalFractionCanvas->Close();
    backgroundCanvas3->Close();
    backgroundCanvas2->Close();
    backgroundCanvas->Close();
    sigmaCanvas->Close();
    meanCanvas->Close();
  }

  void operator()(TH3* histo,
                  const double& xMin,
                  const double& xMax,
                  const TString& signalType,
                  const TString& backgroundType,
                  unsigned int rebinZ) {
    // Create and move in a subdir
    gDirectory->mkdir("allHistos");
    gDirectory->cd("allHistos");

    // Loop on all X bins, project on Y and fit the resulting TH2
    TString name = histo->GetName();
    unsigned int binsX = histo->GetNbinsX();
    unsigned int binsY = histo->GetNbinsY();

    // std::cout<< "number of bins in x --> "<<binsX<<std::endl;
    // std::cout<< "number of bins in y --> "<<binsY<<std::endl;

    // The canvas for the results of the fit (the mean values for the gaussians +- errors)
    TCanvas* meanCanvas = new TCanvas("meanCanvas", "meanCanvas", 1000, 800);
    TH2D* meanHisto = new TH2D("meanHisto",
                               "meanHisto",
                               binsX,
                               histo->GetXaxis()->GetXmin(),
                               histo->GetXaxis()->GetXmax(),
                               binsY,
                               histo->GetYaxis()->GetXmin(),
                               histo->GetYaxis()->GetXmax());

    TCanvas* errorMeanCanvas = new TCanvas("errorMeanCanvas", "errorMeanCanvas", 1000, 800);
    TH2D* errorMeanHisto = new TH2D("errorMeanHisto",
                                    "errorMeanHisto",
                                    binsX,
                                    histo->GetXaxis()->GetXmin(),
                                    histo->GetXaxis()->GetXmax(),
                                    binsY,
                                    histo->GetYaxis()->GetXmin(),
                                    histo->GetYaxis()->GetXmax());

    TCanvas* sigmaCanvas = new TCanvas("sigmaCanvas", "sigmaCanvas", 1000, 800);
    TH2D* sigmaHisto = new TH2D("sigmaHisto",
                                "sigmaHisto",
                                binsX,
                                histo->GetXaxis()->GetXmin(),
                                histo->GetXaxis()->GetXmax(),
                                binsY,
                                histo->GetYaxis()->GetXmin(),
                                histo->GetYaxis()->GetXmax());

    TCanvas* backgroundCanvas = new TCanvas("backgroundCanvas", "backgroundCanvas", 1000, 800);
    TH2D* backgroundHisto = new TH2D("backgroundHisto",
                                     "backgroundHisto",
                                     binsX,
                                     histo->GetXaxis()->GetXmin(),
                                     histo->GetXaxis()->GetXmax(),
                                     binsY,
                                     histo->GetYaxis()->GetXmin(),
                                     histo->GetYaxis()->GetXmax());
    TCanvas* backgroundCanvas2 = new TCanvas("backgroundCanvas2", "backgroundCanvas2", 1000, 800);
    TH2D* backgroundHisto2 = new TH2D("backgroundHisto2",
                                      "a1",
                                      binsX,
                                      histo->GetXaxis()->GetXmin(),
                                      histo->GetXaxis()->GetXmax(),
                                      binsY,
                                      histo->GetYaxis()->GetXmin(),
                                      histo->GetYaxis()->GetXmax());
    TCanvas* backgroundCanvas3 = new TCanvas("backgroundCanvas3", "backgroundCanvas3", 1000, 800);
    TH2D* backgroundHisto3 = new TH2D("backgroundHisto3",
                                      "a2",
                                      binsX,
                                      histo->GetXaxis()->GetXmin(),
                                      histo->GetXaxis()->GetXmax(),
                                      binsY,
                                      histo->GetYaxis()->GetXmin(),
                                      histo->GetYaxis()->GetXmax());

    TCanvas* signalFractionCanvas = new TCanvas("signalFractionCanvas", "signalFractionCanvas", 1000, 800);
    TH2D* signalFractionHisto = new TH2D("signalFractionHisto",
                                         "signalFractionHisto",
                                         binsX,
                                         histo->GetXaxis()->GetXmin(),
                                         histo->GetXaxis()->GetXmax(),
                                         binsY,
                                         histo->GetYaxis()->GetXmin(),
                                         histo->GetYaxis()->GetXmax());

    // Store all the non-empty slices
    std::map<unsigned int, TH1*> slices;
    for (unsigned int x = 1; x <= binsX; ++x) {
      for (unsigned int y = 1; y <= binsY; ++y) {
        std::stringstream ss;
        ss << x << "_" << y;
        TH1* sliceHisto = histo->ProjectionZ((std::string{name.Data()} + ss.str()).c_str(), x, x, y, y);
        if (sliceHisto->GetEntries() != 0) {
          sliceHisto->Rebin(rebinZ);
          // std::cout << "filling for x = " << x << endl;
          slices.insert(std::make_pair(x + (binsX + 1) * y, sliceHisto));
        }
      }
    }

    // Create the canvas for all the fits
    TCanvas* fitsCanvas = new TCanvas("fitsCanvas", "canvas of all fits", 1000, 800);
    // cout << "slices.size = " << slices.size() << endl;
    unsigned int x = sqrt(slices.size());
    unsigned int y = x;
    if (x * y < slices.size()) {
      x += 1;
      y += 1;
    }
    fitsCanvas->Divide(x, y);

    // Loop on the saved slices and fit
    std::map<unsigned int, TH1*>::iterator it = slices.begin();
    unsigned int i = 1;
    for (; it != slices.end(); ++it, ++i) {
      fitsCanvas->cd(i);

      fitter_.fit(it->second, signalType, backgroundType, xMin, xMax);

      RooRealVar* mean = fitter_.mean();
      meanHisto->SetBinContent(it->first % (binsX + 1), int(it->first / (binsX + 1)), mean->getVal());
      errorMeanHisto->SetBinContent(it->first % (binsX + 1), int(it->first / (binsX + 1)), mean->getError());
      //      meanHisto->SetBinError(it->first%binsX, int(it->first/binsX), mean->getError());
      //std::cout<<"int i -->"<<i<<std::endl;
      //std::cout<< " it->first%(binsX+1) --> "<<it->first%(binsX+1)<<std::endl;
      //std::cout<< " it->first/(binsX+1) --> "<<int(it->first/(binsX+1))<<std::endl;

      RooRealVar* sigma = fitter_.sigma();
      sigmaHisto->SetBinContent(it->first % binsX, int(it->first / binsX), sigma->getVal());
      sigmaHisto->SetBinError(it->first % binsX, int(it->first / binsX), sigma->getError());

      std::cout << "backgroundType = " << backgroundType << std::endl;
      if (backgroundType == "exponential") {
        RooRealVar* expCoeff = fitter_.expCoeffa1();
        backgroundHisto->SetBinContent(it->first % binsX, int(it->first / binsX), expCoeff->getVal());
        backgroundHisto->SetBinError(it->first % binsX, int(it->first / binsX), expCoeff->getError());
      } else if (backgroundType == "exponentialpol") {
        RooRealVar* expCoeffa0 = fitter_.expCoeffa0();
        backgroundHisto->SetBinContent(it->first % binsX, int(it->first / binsX), expCoeffa0->getVal());
        backgroundHisto->SetBinError(it->first % binsX, int(it->first / binsX), expCoeffa0->getError());

        RooRealVar* expCoeffa1 = fitter_.expCoeffa1();
        backgroundHisto2->SetBinContent(it->first % binsX, int(it->first / binsX), expCoeffa1->getVal());
        backgroundHisto2->SetBinError(it->first % binsX, int(it->first / binsX), expCoeffa1->getError());

        RooRealVar* expCoeffa2 = fitter_.expCoeffa2();
        backgroundHisto3->SetBinContent(it->first % binsX, int(it->first / binsX), expCoeffa2->getVal());
        backgroundHisto3->SetBinError(it->first % binsX, int(it->first / binsX), expCoeffa2->getError());
      } else if (backgroundType == "linear") {
        RooRealVar* linearTerm = fitter_.a1();
        backgroundHisto->SetBinContent(it->first % binsX, int(it->first / binsX), linearTerm->getVal());
        backgroundHisto->SetBinError(it->first % binsX, int(it->first / binsX), linearTerm->getError());

        RooRealVar* constant = fitter_.a0();
        backgroundHisto2->SetBinContent(it->first % binsX, int(it->first / binsX), constant->getVal());
        backgroundHisto2->SetBinError(it->first % binsX, int(it->first / binsX), constant->getError());
      }

      RooRealVar* fsig = fitter_.fsig();
      signalFractionHisto->SetBinContent(it->first % binsX, int(it->first / binsX), fsig->getVal());
      signalFractionHisto->SetBinError(it->first % binsX, int(it->first / binsX), fsig->getError());
    }
    // Go back to the main dir before saving the canvases
    gDirectory->GetMotherDir()->cd();

    meanCanvas->cd();
    meanHisto->GetXaxis()->SetRangeUser(-3.14, 3.14);
    meanHisto->GetYaxis()->SetRangeUser(-2.5, 2.5);
    meanHisto->GetXaxis()->SetTitle("#phi (rad)");
    meanHisto->GetYaxis()->SetTitle("#eta");
    meanHisto->Draw("COLZ");

    sigmaCanvas->cd();
    sigmaHisto->GetXaxis()->SetRangeUser(-3.14, 3.14);
    sigmaHisto->GetYaxis()->SetRangeUser(-2.5, 2.5);
    sigmaHisto->GetXaxis()->SetTitle("#phi (rad)");
    sigmaHisto->GetYaxis()->SetTitle("#eta");
    sigmaHisto->Draw("COLZ");

    backgroundCanvas->cd();
    backgroundHisto->Draw("COLZ");
    if (backgroundType == "linear") {
      backgroundCanvas2->cd();
      backgroundHisto2->Draw("COLZ");
    }
    signalFractionCanvas->cd();
    signalFractionHisto->Draw("COLZ");

    fitsCanvas->Write();
    meanCanvas->Write();
    sigmaCanvas->Write();
    backgroundCanvas->Write();
    signalFractionCanvas->Write();
    if (backgroundType == "linear") {
      backgroundCanvas2->Write();
    }

    fitsCanvas->Close();
    signalFractionCanvas->Close();
    backgroundCanvas3->Close();
    backgroundCanvas2->Close();
    backgroundCanvas->Close();
    sigmaCanvas->Close();
    errorMeanCanvas->Close();
    meanCanvas->Close();
  }

protected:
  struct Parameter {
    Parameter(const double& inputValue, const double& inputError) : value(inputValue), error(inputError) {}

    double value;
    double error;
  };

  FitWithRooFit fitter_;
};

#endif
