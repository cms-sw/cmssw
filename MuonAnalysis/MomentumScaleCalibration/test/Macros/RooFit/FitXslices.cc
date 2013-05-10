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

class FitXslices
{
public:
  FitXslices()
  {
    //fitter_.initMean( 9.46, 9.1, 9.7 );
    fitter_.initMean2( 0., -20., 20. );
    fitter_.mean2()->setConstant(kTRUE);
    // fitter_.initSigma( 2.3, 0., 10. );
    fitter_.initSigma( 2., 0., 10. );
    fitter_.initSigma2( 0.2, 0.0001, 2. );

    // Fix the gamma for the Z
    fitter_.initGamma( 2.4952, 0., 10. );
    fitter_.gamma()->setConstant(kTRUE);

    fitter_.initGaussFrac( 0.5, 0., 1. );
    fitter_.initExpCoeff( -0.1, -5., 0. );
    fitter_.initFsig(0.9, 0., 1.);

    fitter_.initConstant(500., 0, 10000);
    fitter_.initLinearTerm(0, -10., 10);
    fitter_.initParabolicTerm(0, -1., 1.);
    fitter_.initQuarticTerm(0, -0.1, 0.1);

    fitter_.initAlpha(3., 0., 30.);
    fitter_.initN(2, 0., 50.);

    fitter_.useChi2_ = false;
  }

  FitWithRooFit * fitter()
  {
    return( &fitter_ );
  }

  //  void fitSlices( std::map<unsigned int, TH1*> & slices, const double & xMin, const double & xMax, const TString & signalType, const TString & backgroundType, const bool twoD ){    }

  void operator()(TH2 * histo, const double & xMin, const double & xMax, const TString & signalType, const TString & backgroundType)
  {
    // Create and move in a subdir
    gDirectory->mkdir("allHistos");
    gDirectory->cd("allHistos");

/*
    const Int_t NRGBs = 3;
    const Int_t NCont = 255;
    Double_t stops[NRGBs] = {0.00,   0.50,   1.00};
    Double_t red[NRGBs]   = {1.00,   0.50,   0.00};
    Double_t green[NRGBs] = {0.00,   0.00,   0.00};
    Double_t blue[NRGBs]  = {0.00,   0.50,   1.00};    
    TColor::CreateGradientColorTable(NRGBs, stops, red, green, blue, NCont);
    gStyle->SetNumberContours(NCont);
*/
    gStyle->SetPalette(1);
  // Loop on all X bins, project on Y and fit the resulting TH1
    TString name = histo->GetName();
    unsigned int binsX = histo->GetNbinsX();

    // The canvas for the results of the fit (the mean values for the gaussians +- errors)
    TCanvas * meanCanvas = new TCanvas("meanCanvas", "meanCanvas", 1000, 800);
    TH1D * meanHisto = new TH1D("meanHisto", "meanHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas * sigmaCanvas = new TCanvas("sigmaCanvas", "sigmaCanvas", 1000, 800);
    TH1D * sigmaHisto = new TH1D("sigmaHisto", "sigmaHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas * backgroundCanvas = new TCanvas("backgroundCanvas", "backgroundCanvas", 1000, 800);
    TH1D * backgroundHisto = new TH1D("backgroundHisto", "backgroundHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());
    TCanvas * backgroundCanvas2 = new TCanvas("backgroundCanvas2", "backgroundCanvas2", 1000, 800);
    TH1D * backgroundHisto2 = new TH1D("backgroundHisto2", "constant", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas * signalFractionCanvas = new TCanvas("signalFractionCanvas", "signalFractionCanvas", 1000, 800);
    TH1D * signalFractionHisto = new TH1D("signalFractionHisto", "signalFractionHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());

    TCanvas * probChi2Canvas = new TCanvas("probChi2Canvas", "probChi2Canvas", 1000, 800);
    TH1D * probChi2Histo = new TH1D("probChi2Histo", "probChi2Histo", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax());


    // Store all the non-empty slices
    std::map< unsigned int, TH1 *> slices;
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

    // Loop on the saved slices and fit
    std::map<unsigned int, TH1*>::iterator it = slices.begin();
    unsigned int i=1;
    for( ; it != slices.end(); ++it, ++i ) {
      fitsCanvas->cd(i);
      fitter_.fit(it->second, signalType, backgroundType, xMin, xMax);
      //      fitsCanvas->GetPad(i)->SetLogy();

      //      probChi2Histo->SetBinContent(it->first, mean->getVal());

      RooRealVar * mean = fitter_.mean();
      
      meanHisto->SetBinContent(it->first, mean->getVal());
      meanHisto->SetBinError(it->first, mean->getError());

      RooRealVar * sigma = fitter_.sigma();
      sigmaHisto->SetBinContent(it->first, sigma->getVal());
      sigmaHisto->SetBinError(it->first, sigma->getError());
      
      std::cout << "backgroundType = " << backgroundType << std::endl;
      if( backgroundType == "exponential" ) {
	RooRealVar * expCoeff = fitter_.expCoeff();
	
	backgroundHisto->SetBinContent(it->first, expCoeff->getVal());
	backgroundHisto->SetBinError(it->first, expCoeff->getError());
	
      }
      else if( backgroundType == "linear" ) {
	RooRealVar * linearTerm = fitter_.linearTerm();
	
	backgroundHisto->SetBinContent(it->first, linearTerm->getVal());
	backgroundHisto->SetBinError(it->first, linearTerm->getError());
	
	RooRealVar * constant = fitter_.constant();
	backgroundHisto2->SetBinContent(it->first, constant->getVal());
	backgroundHisto2->SetBinError(it->first, constant->getError());  
      }
      RooRealVar * fsig = fitter_.fsig(); 
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
    if( backgroundType == "linear" ) {
      backgroundCanvas2->cd();
      backgroundHisto2->Draw();
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
    if( backgroundType == "linear" ) {
      backgroundCanvas2->Write();
    }
    probChi2Canvas->Write();
    // fitSlices(slices, xMin, xMax, signalType, backgroundType, false); ///DEVO PASSARGLI xMin, xMax e il resto....

  }

  void operator()(TH3 * histo, const double & xMin, const double & xMax, const TString & signalType, const TString & backgroundType, unsigned int rebinZ)
  {
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
    TCanvas * meanCanvas = new TCanvas("meanCanvas", "meanCanvas", 1000, 800);
    TH2D * meanHisto = new TH2D("meanHisto", "meanHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());

    TCanvas * errorMeanCanvas = new TCanvas("errorMeanCanvas", "errorMeanCanvas", 1000, 800);
    TH2D * errorMeanHisto = new TH2D("errorMeanHisto", "errorMeanHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());

    TCanvas * sigmaCanvas = new TCanvas("sigmaCanvas", "sigmaCanvas", 1000, 800);
    TH2D * sigmaHisto = new TH2D("sigmaHisto", "sigmaHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(),binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());

    TCanvas * backgroundCanvas = new TCanvas("backgroundCanvas", "backgroundCanvas", 1000, 800);
    TH2D * backgroundHisto = new TH2D("backgroundHisto", "backgroundHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());
    TCanvas * backgroundCanvas2 = new TCanvas("backgroundCanvas2", "backgroundCanvas2", 1000, 800);
    TH2D * backgroundHisto2 = new TH2D("backgroundHisto2", "constant", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());

    TCanvas * signalFractionCanvas = new TCanvas("signalFractionCanvas", "signalFractionCanvas", 1000, 800);
    TH2D * signalFractionHisto = new TH2D("signalFractionHisto", "signalFractionHisto", binsX, histo->GetXaxis()->GetXmin(), histo->GetXaxis()->GetXmax(), binsY, histo->GetYaxis()->GetXmin(), histo->GetYaxis()->GetXmax());

    // Store all the non-empty slices
    std::map<unsigned int, TH1 *> slices;
    for( unsigned int x=1; x<=binsX; ++x ) {
      for( unsigned int y=1; y<=binsY; ++y ) {
	std::stringstream ss;
	ss << x << "_" << y;
	TH1 * sliceHisto = histo->ProjectionZ(name+ss.str(), x, x, y, y);
	if( sliceHisto->GetEntries() != 0 ) {
	  sliceHisto->Rebin(rebinZ);
	  // std::cout << "filling for x = " << x << endl;
	  slices.insert(std::make_pair(x+(binsX+1)*y, sliceHisto));
	}
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

    // Loop on the saved slices and fit
    std::map<unsigned int, TH1*>::iterator it = slices.begin();
    unsigned int i=1;
    for( ; it != slices.end(); ++it, ++i ) {
      fitsCanvas->cd(i);

      fitter_.fit(it->second, signalType, backgroundType, xMin, xMax);
      
      RooRealVar * mean = fitter_.mean();
      meanHisto->SetBinContent(it->first%(binsX+1), int(it->first/(binsX+1)), mean->getVal());
      errorMeanHisto->SetBinContent(it->first%(binsX+1), int(it->first/(binsX+1)), mean->getError());
      //      meanHisto->SetBinError(it->first%binsX, int(it->first/binsX), mean->getError());
      //std::cout<<"int i -->"<<i<<std::endl;
      //std::cout<< " it->first%(binsX+1) --> "<<it->first%(binsX+1)<<std::endl;
      //std::cout<< " it->first/(binsX+1) --> "<<int(it->first/(binsX+1))<<std::endl;    

      RooRealVar * sigma = fitter_.sigma();    
      sigmaHisto->SetBinContent(it->first%binsX, int(it->first/binsX), sigma->getVal());
      sigmaHisto->SetBinError(it->first%binsX, int(it->first/binsX), sigma->getError());
      
      std::cout << "backgroundType = " << backgroundType << std::endl;
      if( backgroundType == "exponential" ) {
	RooRealVar * expCoeff = fitter_.expCoeff();
	backgroundHisto->SetBinContent(it->first%binsX, int(it->first/binsX), expCoeff->getVal());
	backgroundHisto->SetBinError(it->first%binsX, int(it->first/binsX), expCoeff->getError());
      }
      else if( backgroundType == "linear" ) {
	
	RooRealVar * linearTerm = fitter_.linearTerm();
	backgroundHisto->SetBinContent(it->first%binsX, int(it->first/binsX), linearTerm->getVal());
	backgroundHisto->SetBinError(it->first%binsX, int(it->first/binsX), linearTerm->getError());
		
	RooRealVar * constant = fitter_.constant();
	backgroundHisto2->SetBinContent(it->first%binsX, int(it->first/binsX), constant->getVal());
	backgroundHisto2->SetBinError(it->first%binsX, int(it->first/binsX), constant->getError());
 
      }
   
      RooRealVar * fsig = fitter_.fsig();    
      signalFractionHisto->SetBinContent(it->first%binsX, int(it->first/binsX), fsig->getVal());
      signalFractionHisto->SetBinError(it->first%binsX, int(it->first/binsX), fsig->getError());

    }
    // Go back to the main dir before saving the canvases
    gDirectory->GetMotherDir()->cd();

    meanCanvas->cd();
    meanHisto->GetXaxis()->SetRangeUser(-3.14,3.14);
    meanHisto->GetYaxis()->SetRangeUser(-2.5,2.5);
    meanHisto->GetXaxis()->SetTitle("#phi (rad)");
    meanHisto->GetYaxis()->SetTitle("#eta");
    meanHisto->Draw("COLZ");

    sigmaCanvas->cd();
    sigmaHisto->GetXaxis()->SetRangeUser(-3.14,3.14);    
    sigmaHisto->GetYaxis()->SetRangeUser(-2.5,2.5);
    sigmaHisto->GetXaxis()->SetTitle("#phi (rad)");
    sigmaHisto->GetYaxis()->SetTitle("#eta");
    sigmaHisto->Draw("COLZ");

    backgroundCanvas->cd();
    backgroundHisto->Draw("COLZ");
    if( backgroundType == "linear" ) {
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
    if( backgroundType == "linear" ) {
      backgroundCanvas2->Write();
    }
    //  fitSlices(slices, xMin, xMax, signalType, backgroundType, true);

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

#endif
