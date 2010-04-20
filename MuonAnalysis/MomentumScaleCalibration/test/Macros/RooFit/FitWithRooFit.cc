#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooVoigtian.h"
#include "RooExponential.h"
#include "TCanvas.h"
#include "RooPlot.h"
#include "TTree.h"
#include "TH1D.h"
#include "TRandom.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooGaussModel.h"
#include "RooAddModel.h"

/**
 * This macro allows to use RooFit to perform a fit on a TH1 histogram. <br>
 * The currently implemented functions are: <br>
 * - signal: <br>
 * -- gaussian <br>
 * -- double gaussian <br>
 * -- voigtian <br>
 * - background: <br>
 * -- exponential <br>
 * It is possible to select any combination of signal and background. <br>
 * The fit() method receives the TH1, two strings specifying the signal and background type
 * and the min and max x of the histogram over which to perform the fit. <br>
 * The variables of the fit must be initialized separately. For example when doing a gaussian+exponential
 * fit the initMean, initSigma, initFSig and initExpCoeff methods must be used to create and initialize
 * the corresponding variables. <br>
 * The methods names after the variables return the fit result.
 */

using namespace RooFit;

namespace
{
  typedef std::pair<RooRealVar, RooDataHist*> rooPair;
}

class FitWithRooFit
{
public:

  FitWithRooFit() :
    mean_(0), sigma_(0), gamma_(0), sigma2_(0), gaussFrac_(0), expCoeff_(0), fsig_(0)
  {
  }

  // Import TH1 histogram into a RooDataHist
  rooPair importTH1(TH1 * histo, const double & inputXmin, const double & inputXmax)
  {
    double xMin = inputXmin;
    double xMax = inputXmax;
    if( (xMin == xMax) && xMin == 0 ) {
      xMin = histo->GetXaxis()->GetXmin();
      xMax = histo->GetXaxis()->GetXmax();
    }
    // Declare observable x
    RooRealVar x("x", "x", xMin, xMax);
    // Create a binned dataset that imports contents of TH1 and associates its contents to observable 'x'
    return( std::make_pair(x, new RooDataHist("dh","dh",x,Import(*histo))) );
  }

  // Plot and fit a RooDataHist
  void fit(TH1 * histo, const TString signalType, const TString backgroundType, const double & xMin = 0., const double & xMax = 0.)
  {
    rooPair imported = importTH1(histo, xMin, xMax);
    RooRealVar x(imported.first);
    RooDataHist * dh = imported.second;

    // Make plot of binned dataset showing Poisson error bars (RooFit default)
    RooPlot* frame = x.frame(Title("Imported TH1 with Poisson error bars")) ;
    frame->SetName(TString(histo->GetName())+"_frame");
    dh->plotOn(frame) ; 

    // Signal
    // ------
    RooAbsPdf * signal = 0;
    if( signalType == "gaussian" ) {
      // Fit a Gaussian p.d.f to the data
      if( (mean_ == 0) || (sigma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean and sigma" << std::endl;
	exit(1);
      }
      signal = new RooGaussian("gauss","gauss",x,*mean_,*sigma_);
    }
    else if( signalType == "doubleGaussian" ) {
      // Fit with double gaussian
      if( (mean_ == 0) || (sigma_ == 0) || (sigma2_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and sigma2" << std::endl;
	exit(1);
      }
      RooGaussModel * gaussModel = new RooGaussModel("gaussModel","gaussModel",x,*mean_,*sigma_);
      RooGaussModel * gaussModel2 = new RooGaussModel("gaussModel2","gaussModel2",x,*mean_,*sigma2_);
      signal = new RooAddModel("doubleGaussian", "double gaussian", RooArgList(*gaussModel, *gaussModel2), *gaussFrac_);
    }
    else if( signalType == "voigtian" ) {
      // Fit a Voigtian
      if( (mean_ == 0) || (sigma_ == 0) || (gamma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and gamma" << std::endl;
	exit(1);
      }
      signal = new RooVoigtian("voigt", "voigt", x, *mean_, *gamma_, *sigma_);
    }
    else if( signalType != "" ) {
      std::cout << "Unknown signal function: " << signalType << ". Signal will not be in the model" << std::endl;
    }

    // Background
    // ----------
    RooAbsPdf * background = 0;
    if( backgroundType == "exponential" ) {
      // Add an exponential for the background
      if( (expCoeff_ == 0) || (fsig_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeff and fsig" << std::endl;
	exit(1);
      }
      background = new RooExponential("expon", "expon", x, *expCoeff_);
    }

    // Build the model to fit
    // ----------------------

    // Parameters
    // RooRealVar fsig("fsig", "signal fraction", 0.5, 0., 1.);

    RooAbsPdf * model = 0;
    if( (signal != 0) && (background != 0) ) {
      // Combine signal and background pdfs
      model = new RooAddPdf("model", "model", RooArgList(*signal, *background), *fsig_);
    }
    else if( signal != 0 ) {
      model = signal;
    }
    else if( background != 0 ) {
      model = background;
    }
    else {
      std::cout << "Nothing to fit" << endl;
      exit(0);
    }
    // Fit the composite model
    model->fitTo(*dh);
    model->plotOn(frame);
    model->paramOn(frame, Label("fit result"), Format("NEU", AutoPrecision(2)));

    // P l o t   a n d   f i t   a   R o o D a t a H i s t   w i t h   i n t e r n a l   e r r o r s
    // ---------------------------------------------------------------------------------------------

    // If histogram has custom error (i.e. its contents is does not originate from a Poisson process
    // but e.g. is a sum of weighted events) you can data with symmetric 'sum-of-weights' error instead
    // (same error bars as shown by ROOT)
    RooPlot* frame2 = x.frame(Title("Imported TH1 with internal errors")) ;
    dh->plotOn(frame2,DataError(RooAbsData::SumW2)) ; 
    model->plotOn(frame2);
    model->paramOn(frame2, Label("fit result"), Format("NEU", AutoPrecision(2)));

    // Please note that error bars shown (Poisson or SumW2) are for visualization only, the are NOT used
    // in a maximum likelihood fit
    //
    // A (binned) ML fit will ALWAYS assume the Poisson error interpretation of data (the mathematical definition 
    // of likelihood does not take any external definition of errors). Data with non-unit weights can only be correctly
    // fitted with a chi^2 fit (see rf602_chi2fit.C) 

    // Draw all frames on a canvas
    // if( canvas == 0 ) {
    //   canvas = new TCanvas("rf102_dataimport","rf102_dataimport",800,800);
    //   canvas->cd(1);
    // }
    // canvas->Divide(2,1);
    // canvas->cd(1) ; frame->Draw();
    // canvas->cd(2) ; frame2->Draw();

    frame2->Draw();
  }

  // Initialization methods for all the parameters
  void initMean(const double & value, const double & min, const double & max, const TString & name = "mean", const TString & title = "mean")
  {
    if( mean_ != 0 ) delete mean_;
    mean_ = new RooRealVar(name, title, value, min, max);
  }

  void initSigma(const double & value, const double & min, const double & max, const TString & name = "sigma", const TString & title = "sigma")
  {
    if( sigma_ != 0 ) delete sigma_;
    sigma_ = new RooRealVar(name, title, value, min, max);
  }

  void initGamma(const double & value, const double & min, const double & max, const TString & name = "gamma", const TString & title = "gamma")
  {
    if( gamma_ != 0 ) delete gamma_;
    gamma_ = new RooRealVar(name, title, value, min, max);
  }

  void initSigma2(const double & value, const double & min, const double & max, const TString & name = "sigma2", const TString & title = "sigma2")
  {
    if( sigma2_ != 0 ) delete sigma2_;
    sigma2_ = new RooRealVar(name, title, value, min, max);
  }

  void initGaussFrac(const double & value, const double & min, const double & max, const TString & name = "sigma2", const TString & title = "sigma2")
  {
    if( gaussFrac_ != 0 ) delete gaussFrac_;
    gaussFrac_ = new RooRealVar(name, title, value, min, max);
  }

  void initExpCoeff(const double & value, const double & min, const double & max, const TString & name = "expCoeff", const TString & title = "expCoeff")
  {
    if( expCoeff_ != 0 ) delete expCoeff_;
    expCoeff_ = new RooRealVar(name, title, value, min, max);
  }
  void initFsig(const double & value, const double & min, const double & max, const TString & name = "fsig", const TString & title = "signal fraction")
  {
    std::cout << "Created fsig" << std::endl;
    if( fsig_ != 0 ) delete fsig_;
    fsig_ = new RooRealVar(name, title, value, min, max);
  }

  inline RooRealVar * mean()
  {
    return mean_;
  }

  inline RooRealVar * sigma()
  {
    return sigma_;
  }

  inline RooRealVar * expCoeff()
  {
    return expCoeff_;
  }

  inline RooRealVar * fsig()
  {
    return fsig_;
  }

protected:

  // Declare all variables
  RooRealVar * mean_;// ("mean","mean", 3, 2.4, 3.8);
  RooRealVar * sigma_;// ("sigma","sigma", 0.03, 0., 10);
  RooRealVar * gamma_;// ("sigma","sigma", 0.0001, 0., 0.001);
  RooRealVar * sigma2_;// ("sigma2","sigma2", 1., 0., 10.);
  RooRealVar * gaussFrac_;// ("gaussFrac","gaussFrac", 0.5, 0., 1.);
  RooRealVar * expCoeff_;// ("expCoeff","exponential coefficient", -3, -10., 10);
  RooRealVar * fsig_;
};
