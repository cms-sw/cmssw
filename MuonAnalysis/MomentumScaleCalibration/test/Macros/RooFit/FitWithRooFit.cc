#ifndef FitWithRooFit_cc
#define FitWithRooFit_cc

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
#include "RooGenericPdf.h"
#include "RooGaussModel.h"
#include "RooAddModel.h"
#include "RooPolynomial.h"
#include "RooCBShape.h"
#include "RooChi2Var.h"
#include "RooMinuit.h"
#include "RooBreitWigner.h"
#include "RooFFTConvPdf.h"


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
    useChi2_(false), mean_(0), mean2_(0), sigma_(0), sigma2_(0), gamma_(0), gaussFrac_(0), expCoeff_(0), fsig_(0),
    constant_(0), linearTerm_(0), parabolicTerm_(0), quarticTerm_(0), alpha_(0), n_(0), fGCB_(0)
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

  // Plot and fit a RooDataHist fitting signal and background
  void fit(TH1 * histo, const TString signalType, const TString backgroundType, const double & xMin = 0., const double & xMax = 0., bool sumW2Error = false)
  {
    rooPair imported = importTH1(histo, xMin, xMax);
    RooRealVar x(imported.first);
    RooDataHist * dh = imported.second;

    // Make plot of binned dataset showing Poisson error bars (RooFit default)
    RooPlot* frame = x.frame(Title("Imported TH1 with Poisson error bars")) ;
    frame->SetName(TString(histo->GetName())+"_frame");
    dh->plotOn(frame);

    // Build the composite model
    RooAbsPdf * model = buildModel(&x, signalType, backgroundType);

    RooChi2Var chi2("chi2", "chi2", *model, *dh, DataError(RooAbsData::SumW2));

    // Fit the composite model
    // -----------------------
    // Fit with likelihood
    if( !useChi2_ ) {
      if( sumW2Error ) {
	model->fitTo(*dh, Save(), SumW2Error(kTRUE));
      }
      else {
	model->fitTo(*dh);
      }
    }
    // Fit with chi^2
    else {
      std::cout << "FITTING WITH CHI^2" << std::endl;
      RooMinuit m(chi2);
      m.migrad();
      m.hesse();
      // RooFitResult* r_chi2_wgt = m.save();
    }
    model->plotOn(frame);
    model->plotOn(frame, Components(backgroundType), LineStyle(kDotted), LineColor(kRed));
    model->paramOn(frame, Label("fit result"), Format("NEU", AutoPrecision(2)));

    // P l o t   a n d   f i t   a   R o o D a t a H i s t   w i t h   i n t e r n a l   e r r o r s
    // ---------------------------------------------------------------------------------------------

    // If histogram has custom error (i.e. its contents is does not originate from a Poisson process
    // but e.g. is a sum of weighted events) you can data with symmetric 'sum-of-weights' error instead
    // (same error bars as shown by ROOT)
    RooPlot* frame2 = x.frame(Title("Imported TH1 with internal errors")) ;
    dh->plotOn(frame2,DataError(RooAbsData::SumW2)) ; 
    model->plotOn(frame2);
    model->plotOn(frame2, Components(backgroundType), LineColor(kRed));
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

  void initMean2(const double & value, const double & min, const double & max, const TString & name = "mean2", const TString & title = "mean2")
  {
    if( mean2_ != 0 ) delete mean2_;
    mean2_ = new RooRealVar(name, title, value, min, max);
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

  void initGaussFrac(const double & value, const double & min, const double & max, const TString & name = "GaussFrac", const TString & title = "GaussFrac")
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
    if( fsig_ != 0 ) delete fsig_;
    fsig_ = new RooRealVar(name, title, value, min, max);
  }

  void initConstant(const double & value, const double & min, const double & max, const TString & name = "constant", const TString & title = "constant")
  {
    if( constant_ != 0 ) delete constant_;
    constant_ = new RooRealVar(name, title, value, min, max);
  }

  void initLinearTerm(const double & value, const double & min, const double & max, const TString & name = "linearTerm", const TString & title = "linearTerm")
  {
    if( linearTerm_ != 0 ) delete linearTerm_;
    linearTerm_ = new RooRealVar(name, title, value, min, max);
  }

  void initParabolicTerm(const double & value, const double & min, const double & max, const TString & name = "parabolicTerm", const TString & title = "parabolicTerm")
  {
    if( parabolicTerm_ != 0 ) delete parabolicTerm_;
    parabolicTerm_ = new RooRealVar(name, title, value, min, max);
  }

  void initQuarticTerm(const double & value, const double & min, const double & max, const TString & name = "quarticTerm", const TString & title = "quarticTerm")
  {
    if( quarticTerm_ != 0 ) delete quarticTerm_;
    quarticTerm_ = new RooRealVar(name, title, value, min, max);
  }

  void initAlpha(const double & value, const double & min, const double & max, const TString & name = "alpha", const TString & title = "alpha")
  {
    if( alpha_ != 0 ) delete alpha_;
    alpha_ = new RooRealVar(name, title, value, min, max);
  }

  void initN(const double & value, const double & min, const double & max, const TString & name = "n", const TString & title = "n")
  {
    if( n_ != 0 ) delete n_;
    n_ = new RooRealVar(name, title, value, min, max);
  }

  void initFGCB(const double & value, const double & min, const double & max, const TString & name = "fGCB", const TString & title = "fGCB")
  {
    if( fGCB_ != 0 ) delete fGCB_;
    fGCB_ = new RooRealVar(name, title, value, min, max);
  }

  inline RooRealVar * mean()
  {
    return mean_;
  }

  inline RooRealVar * mean2()
  {
    return mean2_;
  }

  inline RooRealVar * sigma()
  {
    return sigma_;
  }

  inline RooRealVar * sigma2()
  {
    return sigma2_;
  }

  inline RooRealVar * gamma()
  {
    return gamma_;
  }

  inline RooRealVar * expCoeff()
  {
    return expCoeff_;
  }

  inline RooRealVar * fsig()
  {
    return fsig_;
  }

  inline RooRealVar * constant()
  {
    return constant_;
  }

  inline RooRealVar * linearTerm()
  {
    return linearTerm_;
  }

  inline RooRealVar * parabolicTerm()
  {
    return parabolicTerm_;
  }

  inline RooRealVar * quarticTerm()
  {
    return quarticTerm_;
  }

  inline RooRealVar * alpha()
  {
    return alpha_;
  }

  inline RooRealVar * n()
  {
    return n_;
  }

  inline RooRealVar * fGCB()
  {
    return fGCB_;
  }

  /// Build the model for the specified signal type
  RooAbsPdf * buildSignalModel(RooRealVar * x, const TString & signalType)
  {
    RooAbsPdf * signal = 0;
    if( signalType == "gaussian" ) {
      // Fit a Gaussian p.d.f to the data
      if( (mean_ == 0) || (sigma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean and sigma" << std::endl;
	exit(1);
      }
      signal = new RooGaussian("gauss","gauss",*x,*mean_,*sigma_);
    }
    else if( signalType == "doubleGaussian" ) {
      // Fit with double gaussian
      if( (mean_ == 0) || (sigma_ == 0) || (sigma2_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and sigma2" << std::endl;
	exit(1);
      }
      RooGaussModel * gaussModel = new RooGaussModel("gaussModel","gaussModel",*x,*mean_,*sigma_);
      RooGaussModel * gaussModel2 = new RooGaussModel("gaussModel2","gaussModel2",*x,*mean_,*sigma2_);
      signal = new RooAddModel("doubleGaussian", "double gaussian", RooArgList(*gaussModel, *gaussModel2), *gaussFrac_);
    }
    else if( signalType == "breitWigner" ) {
      // Fit a Breit-Wigner
      if( (mean_ == 0) || (gamma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma" << std::endl;
	exit(1);
      }
      signal = new RooBreitWigner("breiWign", "breitWign", *x, *mean_, *gamma_);
    }
    else if( signalType == "voigtian" ) {
      // Fit a Voigtian
      if( (mean_ == 0) || (sigma_ == 0) || (gamma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma and gamma" << std::endl;
	exit(1);
      }
      signal = new RooVoigtian("voigt", "voigt", *x, *mean_, *gamma_, *sigma_);
    }
    else if( signalType == "crystalBall" ) {
      // Fit a CrystalBall
      if( (mean_ == 0) || (sigma_ == 0) || (alpha_ == 0) || (n_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma, alpha and n" << std::endl;
	exit(1);
      }
      signal = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma_, *alpha_, *n_);
    }
    else if( signalType == "breitWignerTimesCB" ) {
      // Fit a Breit Wigner convoluted with a CrystalBall
      if( (mean_ == 0) || (mean2_==0 )|| (sigma_ == 0) || (gamma_==0) || (alpha_ == 0) || (n_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, sigma, gamma, alpha and n" << std::endl;
	exit(1);
      }
      RooAbsPdf* bw = new RooBreitWigner("breiWigner", "breitWigner", *x, *mean_, *gamma_);
      RooAbsPdf* cb = new RooCBShape("crystalBall", "crystalBall", *x, *mean2_, *sigma_, *alpha_, *n_);
      signal = new RooFFTConvPdf("breitWignerTimesCB","breitWignerTimesCB",*x, *bw, *cb);
    }
    else if( signalType == "relBreitWignerTimesCB" ) {
      // Fit a relativistic Breit Wigner convoluted with a CrystalBall
      if( (mean_ == 0) || (mean2_==0 )|| (sigma_ == 0) || (gamma_==0) || (alpha_ == 0) || (n_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, sigma, gamma, alpha and n" << std::endl;
	exit(1);
      }
      RooGenericPdf *bw = new RooGenericPdf("Relativistic Breit-Wigner","RBW","@0/(pow(@0*@0 - @1*@1,2) + @2*@2*@0*@0*@0*@0/(@1*@1))", RooArgList(*x,*mean_,*gamma_));
      RooAbsPdf* cb = new RooCBShape("crystalBall", "crystalBall", *x, *mean2_, *sigma_, *alpha_, *n_);
      signal = new RooFFTConvPdf("breitWignerTimesCB","breitWignerTimesCB",*x, *bw, *cb);
    }
    else if( signalType == "gaussianPlusCrystalBall" ) {
      // Fit a Gaussian + CrystalBall with the same mean
      if( (mean_ == 0) || (sigma_ == 0) || (alpha_ == 0) || (n_ == 0) || (sigma2_ == 0) || (fGCB_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, sigma, sigma2, alpha, n and fGCB" << std::endl;
	exit(1);
      }
      RooAbsPdf * tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma_, *alpha_, *n_);
      RooAbsPdf * tempGaussian = new RooGaussian("gauss", "gauss", *x, *mean_, *sigma2_);

      signal = new RooAddPdf("gaussianPlusCrystalBall", "gaussianPlusCrystalBall", RooArgList(*tempCB, *tempGaussian), *fGCB_);
    }
    else if( signalType == "voigtianPlusCrystalBall" ) {
      // Fit a Voigtian + CrystalBall with the same mean
      if( (mean_ == 0) || (sigma_ == 0) || (gamma_ == 0) || (alpha_ == 0) || (n_ == 0) || (sigma2_ == 0) || (fGCB_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, gamma, sigma, sigma2, alpha, n and fGCB" << std::endl;
	exit(1);
      }
      RooAbsPdf * tempVoigt = new RooVoigtian("voigt", "voigt", *x, *mean_, *gamma_, *sigma_);
      RooAbsPdf * tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma2_, *alpha_, *n_);

      signal = new RooAddPdf("voigtianPlusCrystalBall", "voigtianPlusCrystalBall", RooArgList(*tempCB, *tempVoigt), *fGCB_);
    }
    else if( signalType == "breitWignerPlusCrystalBall" ) {
      // Fit a Breit-Wigner + CrystalBall with the same mean
      if( (mean_ == 0) || (gamma_ == 0) || (alpha_ == 0) || (n_ == 0) || (sigma2_ == 0) || (fGCB_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, gamma, sigma, alpha, n and fGCB" << std::endl;
	exit(1);
      }
      RooAbsPdf * tempBW = new RooBreitWigner("breitWign", "breitWign", *x, *mean_, *gamma_);
      RooAbsPdf * tempCB = new RooCBShape("crystalBall", "crystalBall", *x, *mean_, *sigma2_, *alpha_, *n_);

      signal = new RooAddPdf("breitWignerPlusCrystalBall", "breitWignerPlusCrystalBall", RooArgList(*tempCB, *tempBW), *fGCB_);
    }

    else if( signalType != "" ) {
      std::cout << "Unknown signal function: " << signalType << ". Signal will not be in the model" << std::endl;
    }
    return signal;
  }

  /// Build the model for the specified background type
  RooAbsPdf * buildBackgroundModel(RooRealVar * x, const TString & backgroundType)
  {
    RooAbsPdf * background = 0;
    if( backgroundType == "exponential" ) {
      // Add an exponential for the background
      if( (expCoeff_ == 0) || (fsig_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeff and fsig" << std::endl;
	exit(1);
      }
      background = new RooExponential("exponential", "exponential", *x, *expCoeff_);
    }
    else if( backgroundType == "uniform" ) {
      // Add a constant background
      if( constant_ == 0 ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize constant" << std::endl;
	exit(1);
      }
      background = new RooPolynomial("uniform", "uniform", *x, *constant_, 0);
    }
    else if( backgroundType == "linear" ) {
      // Add a linear background
      if( (constant_ == 0) || (linearTerm_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize constant and linearTerm" << std::endl;
	exit(1);
      }
      background = new RooPolynomial("linear", "linear", *x, RooArgList(*constant_, *linearTerm_), 0);
    }
    else if( backgroundType == "3rdOrderPol" ) {
      // Add a 3rd order polynomial background
      if( (constant_ == 0) || (linearTerm_ == 0) || (parabolicTerm_ == 0) || (quarticTerm_ == 0)) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize constant, linearTerm, parabolicTerm and quarticTerm" << std::endl;
	exit(1);
      }
      background = new RooPolynomial("3rdOrderPol", "3rdOrderPol", *x, RooArgList(*constant_, *linearTerm_, *parabolicTerm_, *quarticTerm_), 0);
    }

    return background;
  }

  /// Build the model to fit
  RooAbsPdf * buildModel(RooRealVar * x, const TString & signalType, const TString & backgroundType)
  {
    RooAbsPdf * model = 0;

    RooAbsPdf * signal = buildSignalModel(x, signalType);
    RooAbsPdf * background = buildBackgroundModel(x, backgroundType);

    if( (signal != 0) && (background != 0) ) {
      // Combine signal and background pdfs
      std::cout << "Building model with signal and backgound" << std::endl;
      model = new RooAddPdf("model", "model", RooArgList(*signal, *background), *fsig_);
    }
    else if( signal != 0 ) {
      std::cout << "Building model with signal" << std::endl;
      model = signal;
    }
    else if( background != 0 ) {
      std::cout << "Building model with backgound" << std::endl;
      model = background;
    }
    else {
      std::cout << "Nothing to fit" << endl;
      exit(0);
    }
    return model;
  }

  bool useChi2_;

protected:

  // Declare all variables
  RooRealVar * mean_;
  RooRealVar * mean2_;
  RooRealVar * sigma_;
  RooRealVar * gamma_;
  RooRealVar * sigma2_;
  RooRealVar * gaussFrac_;
  RooRealVar * expCoeff_;
  RooRealVar * fsig_;
  RooRealVar * constant_;
  RooRealVar * linearTerm_;
  RooRealVar * parabolicTerm_;
  RooRealVar * quarticTerm_;
  RooRealVar * alpha_;
  RooRealVar * n_;
  RooRealVar * fGCB_;
};

#endif
