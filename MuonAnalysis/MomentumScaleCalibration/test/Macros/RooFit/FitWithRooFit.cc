#ifndef FitWithRooFit_cc
#define FitWithRooFit_cc

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "TCanvas.h"
#include "TTree.h"
#include "TH1D.h"
#include "TRandom.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooVoigtian.h"
#include "RooExponential.h"
#include "RooPlot.h"
#include "RooDataHist.h"
#include "RooAddPdf.h"
#include "RooChebychev.h"
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

namespace
{
  typedef std::pair<RooRealVar, RooDataHist*> rooPair;
}

class FitWithRooFit
{
public:

  FitWithRooFit() :
    useChi2_(false), mean_(0), mean2_(0),  mean3_(0), sigma_(0), sigma2_(0), sigma3_(0), gamma_(0), gaussFrac_(0), gaussFrac2_(0),
    expCoeffa0_(0), expCoeffa1_(0), expCoeffa2_(0), fsig_(0),
    a0_(0), a1_(0), a2_(0), a3_(0), a4_(0), a5_(0), a6_(0), alpha_(0), n_(0), fGCB_(0)
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
    return( std::make_pair(x, new RooDataHist("dh","dh",x,RooFit::Import(*histo))) );
  }

  // Plot and fit a RooDataHist fitting signal and background
  void fit(TH1 * histo, const TString signalType, const TString backgroundType, const double & xMin = 0., const double & xMax = 0., bool sumW2Error = false)
  {
    rooPair imported = importTH1(histo, xMin, xMax);
    RooRealVar x(imported.first);
    RooDataHist * dh = imported.second;

    // Make plot of binned dataset showing Poisson error bars (RooFit default)
    RooPlot* frame = x.frame(RooFit::Title("Imported TH1 with Poisson error bars")) ;
    frame->SetName(TString(histo->GetName())+"_frame");
    dh->plotOn(frame);

    // Build the composite model
    RooAbsPdf * model = buildModel(&x, signalType, backgroundType);

    RooChi2Var chi2("chi2", "chi2", *model, *dh, RooFit::DataError(RooAbsData::SumW2));

    // Fit the composite model
    // -----------------------
    // Fit with likelihood
    if( !useChi2_ ) {
      if( sumW2Error ) {
	model->fitTo(*dh, RooFit::Save(), RooFit::SumW2Error(kTRUE));
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
    model->plotOn(frame, RooFit::Components(backgroundType), RooFit::LineStyle(kDotted), RooFit::LineColor(kRed));
    model->paramOn(frame, RooFit::Label("fit result"), RooFit::Format("NEU", RooFit::AutoPrecision(2)));

    // TODO: fix next lines to get the prob(chi2) (ndof should be dynamically set according to the choosen pdf) 
    // double chi2 = xframe.chiSquare("model","data",ndof);
    // double ndoff = xframeGetNbinsX();
    // double chi2prob = TMath::Prob(chi2,ndoff);

    // P l o t   a n d   f i t   a   R o o D a t a H i s t   w i t h   i n t e r n a l   e r r o r s
    // ---------------------------------------------------------------------------------------------

    // If histogram has custom error (i.e. its contents is does not originate from a Poisson process
    // but e.g. is a sum of weighted events) you can data with symmetric 'sum-of-weights' error instead
    // (same error bars as shown by ROOT)
    RooPlot* frame2 = x.frame(RooFit::Title("Imported TH1 with internal errors")) ;
    dh->plotOn(frame2,RooFit::DataError(RooAbsData::SumW2)) ; 
    model->plotOn(frame2);
    model->plotOn(frame2, RooFit::Components(backgroundType), RooFit::LineColor(kRed));
    model->paramOn(frame2, RooFit::Label("fit result"), RooFit::Format("NEU", RooFit::AutoPrecision(2)));


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

  void initMean3(const double & value, const double & min, const double & max, const TString & name = "mean3", const TString & title = "mean3")
  {
    if( mean3_ != 0 ) delete mean3_;
    mean3_ = new RooRealVar(name, title, value, min, max);
  }


  void initSigma(const double & value, const double & min, const double & max, const TString & name = "sigma", const TString & title = "sigma")
  {
    if( sigma_ != 0 ) delete sigma_;
    sigma_ = new RooRealVar(name, title, value, min, max);
  }

  void initSigma2(const double & value, const double & min, const double & max, const TString & name = "sigma2", const TString & title = "sigma2")
  {
    if( sigma2_ != 0 ) delete sigma2_;
    sigma2_ = new RooRealVar(name, title, value, min, max);
  }


  void initSigma3(const double & value, const double & min, const double & max, const TString & name = "sigma3", const TString & title = "sigma3")
  {
    if( sigma3_ != 0 ) delete sigma3_;
    sigma3_ = new RooRealVar(name, title, value, min, max);
  }

  void initGamma(const double & value, const double & min, const double & max, const TString & name = "gamma", const TString & title = "gamma")
  {
    if( gamma_ != 0 ) delete gamma_;
    gamma_ = new RooRealVar(name, title, value, min, max);
  }

  void initGaussFrac(const double & value, const double & min, const double & max, const TString & name = "GaussFrac", const TString & title = "GaussFrac")
  {
    if( gaussFrac_ != 0 ) delete gaussFrac_;
    gaussFrac_ = new RooRealVar(name, title, value, min, max);
  }

  void initGaussFrac2(const double & value, const double & min, const double & max, const TString & name = "GaussFrac2", const TString & title = "GaussFrac2")
  {
    if( gaussFrac2_ != 0 ) delete gaussFrac2_;
    gaussFrac2_ = new RooRealVar(name, title, value, min, max);
  }

  void initExpCoeffA0(const double & value, const double & min, const double & max, const TString & name = "expCoeffa0", const TString & title = "expCoeffa0")
  {
    if( expCoeffa0_ != 0 ) delete expCoeffa0_;
    expCoeffa0_ = new RooRealVar(name, title, value, min, max);
  }
  void initExpCoeffA1(const double & value, const double & min, const double & max, const TString & name = "expCoeffa1", const TString & title = "expCoeffa1")
  {
    if( expCoeffa1_ != 0 ) delete expCoeffa1_;
    expCoeffa1_ = new RooRealVar(name, title, value, min, max);
  }
  void initExpCoeffA2(const double & value, const double & min, const double & max, const TString & name = "expCoeffa2", const TString & title = "expCoeffa2")
  {
    if( expCoeffa2_ != 0 ) delete expCoeffa2_;
    expCoeffa2_ = new RooRealVar(name, title, value, min, max);
  }

  void initFsig(const double & value, const double & min, const double & max, const TString & name = "fsig", const TString & title = "signal fraction")
  {
    if( fsig_ != 0 ) delete fsig_;
    fsig_ = new RooRealVar(name, title, value, min, max);
  }

  void initA0(const double & value, const double & min, const double & max, const TString & name = "a0", const TString & title = "a0")
  {
    if( a0_ != 0 ) delete a0_;
    a0_ = new RooRealVar(name, title, value, min, max);
  }

  void initA1(const double & value, const double & min, const double & max, const TString & name = "a1", const TString & title = "a1")
  {
    if( a1_ != 0 ) delete a1_;
    a1_ = new RooRealVar(name, title, value, min, max);
  }

  void initA2(const double & value, const double & min, const double & max, const TString & name = "a2", const TString & title = "a2")
  {
    if( a2_ != 0 ) delete a2_;
    a2_ = new RooRealVar(name, title, value, min, max);
  }

  void initA3(const double & value, const double & min, const double & max, const TString & name = "a3", const TString & title = "a3")
  {
    if( a3_ != 0 ) delete a3_;
    a3_ = new RooRealVar(name, title, value, min, max);
  }

  void initA4(const double & value, const double & min, const double & max, const TString & name = "a4", const TString & title = "a4")
  {
    if( a4_ != 0 ) delete a4_;
    a4_ = new RooRealVar(name, title, value, min, max);
  }

  void initA5(const double & value, const double & min, const double & max, const TString & name = "a5", const TString & title = "a5")
  {
    if( a5_ != 0 ) delete a5_;
    a5_ = new RooRealVar(name, title, value, min, max);
  }

  void initA6(const double & value, const double & min, const double & max, const TString & name = "a6", const TString & title = "a6")
  {
    if( a6_ != 0 ) delete a6_;
    a6_ = new RooRealVar(name, title, value, min, max);
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

  inline RooRealVar * mean3()
  {
    return mean3_;
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

  inline RooRealVar * expCoeffa0()
  {
    return expCoeffa0_;
  }
  inline RooRealVar * expCoeffa1()
  {
    return expCoeffa1_;
  }
  inline RooRealVar * expCoeffa2()
  {
    return expCoeffa2_;
  }

  inline RooRealVar * fsig()
  {
    return fsig_;
  }

  inline RooRealVar * a0()
  {
    return a0_;
  }

  inline RooRealVar * a1()
  {
    return a1_;
  }

  inline RooRealVar * a2()
  {
    return a2_;
  }

  inline RooRealVar * a3()
  {
    return a3_;
  }

  inline RooRealVar * a4()
  {
    return a4_;
  }

  inline RooRealVar * a5()
  {
    return a5_;
  }

  inline RooRealVar * a6()
  {
    return a6_;
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
    else if( signalType == "tripleGaussian" ) {
      // Fit with triple gaussian
      if( (mean_ == 0) || (mean2_ == 0) || (mean3_ == 0) || (sigma_ == 0) || (sigma2_ == 0) || (sigma3_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean, mean2, mean3, sigma, sigma2, sigma3" << std::endl;
	exit(1);
      }
      RooGaussModel * gaussModel = new RooGaussModel("gaussModel","gaussModel",*x,*mean_,*sigma_);
      RooGaussModel * gaussModel2 = new RooGaussModel("gaussModel2","gaussModel2",*x,*mean2_,*sigma2_);
      RooGaussModel * gaussModel3 = new RooGaussModel("gaussModel3","gaussModel3",*x,*mean3_,*sigma3_);
      signal = new RooAddModel("tripleGaussian", "triple gaussian", RooArgList(*gaussModel, *gaussModel2, *gaussModel3), RooArgList(*gaussFrac_,*gaussFrac2_));
    }
    else if( signalType == "breitWigner" ) {
      // Fit a Breit-Wigner
      if( (mean_ == 0) || (gamma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma" << std::endl;
	exit(1);
      }
      signal = new RooBreitWigner("breiWign", "breitWign", *x, *mean_, *gamma_);
    }
    else if( signalType == "relBreitWigner" ) {
      // Fit a relativistic Breit-Wigner
      if( (mean_ == 0) || (gamma_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize mean and gamma" << std::endl;
	exit(1);
      }
      signal = new RooGenericPdf ("Relativistic Breit-Wigner","RBW","@0/(pow(@0*@0 - @1*@1,2) + @2*@2*@0*@0*@0*@0/(@1*@1))",RooArgList(*x,*mean_,*gamma_));
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
      signal = new RooFFTConvPdf("relBreitWignerTimesCB","relBreitWignerTimesCB",*x, *bw, *cb);
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
      if( (expCoeffa1_ == 0) || (fsig_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeffa1 and fsig" << std::endl;
	exit(1);
      }
      background = new RooExponential("exponential", "exponential", *x, *expCoeffa1_);
    }

    if( backgroundType == "exponentialpol" ) {
      // Add an exponential for the background
      if( (expCoeffa0_ == 0) ||  (expCoeffa1_ == 0) || (expCoeffa2_ == 0) || (fsig_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize expCoeff and fsig" << std::endl;
	exit(1);
      }
      background = new RooGenericPdf("exponential","exponential","TMath::Exp(@1+@2*@0+@3*@0*@0)",RooArgList(*x, *expCoeffa0_, *expCoeffa1_, *expCoeffa2_));
    }

    else if( backgroundType == "chebychev0" ) {
      // Add a linear background
      if( a0_ == 0 ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize a0" << std::endl;
	exit(1);
      }
      background = new RooChebychev("chebychev0", "chebychev0", *x, *a0_);
    }
    else if( backgroundType == "chebychev1" ) {
      // Add a 2nd order chebychev polynomial background
      if( (a0_ == 0) || (a1_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize a0 and a1" << std::endl;
	exit(1);
      }
      background = new RooChebychev("chebychev1", "chebychev1", *x, RooArgList(*a0_, *a1_));
    }
    else if( backgroundType == "chebychev3" ) {
      // Add a 3rd order chebychev polynomial background
      if( (a0_ == 0) || (a1_ == 0) || (a2_ == 0) || (a3_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize a0, a1, a2 and a3" << std::endl;
	exit(1);
      }
      background = new RooChebychev("3rdOrderPol", "3rdOrderPol", *x, RooArgList(*a0_,*a1_,*a2_,*a3_));
    }

    else if( backgroundType == "chebychev6" ) {
      // Add a 6th order chebychev polynomial background
      if( (a0_ == 0) || (a1_ == 0) || (a2_ == 0) || (a3_ == 0) || (a4_ == 0) || (a5_ == 0) || (a6_ == 0) ) {
	std::cout << "Error: one or more parameters are not initialized. Please be sure to initialize a0, a1, a2, a3, a4, a5 and a6" << std::endl;
	exit(1);
      }
      background = new RooChebychev("6thOrderPol", "6thOrderPol", *x, RooArgList(*a0_,*a1_,*a2_,*a3_,*a4_,*a5_,*a6_));
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
      std::cout << "Nothing to fit" << std::endl;
      exit(0);
    }
    return model;
  }

  bool useChi2_;

protected:

  // Declare all variables
  RooRealVar * mean_;
  RooRealVar * mean2_;
  RooRealVar * mean3_;
  RooRealVar * sigma_;
  RooRealVar * sigma2_;
  RooRealVar * sigma3_;
  RooRealVar * gamma_;
  RooRealVar * gaussFrac_;
  RooRealVar * gaussFrac2_;
  RooRealVar * expCoeffa0_;
  RooRealVar * expCoeffa1_;
  RooRealVar * expCoeffa2_;
  RooRealVar * fsig_;
  RooRealVar * a0_;
  RooRealVar * a1_;
  RooRealVar * a2_;
  RooRealVar * a3_;
  RooRealVar * a4_;
  RooRealVar * a5_;
  RooRealVar * a6_;
  RooRealVar * alpha_;
  RooRealVar * n_;
  RooRealVar * fGCB_;
};

#endif
