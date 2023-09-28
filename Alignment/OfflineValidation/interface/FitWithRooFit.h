#ifndef Alignment_OfflineValidation_FitWithRooFit_h
#define Alignment_OfflineValidation_FitWithRooFit_h

#include "RooGlobalFunc.h"
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
#include "RooMinimizer.h"
#include "RooBreitWigner.h"
#include "RooFFTConvPdf.h"

/**
 * This class allows to use RooFit to perform a fit on a TH1 histogram. <br>
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

class FitWithRooFit {
public:
  std::unique_ptr<RooDataHist> importTH1(TH1* histo, double xMin, double xMax);

  void fit(TH1* histo,
           const TString signalType,
           const TString backgroundType,
           double xMin = 0.,
           double xMax = 0.,
           bool sumW2Error = false);

  void initMean(double value, double min, double max, const TString& name = "mean", const TString& title = "mean");
  void initMean2(double value, double min, double max, const TString& name = "mean2", const TString& title = "mean2");
  void initMean3(double value, double min, double max, const TString& name = "mean3", const TString& title = "mean3");
  void initSigma(double value, double min, double max, const TString& name = "sigma", const TString& title = "sigma");
  void initSigma2(double value, double min, double max, const TString& name = "sigma2", const TString& title = "sigma2");
  void initSigma3(double value, double min, double max, const TString& name = "sigma3", const TString& title = "sigma3");
  void initGamma(double value, double min, double max, const TString& name = "gamma", const TString& title = "gamma");
  void initGaussFrac(
      double value, double min, double max, const TString& name = "GaussFrac", const TString& title = "GaussFrac");
  void initGaussFrac2(
      double value, double min, double max, const TString& name = "GaussFrac2", const TString& title = "GaussFrac2");
  void initExpCoeffA0(
      double value, double min, double max, const TString& name = "expCoeffa0", const TString& title = "expCoeffa0");
  void initExpCoeffA1(
      double value, double min, double max, const TString& name = "expCoeffa1", const TString& title = "expCoeffa1");
  void initExpCoeffA2(
      double value, double min, double max, const TString& name = "expCoeffa2", const TString& title = "expCoeffa2");
  void initFsig(
      double value, double min, double max, const TString& name = "fsig", const TString& title = "signal fraction");
  void initA0(double value, double min, double max, const TString& name = "a0", const TString& title = "a0");
  void initA1(double value, double min, double max, const TString& name = "a1", const TString& title = "a1");
  void initA2(double value, double min, double max, const TString& name = "a2", const TString& title = "a2");
  void initA3(double value, double min, double max, const TString& name = "a3", const TString& title = "a3");
  void initA4(double value, double min, double max, const TString& name = "a4", const TString& title = "a4");
  void initA5(double value, double min, double max, const TString& name = "a5", const TString& title = "a5");
  void initA6(double value, double min, double max, const TString& name = "a6", const TString& title = "a6");
  void initAlpha(double value, double min, double max, const TString& name = "alpha", const TString& title = "alpha");
  void initN(double value, double min, double max, const TString& name = "n", const TString& title = "n");
  void initFGCB(double value, double min, double max, const TString& name = "fGCB", const TString& title = "fGCB");

  inline RooRealVar* mean() { return mean_.get(); }
  inline RooRealVar* mean2() { return mean2_.get(); }
  inline RooRealVar* mean3() { return mean3_.get(); }
  inline RooRealVar* sigma() { return sigma_.get(); }
  inline RooRealVar* sigma2() { return sigma2_.get(); }
  inline RooRealVar* sigma3() { return sigma3_.get(); }
  inline RooRealVar* gamma() { return gamma_.get(); }
  inline RooRealVar* gaussFrac() { return gaussFrac_.get(); }
  inline RooRealVar* gaussFrac2() { return gaussFrac2_.get(); }
  inline RooRealVar* expCoeffa0() { return expCoeffa0_.get(); }
  inline RooRealVar* expCoeffa1() { return expCoeffa1_.get(); }
  inline RooRealVar* expCoeffa2() { return expCoeffa2_.get(); }
  inline RooRealVar* fsig() { return fsig_.get(); }
  inline RooRealVar* a0() { return a0_.get(); }
  inline RooRealVar* a1() { return a1_.get(); }
  inline RooRealVar* a2() { return a2_.get(); }
  inline RooRealVar* a3() { return a3_.get(); }
  inline RooRealVar* a4() { return a4_.get(); }
  inline RooRealVar* a5() { return a5_.get(); }
  inline RooRealVar* a6() { return a6_.get(); }
  inline RooRealVar* alpha() { return alpha_.get(); }
  inline RooRealVar* n() { return n_.get(); }
  inline RooRealVar* fGCB() { return fGCB_.get(); }

  void reinitializeParameters();

  std::unique_ptr<RooAbsPdf> buildSignalModel(RooRealVar* x, const TString& signalType);
  std::unique_ptr<RooAbsPdf> buildBackgroundModel(RooRealVar* x, const TString& backgroundType);
  std::unique_ptr<RooAbsPdf> buildModel(RooRealVar* x, const TString& signalType, const TString& backgroundType);

  bool useChi2_ = false;

protected:
  // Declare all variables
  std::unique_ptr<RooRealVar> mean_;
  std::unique_ptr<RooRealVar> mean2_;
  std::unique_ptr<RooRealVar> mean3_;
  std::unique_ptr<RooRealVar> sigma_;
  std::unique_ptr<RooRealVar> sigma2_;
  std::unique_ptr<RooRealVar> sigma3_;
  std::unique_ptr<RooRealVar> gamma_;
  std::unique_ptr<RooRealVar> gaussFrac_;
  std::unique_ptr<RooRealVar> gaussFrac2_;
  std::unique_ptr<RooRealVar> expCoeffa0_;
  std::unique_ptr<RooRealVar> expCoeffa1_;
  std::unique_ptr<RooRealVar> expCoeffa2_;
  std::unique_ptr<RooRealVar> fsig_;
  std::unique_ptr<RooRealVar> a0_;
  std::unique_ptr<RooRealVar> a1_;
  std::unique_ptr<RooRealVar> a2_;
  std::unique_ptr<RooRealVar> a3_;
  std::unique_ptr<RooRealVar> a4_;
  std::unique_ptr<RooRealVar> a5_;
  std::unique_ptr<RooRealVar> a6_;
  std::unique_ptr<RooRealVar> alpha_;
  std::unique_ptr<RooRealVar> n_;
  std::unique_ptr<RooRealVar> fGCB_;

  // Initial values
  double initVal_mean = 0.0;
  double initVal_mean2 = 0.0;
  double initVal_mean3 = 0.0;
  double initVal_sigma = 0.0;
  double initVal_sigma2 = 0.0;
  double initVal_sigma3 = 0.0;
  double initVal_gamma = 0.0;
  double initVal_gaussFrac = 0.0;
  double initVal_gaussFrac2 = 0.0;
  double initVal_expCoeffa0 = 0.0;
  double initVal_expCoeffa1 = 0.0;
  double initVal_expCoeffa2 = 0.0;
  double initVal_fsig = 0.0;
  double initVal_a0 = 0.0;
  double initVal_a1 = 0.0;
  double initVal_a2 = 0.0;
  double initVal_a3 = 0.0;
  double initVal_a4 = 0.0;
  double initVal_a5 = 0.0;
  double initVal_a6 = 0.0;
  double initVal_alpha = 0.0;
  double initVal_n = 0.0;
  double initVal_fGCB = 0.0;
};

#endif
