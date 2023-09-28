#ifndef Alignment_OfflineValidation_FitWithRooFit_h
#define Alignment_OfflineValidation_FitWithRooFit_h

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
#include "RooFitResult.h"
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

namespace {
  typedef std::pair<RooRealVar, RooDataHist*> rooPair;
}

class FitWithRooFit {
public:
  FitWithRooFit()
      : useChi2_(false),
        mean_(nullptr),
        mean2_(nullptr),
        mean3_(nullptr),
        sigma_(nullptr),
        sigma2_(nullptr),
        sigma3_(nullptr),
        gamma_(nullptr),
        gaussFrac_(nullptr),
        gaussFrac2_(nullptr),
        expCoeffa0_(nullptr),
        expCoeffa1_(nullptr),
        expCoeffa2_(nullptr),
        fsig_(nullptr),
        a0_(nullptr),
        a1_(nullptr),
        a2_(nullptr),
        a3_(nullptr),
        a4_(nullptr),
        a5_(nullptr),
        a6_(nullptr),
        alpha_(nullptr),
        n_(nullptr),
        fGCB_(nullptr) {}

  rooPair importTH1(TH1* histo, const double& inputXmin, const double& inputXmax);

  void fit(TH1* histo,
           const TString signalType,
           const TString backgroundType,
           const double& xMin = 0.,
           const double& xMax = 0.,
           bool sumW2Error = false);

  void initMean(const double& value,
                const double& min,
                const double& max,
                const TString& name = "mean",
                const TString& title = "mean");

  void initMean2(const double& value,
                 const double& min,
                 const double& max,
                 const TString& name = "mean2",
                 const TString& title = "mean2");

  void initMean3(const double& value,
                 const double& min,
                 const double& max,
                 const TString& name = "mean3",
                 const TString& title = "mean3");

  void initSigma(const double& value,
                 const double& min,
                 const double& max,
                 const TString& name = "sigma",
                 const TString& title = "sigma");

  void initSigma2(const double& value,
                  const double& min,
                  const double& max,
                  const TString& name = "sigma2",
                  const TString& title = "sigma2");

  void initSigma3(const double& value,
                  const double& min,
                  const double& max,
                  const TString& name = "sigma3",
                  const TString& title = "sigma3");

  void initGamma(const double& value,
                 const double& min,
                 const double& max,
                 const TString& name = "gamma",
                 const TString& title = "gamma");

  void initGaussFrac(const double& value,
                     const double& min,
                     const double& max,
                     const TString& name = "GaussFrac",
                     const TString& title = "GaussFrac");

  void initGaussFrac2(const double& value,
                      const double& min,
                      const double& max,
                      const TString& name = "GaussFrac2",
                      const TString& title = "GaussFrac2");

  void initExpCoeffA0(const double& value,
                      const double& min,
                      const double& max,
                      const TString& name = "expCoeffa0",
                      const TString& title = "expCoeffa0");

  void initExpCoeffA1(const double& value,
                      const double& min,
                      const double& max,
                      const TString& name = "expCoeffa1",
                      const TString& title = "expCoeffa1");

  void initExpCoeffA2(const double& value,
                      const double& min,
                      const double& max,
                      const TString& name = "expCoeffa2",
                      const TString& title = "expCoeffa2");

  void initFsig(const double& value,
                const double& min,
                const double& max,
                const TString& name = "fsig",
                const TString& title = "signal fraction");

  void initA0(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a0",
              const TString& title = "a0");

  void initA1(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a1",
              const TString& title = "a1");

  void initA2(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a2",
              const TString& title = "a2");

  void initA3(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a3",
              const TString& title = "a3");

  void initA4(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a4",
              const TString& title = "a4");

  void initA5(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a5",
              const TString& title = "a5");

  void initA6(const double& value,
              const double& min,
              const double& max,
              const TString& name = "a6",
              const TString& title = "a6");

  void initAlpha(const double& value,
                 const double& min,
                 const double& max,
                 const TString& name = "alpha",
                 const TString& title = "alpha");

  void initN(
      const double& value, const double& min, const double& max, const TString& name = "n", const TString& title = "n");

  void initFGCB(const double& value,
                const double& min,
                const double& max,
                const TString& name = "fGCB",
                const TString& title = "fGCB");

  inline RooRealVar* mean() { return mean_; }
  inline RooRealVar* mean2() { return mean2_; }
  inline RooRealVar* mean3() { return mean3_; }
  inline RooRealVar* sigma() { return sigma_; }
  inline RooRealVar* sigma2() { return sigma2_; }
  inline RooRealVar* sigma3() { return sigma3_; }
  inline RooRealVar* gamma() { return gamma_; }
  inline RooRealVar* gaussFrac() { return gaussFrac_; }
  inline RooRealVar* gaussFrac2() { return gaussFrac2_; }
  inline RooRealVar* expCoeffa0() { return expCoeffa0_; }
  inline RooRealVar* expCoeffa1() { return expCoeffa1_; }
  inline RooRealVar* expCoeffa2() { return expCoeffa2_; }
  inline RooRealVar* fsig() { return fsig_; }
  inline RooRealVar* a0() { return a0_; }
  inline RooRealVar* a1() { return a1_; }
  inline RooRealVar* a2() { return a2_; }
  inline RooRealVar* a3() { return a3_; }
  inline RooRealVar* a4() { return a4_; }
  inline RooRealVar* a5() { return a5_; }
  inline RooRealVar* a6() { return a6_; }
  inline RooRealVar* alpha() { return alpha_; }
  inline RooRealVar* n() { return n_; }
  inline RooRealVar* fGCB() { return fGCB_; }

  void reinitializeParameters();

  RooAbsPdf* buildSignalModel(RooRealVar* x, const TString& signalType);
  RooAbsPdf* buildBackgroundModel(RooRealVar* x, const TString& backgroundType);
  RooAbsPdf* buildModel(RooRealVar* x, const TString& signalType, const TString& backgroundType);

  bool useChi2_;

protected:
  // Declare all variables
  RooRealVar* mean_;
  RooRealVar* mean2_;
  RooRealVar* mean3_;
  RooRealVar* sigma_;
  RooRealVar* sigma2_;
  RooRealVar* sigma3_;
  RooRealVar* gamma_;
  RooRealVar* gaussFrac_;
  RooRealVar* gaussFrac2_;
  RooRealVar* expCoeffa0_;
  RooRealVar* expCoeffa1_;
  RooRealVar* expCoeffa2_;
  RooRealVar* fsig_;
  RooRealVar* a0_;
  RooRealVar* a1_;
  RooRealVar* a2_;
  RooRealVar* a3_;
  RooRealVar* a4_;
  RooRealVar* a5_;
  RooRealVar* a6_;
  RooRealVar* alpha_;
  RooRealVar* n_;
  RooRealVar* fGCB_;

  // Initial values
  double initVal_mean;
  double initVal_mean2;
  double initVal_mean3;
  double initVal_sigma;
  double initVal_sigma2;
  double initVal_sigma3;
  double initVal_gamma;
  double initVal_gaussFrac;
  double initVal_gaussFrac2;
  double initVal_expCoeffa0;
  double initVal_expCoeffa1;
  double initVal_expCoeffa2;
  double initVal_fsig;
  double initVal_a0;
  double initVal_a1;
  double initVal_a2;
  double initVal_a3;
  double initVal_a4;
  double initVal_a5;
  double initVal_a6;
  double initVal_alpha;
  double initVal_n;
  double initVal_fGCB;
};

#endif
