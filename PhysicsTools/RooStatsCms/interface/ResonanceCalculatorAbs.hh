#ifndef __RESONANCE_CALCULATOR_ABS_HH__
#define __RESONANCE_CALCULATOR_ABS_HH__

//
// author: J.P. Chou (Brown University)
//
// Definition of the abstract resonance calculator.  This takes as input a datafile or a 1D histogram, and scans
// the space for the single most significant resonance.  The significance of this resonance is computed, accounting
// for the look-elsewhere effect by throwing pseudoexperiments.  The calculator does not make assumptions about the
// shape of the background or the signal, but leaves it to the concrete implementation to specify.
//

#include "RooBinning.h"
#include "RooRealVar.h"

#include <vector>

class RooWorkspace;
class RooAbsPdf;
class RooAbsReal;
class RooArgSet;
class RooAbsData;
class RooDataHist;
class RooDataSet;
class TH1;
class TFile;

class ResonanceCalculatorAbs
{

public:

  ////////////////////////////////////////////////////////////////////////////////
  // constructors and destructor
  ////////////////////////////////////////////////////////////////////////////////

  ResonanceCalculatorAbs();
  ResonanceCalculatorAbs(const ResonanceCalculatorAbs& leec);
  virtual ~ResonanceCalculatorAbs();

  ////////////////////////////////////////////////////////////////////////////////
  // primary function (this is where the magic happens)
  ////////////////////////////////////////////////////////////////////////////////

  // Runs the calculator, putting the results into a TFile with a given name.
  // If the number of pseudoexperiments (PEs) to run is set to 0, it returns
  // the test statistic result of the bump, which roughly approximates the
  // significance uncorrected for the look elsewhere effect (LEE).
  // If the number of PEs to run is set to >0, it returns the significance,
  // corrected for the LEE.
  double calculate(const char* rootfilename);

  // Same as above, but also gives the user the (test statistic, weight) pair from each PE.
  double calculate(const char* rootfilename, std::vector<std::pair<double, double> >& teststats);

  ////////////////////////////////////////////////////////////////////////////////
  // provide the data
  ////////////////////////////////////////////////////////////////////////////////

  // input is a 1-D histogram (minimum and maximum bin values are considered the boundaries)
  void setBinnedData(TH1* dataHist);

  // input is a text file with a list of data points, but the data gets binned
  // the number of bins, as well as the minimum and maximum bin values must be specified
  void setBinnedData(const char* filename, int nbins, double minx, double maxx);

  // input is a text file with a list of data points; the data remains unbinned
  // the minimum and maximum observable values must be specified
  void setUnbinnedData(const char* filename, double minx, double maxx);

  ////////////////////////////////////////////////////////////////////////////////
  // change default values
  ////////////////////////////////////////////////////////////////////////////////

  // number of PEs to run
  // default is 1000
  void setNumPseudoExperiments(int numPEs) { numPEs_=numPEs; return; }

  // set the verbosity level (<0=quiet, 0=nominal, 1=default, 2=verbose)
  static void setPrintLevel(int level) { printLevel_=level; return; }

  // set the minimum and maximum signal mass to consider
  // default is +/- 3 units in signal width from the data boundaries
  // N.B: this functionality gets overridden by the default if data is added after
  // this function is called.
  void setMinMaxSignalMass(double min, double max) { sigmass_->setMin(min); sigmass_->setMax(max); return; }

  // sets the control region and overrides the default behavior.
  // The default behavior of the calculator is to determine the background in the same region as the signal
  // by fitting for the background only while excluding the signal region within +/-3 units of width.
  // One can re-instate the default behavior by setting a minimum greater than the maximum.
  void setMinMaxControlMass(double min, double max) { controlMin_=min; controlMax_=max; return; }
  

  // set the number of bins used to draw the fit results
  // if you use binned data, drawing the data with a different number of bins than what is used in the data
  // can result in artifacts in the plotting.
  // default is 100
  void setNumBinsToDraw(int nbins) { nBinsToDraw_=nbins; }

  // use the simple likelihood ratio as a test statistic
  // (this is the default)
  void useSimpleLikelihoodRatioTestStatistic(void) { whichTestStatistic_=0; }  

  // use the best fit of the number of signal events divided by the fit error as the test statistic
  void useFitErrorTestStatistic(void) { whichTestStatistic_=1; }

  // set the step size to search for a resonance (in units of resonance width)
  // default is 1.0
  void setSearchStepSize(double m) { searchStepSize_=m; }

  // set the random seed (default is 1)
  void setRandomSeed(int seed) { randomSeed_=seed; }

  // set the fit strategy (default is 2)
  void setFitStrategy(int strategy) { fitStrategy_=strategy; }

  ////////////////////////////////////////////////////////////////////////////////
  // get internally used parameters (this for expert use, only)
  ////////////////////////////////////////////////////////////////////////////////

  RooWorkspace* getWorkspace(void) const { return ws_; }
  RooAbsData* getData(void) const { return data_; }
  RooRealVar* getObservable(void) const { return obs_; }
  RooAbsPdf* getModel(void) const { return model_; }
  RooAbsPdf* getSignalPdf(void) const { return signal_; }
  RooAbsPdf* getBackgroundPdf(void) const { return background_; }
  RooRealVar* getSignalMass(void) const { return sigmass_; }
  RooAbsReal* getSignalWidth(void) const { return sigwidth_; }
  RooArgSet* getOtherSignalParameters(void) const { return sigparams_; } // signal parameters that is not the mass
  RooArgSet* getBackgroundParameters(void) const { return bkgparams_; }
  RooRealVar* getSignalNormalization(void) const { return nsig_; }
  RooRealVar* getBackgroundNormalization(void) const { return nbkg_; }

protected:

  // this must be called by the constructor of the final concrete implementation of this class
  void setupWorkspace(void);

  // called by setupWorkspace() to specify the background pdf, signal width, and signal pdf
  virtual RooAbsPdf* setupBackgroundPdf(void) =0;
  virtual RooAbsReal* setupSignalWidth(void) =0;
  virtual RooAbsPdf* setupSignalPdf(void) =0;

  // this can be overwritten by the inheriting class
  virtual void drawFitResult(const char* label, const char* range);

  // parameters
  int numPEs_;
  int nBinsToDraw_;
  int whichTestStatistic_;
  double searchStepSize_;
  int randomSeed_;
  static int printLevel_;
  int fitStrategy_;
  double controlMin_, controlMax_;

  // stored values
  RooBinning dataBinning_;
  double dataIntegral_;

  // workspace
  RooWorkspace *ws_;

  // pointers to objects inside the workspace
  RooAbsData* data_;
  RooRealVar* obs_;
  RooAbsPdf* model_;
  RooAbsPdf* signal_;
  RooAbsPdf* background_;
  RooRealVar* sigmass_;
  RooAbsReal* sigwidth_;
  RooRealVar* nsig_;
  RooRealVar* nbkg_;
  RooArgSet* sigparams_;
  RooArgSet* bkgparams_;

private:

  // helper functions
  void scanForBump(const char* label);
  void scanForBumpWithControl(const char* label);
  RooFitResult* doBkgOnlyFit(const char* label);
  RooFitResult* doBkgOnlyExcludeWindowFit(const char* label);
  RooFitResult* doSigOnlyFixMassFit(const char* label);
  RooFitResult* doSigOnlyFloatMassFit(const char* label, double minMass, double maxMass);
  RooFitResult* doBkgPlusSigFixMassFit(const char* label);
  RooFitResult* doBkgPlusSigFloatMassFit(const char* label, double minMass, double maxMass);
  RooFitResult* doFit(const char* label, const char* range);
  void findSignalAndBackgroundParams(void);
  void setSigParamsConst(bool isConst);
  void setBkgParamsConst(bool isConst);
  void copyValuesToBkgParams(RooArgList* params);
  void findMinMaxMass(void);
  double evaluateTestStatistic(void);
  RooDataHist* generateBinned(RooAbsPdf* pdf, RooDataHist* templateDataHist, int numEntries);
  RooDataSet* generateUnbinned(RooAbsPdf* pdf, int numEntries);
};


#endif
