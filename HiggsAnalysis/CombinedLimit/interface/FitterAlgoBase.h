#ifndef HiggsAnalysis_CombinedLimit_FitterAlgoBase_h
#define HiggsAnalysis_CombinedLimit_FitterAlgoBase_h
/** \class FitterAlgoBase
 *
 * Do a ML fit of the data with background and signal+background hypothesis and print out diagnostics plots 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"
#include "../interface/ProfileLikelihood.h"
class RooFitResult;
class RooMinimizer;
class RooCmdArg;
class RooAbsReal;
class RooArgList;
class CascadeMinimizer;

class FitterAlgoBase : public LimitAlgo {
public:
  FitterAlgoBase(const char *title="<FillMe> specific options") ;

  void applyOptionsBase(const boost::program_options::variables_map &vm) ;

  // configures the minimizer and then calls runSpecific
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);

protected:
  static std::string minimizerAlgo_, minimizerAlgoForMinos_;
  static float       minimizerTolerance_, minimizerToleranceForMinos_;
  static int         minimizerStrategy_, minimizerStrategyForMinos_;

  static float preFitValue_;

  static bool robustFit_, do95_;
  static float stepSize_;
  static int   maxFailedSteps_;

  static bool  saveNLL_, keepFailures_;
  static float nllValue_;
  // method that is implemented in the subclass
  virtual bool runSpecific(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) = 0;

  // utilities
  RooFitResult *doFit(RooAbsPdf &pdf, RooAbsData &data, RooRealVar &r,  const RooCmdArg &constrain, bool doHesse=true) ;
  RooFitResult *doFit(RooAbsPdf &pdf, RooAbsData &data, RooArgList &rs, const RooCmdArg &constrain, bool doHesse=true) ;
  double findCrossing(CascadeMinimizer &minim, RooAbsReal &nll, RooRealVar &r, double level, double rStart, double rBound) ;
};


#endif
