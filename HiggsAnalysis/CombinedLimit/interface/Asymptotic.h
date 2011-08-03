#ifndef HiggsAnalysis_CombinedLimit_Asymptotic_h
#define HiggsAnalysis_CombinedLimit_Asymptotic_h
/** \class Asymptotic
 *
 * CLs asymptotic limits 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"
#include <memory>
class RooRealVar;
#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooFitResult.h>

class Asymptotic : public LimitAlgo {
public:
  Asymptotic() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  virtual void applyDefaultOptions() ; 

  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runLimitExpected(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) ;

  virtual const std::string& name() const { static std::string name_ = "Asymptotic"; return name_; }
private:
  static double rAbsAccuracy_, rRelAccuracy_;
  static bool expected_; 
  static float quantileForExpected_;
  static std::string minimizerAlgo_;
  static float       minimizerTolerance_;

  mutable std::auto_ptr<RooArgSet>  params_;
  mutable std::auto_ptr<RooAbsReal> nllD_, nllA_; 
  mutable std::auto_ptr<RooFitResult> fitFreeD_, fitFreeA_;
  mutable std::auto_ptr<RooFitResult> fitFixD_, fitFixA_;

  RooAbsData *asimovDataset(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data);
  double getCLs(RooRealVar &r, double rVal);
};

#endif
