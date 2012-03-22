#ifndef HiggsAnalysis_CombinedLimit_BayesianFlatPrior_h
#define HiggsAnalysis_CombinedLimit_BayesianFlatPrior_h
/** \class BayesianFlatPrior
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"

class BayesianFlatPrior : public LimitAlgo {
public:
  BayesianFlatPrior() ;
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("BayesianSimple");
    return name;
  }
private:
  static int maxDim_;
};

#endif
