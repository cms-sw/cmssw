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
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class BayesianFlatPrior : public LimitAlgo {
public:
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("BayesianFlatPrior");
    return name;
  }
  virtual const boost::program_options::options_description & options() const {
    return options_;
  }
private:
  boost::program_options::options_description options_;
};

#endif
