#ifndef HiggsAnalysis_CombinedLimit_MarkovChainMC_h
#define HiggsAnalysis_CombinedLimit_MarkovChainMC_h
/** \class MarkovChainMC
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class MarkovChainMC : public LimitAlgo {
public:
  bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("MarkovChainMC");
    return name;
  }
  virtual boost::program_options::options_description options() {
    boost::program_options::options_description d("Markov Chain MC specific options");
    d.add_options()
      ("uniformProposal,u", boost::program_options::value<bool>(&uniformProposal_)->default_value(false), "Uniform proposal");
    return d;
  }
private:
  bool uniformProposal_;
};

#endif
