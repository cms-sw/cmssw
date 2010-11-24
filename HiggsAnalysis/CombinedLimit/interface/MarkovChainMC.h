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
  MarkovChainMC() :
    options_("Markov Chain MC specific options") {
    options_.add_options()
      ("uniformProposal,u", boost::program_options::value<bool>(&uniformProposal_)->default_value(false), "Uniform proposal")
      ("iteration,i", boost::program_options::value<unsigned int>(&iterations_)->default_value(20000), "Number of iterations")
      ("burnInSteps,b", boost::program_options::value<unsigned int>(&burnInSteps_)->default_value(500), "Burn in steps")
      ("nBins,B", boost::program_options::value<unsigned int>(&numberOfBins_)->default_value(1000), "Number of bins");
    
  }
  bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("MarkovChainMC");
    return name;
  }
  virtual const boost::program_options::options_description & options() const {
    return options_;
  }
private:
  boost::program_options::options_description options_;
  bool uniformProposal_;
  unsigned int iterations_;
  unsigned int burnInSteps_;
  unsigned int numberOfBins_;
};

#endif
