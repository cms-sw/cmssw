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
 MarkovChainMC(bool verbose, bool withSystematics, bool uniformProposal) : verbose_(verbose), withSystematics_(withSystematics), uniformProposal_(uniformProposal) { }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
private:
  bool verbose_;
  bool withSystematics_;
  bool uniformProposal_;
};

#endif
