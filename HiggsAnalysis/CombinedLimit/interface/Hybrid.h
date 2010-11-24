#ifndef HiggsAnalysis_CombinedLimit_Hybrid_h
#define HiggsAnalysis_CombinedLimit_Hybrid_h
/** \class Hybrid
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class Hybrid : public LimitAlgo {
public:
 Hybrid(bool verbose, bool withSystematics) : verbose_(verbose), withSystematics_(withSystematics) { }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
private:
  bool verbose_;
  bool withSystematics_;
};

#endif
