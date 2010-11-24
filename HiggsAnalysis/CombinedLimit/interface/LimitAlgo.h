#ifndef HiggsAnalysis_CombinedLimit_LimitAlgo_h
#define HiggsAnalysis_CombinedLimit_LimitAlgo_h
/** \class LimitAlgo
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN)
 *
 *
 */

class RooWorkspace;
class RooAbsData;

class LimitAlgo {
public:
  LimitAlgo() { }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit) = 0;
};

#endif
