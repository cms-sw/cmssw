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
#include <string>
class RooWorkspace;
class RooAbsData;

class LimitAlgo {
public:
  LimitAlgo() { }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit) = 0;
  virtual const std::string & name() const = 0;
};

#endif
