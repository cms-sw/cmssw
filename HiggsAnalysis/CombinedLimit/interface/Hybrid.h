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
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("Hybrid");
    return name;
  }
  virtual boost::program_options::options_description options() {
    boost::program_options::options_description d;
    return d;
  }
private:
};

#endif
