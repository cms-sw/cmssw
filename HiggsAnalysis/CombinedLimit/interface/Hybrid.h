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
  Hybrid() : LimitAlgo("Hybrid specific options") {
    options_.add_options()
      ("toysH,T", boost::program_options::value<unsigned int>(&nToys_)->default_value(500), "Number of Toy MC extractions to compute CLs+b, CLb and CLs");
  }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("Hybrid");
    return name;
  }
private:
  unsigned int nToys_;
};

#endif
