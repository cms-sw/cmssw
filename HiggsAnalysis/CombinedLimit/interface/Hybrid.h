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
      ("toysH,T", boost::program_options::value<unsigned int>(&nToys_)->default_value(500), "Number of Toy MC extractions to compute CLs+b, CLb and CLs")
      ("clsAcc",  boost::program_options::value<double>(&clsAccuracy_ )->default_value(0.005), "Absolute accuracy on CLs to reach to terminate the scan")
      ("rAbsAcc", boost::program_options::value<double>(&rAbsAccuracy_)->default_value(0.1),   "Absolute accuracy on r to reach to terminate the scan")
      ("rRelAcc", boost::program_options::value<double>(&rRelAccuracy_)->default_value(0.05),  "Relative accuracy on r to reach to terminate the scan")
    ;
  }
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit);
  virtual const std::string & name() const {
    static const std::string name("Hybrid");
    return name;
  }
private:
  unsigned int nToys_;
  double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
};

#endif
