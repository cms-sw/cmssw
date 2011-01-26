#ifndef HiggsAnalysis_CombinedLimit_FeldmanCousins_h
#define HiggsAnalysis_CombinedLimit_FeldmanCousins_h
/** \class FeldmanCousins
 *
 * Compute limit using FeldmanCousins++ 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class FeldmanCousins : public LimitAlgo {
public:
  FeldmanCousins() ;
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual const std::string & name() const {
    static const std::string name("FeldmanCousins");
    return name;
  }
private:
  double toysFactor_;
  double rAbsAccuracy_, rRelAccuracy_;
  bool lowerLimit_;
};

#endif
