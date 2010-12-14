#ifndef HiggsAnalysis_CombinedLimit_HybridNew_h
#define HiggsAnalysis_CombinedLimit_HybridNew_h
/** \class HybridNew
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"
#include <algorithm> 

class RooRealVar;
namespace RooStats { class HybridNewCalculator; }

class HybridNew : public LimitAlgo {
public:
  HybridNew() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("HybridNew");
    return name;
  }

  
private:
  unsigned int nToys_;
  double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
  bool   rInterval_;
  bool CLs_;
  std::string rule_, testStat_;
  std::pair<double,double> eval(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, bool adaptive=false, double clsTarget=-1) ;
};

#endif
