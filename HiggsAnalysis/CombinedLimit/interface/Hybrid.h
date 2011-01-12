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
#include <algorithm> 

class RooRealVar;
namespace RooStats { class HybridCalculatorOriginal; class HybridResult; }

class Hybrid : public LimitAlgo {
public:
  Hybrid() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runSignificance(RooStats::HybridCalculatorOriginal *hc, RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runLimit(RooStats::HybridCalculatorOriginal *hc, RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual RooStats::HybridResult *readToysFromFile();
  virtual const std::string & name() const {
    static const std::string name("Hybrid");
    return name;
  }

  
private:
  unsigned int nToys_;
  double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
  bool   rInterval_;
  bool CLs_;
  bool saveHybridResult_, readHybridResults_; 
  std::string rule_, testStat_;
  std::pair<double,double> eval(RooRealVar *r, double rVal, RooStats::HybridCalculatorOriginal *hc, bool adaptive=false, double clsTarget=-1) ;
};

#endif
