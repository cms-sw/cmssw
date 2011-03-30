#ifndef HiggsAnalysis_CombinedLimit_Hybrid_h
#define HiggsAnalysis_CombinedLimit_Hybrid_h
/** \class Hybrid
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), Giovanni Petrucciani (UCSD)
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
  virtual void applyDefaultOptions() ; 
    

  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runSignificance(RooStats::HybridCalculatorOriginal &hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runLimit(RooStats::HybridCalculatorOriginal &hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runSinglePoint(RooStats::HybridCalculatorOriginal &hc, RooWorkspace *w, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual RooStats::HybridResult *readToysFromFile();
  virtual const std::string & name() const {
    static const std::string name("Hybrid");
    return name;
  }
  
private:
  static unsigned int nToys_;
  static double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
  static unsigned int fork_;
  static std::string rule_, testStat_;
  static bool rInterval_;
  static double rValue_;
  static bool CLs_;
  static bool saveHybridResult_, readHybridResults_; 
  static bool singlePointScan_; 

  void validateOptions() ;

  std::pair<double,double> eval(RooRealVar *r, double rVal, RooStats::HybridCalculatorOriginal &hc, bool adaptive=false, double clsTarget=-1) ;

  RooStats::HybridResult *evalWithFork(RooStats::HybridCalculatorOriginal &hc);
};

#endif
