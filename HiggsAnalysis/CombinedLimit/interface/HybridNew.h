#ifndef HiggsAnalysis_CombinedLimit_HybridNew_h
#define HiggsAnalysis_CombinedLimit_HybridNew_h
/** \class HybridNew
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"
#include <algorithm> 
#include <RooStats/ModelConfig.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/ToyMCSampler.h>

class RooRealVar;

class HybridNew : public LimitAlgo {
public:
  HybridNew() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runLimit(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runSignificance(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runSinglePoint(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runTestStatistics(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual RooStats::HypoTestResult *readToysFromFile();
  virtual const std::string & name() const {
    static const std::string name("HybridNew");
    return name;
  }
  
  // I don't understand why if it's declared private it gcc does not like it
  enum WorkingMode { MakeLimit, MakeSignificance, MakePValues, MakeTestStatistics };
private:
  static WorkingMode workingMode_;
  static unsigned int nToys_;
  static double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
  static bool   rInterval_;
  static bool CLs_;
  static bool saveHybridResult_, readHybridResults_; 
  static std::string rule_, testStat_;
  static double rValue_;
  static unsigned int nCpu_, fork_;
  static bool importanceSamplingNull_, importanceSamplingAlt_;
  static std::string plot_;

  struct Setup {
    RooStats::ModelConfig modelConfig, modelConfig_bonly;
    std::auto_ptr<RooStats::TestStatistic> qvar;
    std::auto_ptr<RooStats::ToyMCSampler>  toymcsampler;
    std::auto_ptr<RooStats::ProofConfig> pc;
  };

  std::pair<double,double> eval(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, bool adaptive=false, double clsTarget=-1) ;
  std::auto_ptr<RooStats::HybridCalculator> create(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, Setup &setup);
  std::pair<double,double> eval(RooStats::HybridCalculator &hc, bool adaptive=false, double clsTarget=-1) ;
  RooStats::HypoTestResult *evalWithFork(RooStats::HybridCalculator &hc);

};

#endif
