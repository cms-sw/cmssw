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
class TGraphErrors;

class HybridNew : public LimitAlgo {
public:
  HybridNew() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  virtual void applyDefaultOptions() ; 

  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runSignificance(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runSinglePoint(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual bool runTestStatistics(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
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
  static std::string algo_;
  static std::string plot_;

  static bool optimizeProductPdf_;
  static bool optimizeTestStatistics_;
 
  // plot
  std::auto_ptr<TGraphErrors> limitPlot_;
 
  // performance counter: remember how many toys have been thrown
  unsigned int perf_totalToysRun_;

  struct Setup {
    RooStats::ModelConfig modelConfig, modelConfig_bonly;
    std::auto_ptr<RooStats::TestStatistic> qvar;
    std::auto_ptr<RooStats::ToyMCSampler>  toymcsampler;
    std::auto_ptr<RooStats::ProofConfig> pc;
  };

  void validateOptions() ;

  std::pair<double,double> eval(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, bool adaptive=false, double clsTarget=-1) ;
  std::auto_ptr<RooStats::HybridCalculator> create(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double rVal, Setup &setup);
  std::pair<double,double> eval(RooStats::HybridCalculator &hc, double rVal, bool adaptive=false, double clsTarget=-1) ;
  RooStats::HypoTestResult *evalWithFork(RooStats::HybridCalculator &hc);
  RooStats::HypoTestResult *readToysFromFile(double rValue=0);

};

#endif
