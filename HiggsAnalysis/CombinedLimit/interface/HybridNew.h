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
#include <RooStats/ModelConfig.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/ToyMCSampler.h>

class RooRealVar;
namespace RooStats { class HybridNewCalculator; }

class HybridNew : public LimitAlgo {
public:
  HybridNew() ; 
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runLimit(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual bool runSignificance(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("HybridNew");
    return name;
  }

  
private:
  unsigned int nToys_, nCpu_;
  double clsAccuracy_, rAbsAccuracy_, rRelAccuracy_;
  bool   rInterval_;
  bool CLs_;
  std::string rule_, testStat_;

  struct Setup {
    RooStats::ModelConfig modelConfig, modelConfig_bonly;
    std::auto_ptr<RooStats::TestStatistic> qvar;
    std::auto_ptr<RooStats::ToyMCSampler>  toymcsampler;
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,28,0)
    std::auto_ptr<RooStats::ProofConfig> pc;
#endif
  };

  std::pair<double,double> eval(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, bool adaptive=false, double clsTarget=-1) ;
  std::auto_ptr<RooStats::HybridCalculator> create(RooWorkspace *w, RooAbsData &data, RooRealVar *r, double rVal, Setup &setup);
  std::pair<double,double> eval(RooStats::HybridCalculator &hc, bool adaptive=false, double clsTarget=-1) ;

};

#endif
