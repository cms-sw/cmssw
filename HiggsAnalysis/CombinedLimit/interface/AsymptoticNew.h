#ifndef HiggsAnalysis_CombinedLimit_AsymptoticNew_h
#define HiggsAnalysis_CombinedLimit_AsymptoticNew_h
/** \class AsymptoticNew
 *
 * new CLs asymptotic limits
 *
 * \author Nicholas Wardle
 * (mostly taken from Asymptotic class)
 *
 */

#include "../interface/LimitAlgo.h"
#include <memory>
class RooRealVar;

class AsymptoticNew : public LimitAlgo {
public:
  AsymptoticNew() ;
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  virtual void applyDefaultOptions() ; 
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  std::vector<std::pair<float,float> > runLimit(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) ;
  virtual const std::string& name() const { static std::string name_ = "AsymptoticNew"; return name_; }
  
private:
  static std::string what_;
  static double rValue_;  
  static int nscanpoints_;
  static bool qtilde_;
  static double maxrscan_,minrscan_;

};

#endif
