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
#include "LimitAlgo.h"

class FeldmanCousins : public LimitAlgo {
public:
  FeldmanCousins() ;
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  virtual const std::string & name() const {
    static const std::string name("FeldmanCousins");
    return name;
  }
private:
  static float toysFactor_;
  static float rAbsAccuracy_, rRelAccuracy_;
};

#endif
