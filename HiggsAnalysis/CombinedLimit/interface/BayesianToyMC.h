#ifndef HiggsAnalysis_CombinedLimit_BayesianToyMC_h
#define HiggsAnalysis_CombinedLimit_BayesianToyMC_h
/** \class BayesianToyMC
 *
 * Interface to BayesianCalculator used with toymc algorithms 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"

class BayesianToyMC : public LimitAlgo {
public:
  BayesianToyMC() ;
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  virtual const std::string & name() const {
    static const std::string name("BayesianToyMC");
    return name;
  }
private:
  /// numerical integration algorithm
  static std::string integrationType_;
  /// number of iterations for each toy mc computation
  static int numIters_;
  /// number of toy mc computations to run
  static unsigned int tries_;
  /// Safety factor for hint (integrate up to this number of times the hinted limit)
  static float hintSafetyFactor_;
};

#endif
