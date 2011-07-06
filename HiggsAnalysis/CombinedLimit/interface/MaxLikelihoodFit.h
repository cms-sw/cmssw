#ifndef HiggsAnalysis_CombinedLimit_MaxLikelihoodFit_h
#define HiggsAnalysis_CombinedLimit_MaxLikelihoodFit_h
/** \class MaxLikelihoodFit
 *
 * Do a ML fit of the data with background and signal+background hypothesis and print out diagnostics plots 
 *
 * \author Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"
#include "../interface/ProfileLikelihood.h"

class MaxLikelihoodFit : public LimitAlgo {
public:
  MaxLikelihoodFit() ;
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("MaxLikelihoodFit");
    return name;
  }
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

protected:
  static std::string minimizerAlgo_;
  static float       minimizerTolerance_;
  static int         minimizerStrategy_;
  static std::string minos_;

  static float preFitValue_;

  static std::string out_; 
  static bool        makePlots_;
  static std::string signalPdfNames_, backgroundPdfNames_;
};


#endif
