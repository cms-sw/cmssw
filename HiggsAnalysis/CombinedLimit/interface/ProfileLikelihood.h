#ifndef HiggsAnalysis_CombinedLimit_ProfileLikelihood_h
#define HiggsAnalysis_CombinedLimit_ProfileLikelihood_h
/** \class ProfileLikelihood
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "../interface/LimitAlgo.h"

class ProfileLikelihood : public LimitAlgo {
public:
  ProfileLikelihood() ;
  virtual bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("ProfileLikelihood");
    return name;
  }
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;

  /// Setup Minimizer configuration on creation, reset the previous one on destruction.
  class MinimizerSentry {
     public:
        MinimizerSentry(std::string &algo, double tolerance);
        ~MinimizerSentry();
     private:
        std::string minimizerTypeBackup, minimizerAlgoBackup;
        double minimizerTollBackup;
  };

protected:
  static std::string minimizerAlgo_;
  static float       minimizerTolerance_;

  // ----- options for handling cases where the likelihood fit misbihaves ------
  /// compute the limit N times
  static int         tries_;
  /// trying up to M times from different points
  static int         maxTries_;
  /// maximum relative deviation of the different points from the median to accept 
  static float       maxRelDeviation_;
  /// Ignore up to this fraction of results if they're too far from the median
  static float       maxOutlierFraction_;
  /// Stop trying after finding N outliers
  static int         maxOutliers_;
  /// Try first a plain fit
  static bool        preFit_;

  static std::string plot_;

  bool runSignificance(RooWorkspace *w, RooStats::ModelConfig *mc, RooAbsData &data, double &limit, double &limitErr);
  bool runLimit(RooWorkspace *w, RooStats::ModelConfig *mc, RooAbsData &data, double &limit, double &limitErr);
};

#endif
