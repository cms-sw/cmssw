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
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class ProfileLikelihood : public LimitAlgo {
public:
  ProfileLikelihood() ;
  virtual bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
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
  std::string minimizerAlgo_;
  float       minimizerTolerance_;

  // ----- options for handling cases where the likelihood fit misbihaves ------
  /// compute the limit N times
  int         tries_;
  /// trying up to M times from different points
  int         maxTries_;
  /// maximum relative deviation of the different points from the median to accept 
  float       maxRelDeviation_;
  /// Ignore up to this fraction of results if they're too far from the median
  float       maxOutlierFraction_;
  /// Stop trying after finding N outliers
  int         maxOutliers_;
  /// Try first a plain fit
  bool        preFit_;

  std::string plot_;

  bool runSignificance(RooWorkspace *w, RooAbsData &data, double &limit);
  bool runLimit(RooWorkspace *w, RooAbsData &data, double &limit);
};

#endif
