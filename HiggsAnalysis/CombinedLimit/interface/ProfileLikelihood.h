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
  bool        hitItUntilItConverges_, hitItEvenHarder_, acceptEverything_;

  bool runSignificance(RooWorkspace *w, RooAbsData &data, double &limit);
  bool runLimit(RooWorkspace *w, RooAbsData &data, double &limit);
};

#endif
