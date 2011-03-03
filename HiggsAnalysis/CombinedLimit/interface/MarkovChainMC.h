#ifndef HiggsAnalysis_CombinedLimit_MarkovChainMC_h
#define HiggsAnalysis_CombinedLimit_MarkovChainMC_h
/** \class MarkovChainMC
 *
 * abstract interface for physics objects
 *
 * \author Luca Lista (INFN), from initial implementation by Giovanni Petrucciani (UCSD)
 *
 *
 */
#include "HiggsAnalysis/CombinedLimit/interface/LimitAlgo.h"

class MarkovChainMC : public LimitAlgo {
public:
  MarkovChainMC() ;
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  bool run(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint);
  virtual const std::string & name() const {
    static const std::string name("MarkovChainMC");
    return name;
  }
private:
  enum ProposalType { FitP, UniformP, MultiGaussianP, TestP };
  static std::string proposalTypeName_;
  static ProposalType proposalType_;
  static bool runMinos_, noReset_, updateProposalParams_, updateHint_;
  /// Propose this number of points for the chain
  static unsigned int iterations_;
  /// Discard these points
  static unsigned int burnInSteps_;
  /// compute the limit N times
  static unsigned int tries_;
  /// Ignore up to this fraction of results if they're too far from the median
  static float truncatedMeanFraction_;
  /// do adaptive truncated mean
  static bool adaptiveTruncation_;
  /// Safety factor for hint (integrate up to this number of times the hinted limit)
  static float hintSafetyFactor_;
  static unsigned int numberOfBins_;
  static unsigned int proposalHelperCacheSize_;
  static float        proposalHelperWidthRangeDivisor_, proposalHelperUniformFraction_;
  static float        cropNSigmas_;
  static int          debugProposal_;
  // return number of items in chain, 0 for error
  int runOnce(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) const ;

  void limitAndError(double &limit, double &limitErr, std::vector<double> &limits) const ;
};

#endif
