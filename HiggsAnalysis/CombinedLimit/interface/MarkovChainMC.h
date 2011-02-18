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
  std::string proposalTypeName_;
  ProposalType proposalType_;
  bool runMinos_, noReset_, updateProposalParams_, updateHint_;
  /// Propose this number of points for the chain
  unsigned int iterations_;
  /// Discard these points
  unsigned int burnInSteps_;
  /// compute the limit N times
  unsigned int tries_;
  /// Ignore up to this fraction of results if they're too far from the median
  float maxOutlierFraction_;
  /// Safety factor for hint (integrate up to this number of times the hinted limit)
  float hintSafetyFactor_;
  unsigned int numberOfBins_;
  unsigned int proposalHelperCacheSize_;
  float        proposalHelperWidthRangeDivisor_, proposalHelperUniformFraction_;
  float        cropNSigmas_;
  int          debugProposal_;
  // return number of items in chain, 0 for error
  int runOnce(RooWorkspace *w, RooAbsData &data, double &limit, const double *hint) const ;
};

#endif
