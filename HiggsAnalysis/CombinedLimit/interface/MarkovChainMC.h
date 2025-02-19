#ifndef HiggsAnalysis_CombinedLimit_MarkovChainMC_h
#define HiggsAnalysis_CombinedLimit_MarkovChainMC_h
/** \class MarkovChainMC
 *
 * Interface to RooStats MCMC, with handing of multiple chains
 *
 * \author Giovanni Petrucciani (UCSD) 
 *
 *
 */
#include "../interface/LimitAlgo.h"
#include <TList.h>
class RooArgSet;
namespace RooStats { class MarkovChain; }

class MarkovChainMC : public LimitAlgo {
public:
  MarkovChainMC() ;
  virtual void applyOptions(const boost::program_options::variables_map &vm) ;
  bool run(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint);
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
  /// Discard these fraction of points
  static float burnInFraction_;
  /// Adaptive burn-in (experimental!)
  static bool adaptiveBurnIn_;
  /// compute the limit N times
  static unsigned int tries_;
  /// Ignore up to this fraction of results if they're too far from the median
  static float truncatedMeanFraction_;
  /// do adaptive truncated mean
  static bool adaptiveTruncation_;
  /// Safety factor for hint (integrate up to this number of times the hinted limit)
  static float hintSafetyFactor_;
  /// Save Markov Chain in output file
  static bool saveChain_;
  /// Leave all parameters in the markov chain, not just the POI 
  static bool noSlimChain_;
  /// Merge chains instead of averaging limits
  static bool mergeChains_; 
  /// Read chains from file instead of running them 
  static bool readChains_;
  /// Mass of the Higgs boson (goes into the name of the saved chains)
  float mass_;
  /// Number of degrees of freedom of the problem, approximately
  int   modelNDF_;

  static unsigned int numberOfBins_;
  static unsigned int proposalHelperCacheSize_;
  static bool         alwaysStepPoi_;
  static float        proposalHelperWidthRangeDivisor_, proposalHelperUniformFraction_;
  static float        cropNSigmas_;
  static int          debugProposal_;

  mutable TList chains_;

  // return number of items in chain, 0 for error
  int runOnce(RooWorkspace *w, RooStats::ModelConfig *mc_s, RooStats::ModelConfig *mc_b, RooAbsData &data, double &limit, double &limitErr, const double *hint) const ;

  RooStats::MarkovChain *mergeChains(const RooArgSet &poi, const std::vector<double> &limits) const;
  void readChains(const RooArgSet &poi, std::vector<double> &limits);
  void limitFromChain(double &limit, double &limitErr, const RooArgSet &poi, RooStats::MarkovChain &chain, int burnInSteps=-1 /* -1 = use default */) ;
  void limitAndError(double &limit, double &limitErr, const std::vector<double> &limits) const ;
  RooStats::MarkovChain *slimChain(const RooArgSet &poi, const RooStats::MarkovChain &chain) const;
  int  guessBurnInSteps(const RooStats::MarkovChain &chain) const;

  /// note: this is still being developed
  int stationarityTest(const RooStats::MarkovChain &chain, const RooArgSet &poi, int nchunks) const ;
};

#endif
