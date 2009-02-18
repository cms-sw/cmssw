#ifndef L1GCTJETFINDERPARAMS_H_
#define L1GCTJETFINDERPARAMS_H_

class L1GctJetFinderParams
{
 public:

  L1GctJetFinderParams(double rgnEtLsb,
		       double htLsb,
		       double cJetSeed,
                       double fJetSeed,
                       double tJetSeed,
		       double tauIsoEtThresh,
		       double htJetEtThresh,
		       double mhtJetEtThresh,
                       unsigned etaBoundary);

  ~L1GctJetFinderParams();

  // get methods
  double getRgnEtLsb() { return rgnEtLsb_; }
  double getHtLsb() { return htLsb_; }
  double getCenJetEtSeed() { return cenJetEtSeed_; }
  double getForJetEtSeed() { return forJetEtSeed_; }
  double getTauJetEtSeed() { return tauJetEtSeed_; }
  double getTauIsoEtThreshold() { return tauIsoEtThreshold_; }
  double getHtJetEtThreshold() { return htJetEtThreshold_; }
  double getMHtJetEtThreshold() { return mhtJetEtThreshold_; }
  unsigned getCenForJetEtaBoundary() { return cenForJetEtaBoundary_; }

  // get integers
  unsigned getCenJetSeedRank() { return static_cast<unsigned>(cenJetSeed_/rgnEtLsb_); }
  unsigned getForJetSeedRank() { return static_cast<unsigned>(forJetSeed_/rgnEtLsb_); }
  unsigned getTauJetSeedRank() { return static_cast<unsigned>(tauJetSeed_/rgnEtLsb_); }
  unsigned getTauIsoThreshRank() { return static_cast<unsigned>(tauIsoEtThreshold_/rgnEtLsb_); }
  unsigned getHtJetThreshRank() { return static_cast<unsigned>(htJetEtThreshold_/htLsb_); }
  unsigned getMHtJetThreshRank() { return static_cast<unsigned>(mhtJetEtThreshold_/htLsb_); }


 private:

  // internal scale LSBs
  double rgnEtLsb_;
  double htLsb_;

  // parameters
  double cenJetEtSeed_;
  double forJetEtSeed_;
  double tauJetEtSeed_;
  double tauIsoEtThreshold_;
  double htJetEtThreshold_;
  double mhtJetEtThreshold_;
  unsigned cenForJetEtaBoundary_;

};

#endif /*L1GCTJETPARAMS_H_*/

