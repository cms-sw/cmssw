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
  double getRgnEtLsbGeV() { return rgnEtLsb_; }
  double getHtLsbGeV() { return htLsb_; }
  double getCenJetEtSeedGeV() { return cenJetEtSeed_; }
  double getForJetEtSeedGeV() { return forJetEtSeed_; }
  double getTauJetEtSeedGeV() { return tauJetEtSeed_; }
  double getTauIsoEtThresholdGeV() { return tauIsoEtThreshold_; }
  double getHtJetEtThresholdGeV() { return htJetEtThreshold_; }
  double getMHtJetEtThresholdGeV() { return mhtJetEtThreshold_; }
  unsigned getCenForJetEtaBoundary() { return cenForJetEtaBoundary_; }

  // get integers
  unsigned getCenJetEtSeedGct() { return static_cast<unsigned>(cenJetEtSeed_/rgnEtLsb_); }
  unsigned getForJetEtSeedGct() { return static_cast<unsigned>(forJetEtSeed_/rgnEtLsb_); }
  unsigned getTauJetEtSeedGct() { return static_cast<unsigned>(tauJetEtSeed_/rgnEtLsb_); }
  unsigned getTauIsoEtThresholdGct() { return static_cast<unsigned>(tauIsoEtThreshold_/rgnEtLsb_); }
  unsigned getHtJetEtThresholdGct() { return static_cast<unsigned>(htJetEtThreshold_/htLsb_); }
  unsigned getMHtJetEtThresholdGct() { return static_cast<unsigned>(mhtJetEtThreshold_/htLsb_); }


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

