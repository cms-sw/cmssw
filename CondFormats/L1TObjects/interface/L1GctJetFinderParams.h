#ifndef L1GCTJETFINDERPARAMS_H_
#define L1GCTJETFINDERPARAMS_H_

class L1GctJetFinderParams
{
 public:

  L1GctJetFinderParams();

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
  double getRgnEtLsbGeV() const { return rgnEtLsb_; }
  double getHtLsbGeV() const { return htLsb_; }
  double getCenJetEtSeedGeV() const { return cenJetEtSeed_; }
  double getForJetEtSeedGeV() const { return forJetEtSeed_; }
  double getTauJetEtSeedGeV() const { return tauJetEtSeed_; }
  double getTauIsoEtThresholdGeV() const { return tauIsoEtThreshold_; }
  double getHtJetEtThresholdGeV() const { return htJetEtThreshold_; }
  double getMHtJetEtThresholdGeV() const { return mhtJetEtThreshold_; }
  unsigned getCenForJetEtaBoundary() const { return cenForJetEtaBoundary_; }

  // get integers
  unsigned getCenJetEtSeedGct() const { return static_cast<unsigned>(cenJetEtSeed_/rgnEtLsb_); }
  unsigned getForJetEtSeedGct() const { return static_cast<unsigned>(forJetEtSeed_/rgnEtLsb_); }
  unsigned getTauJetEtSeedGct() const { return static_cast<unsigned>(tauJetEtSeed_/rgnEtLsb_); }
  unsigned getTauIsoEtThresholdGct() const { return static_cast<unsigned>(tauIsoEtThreshold_/rgnEtLsb_); }
  unsigned getHtJetEtThresholdGct() const { return static_cast<unsigned>(htJetEtThreshold_/htLsb_); }
  unsigned getMHtJetEtThresholdGct() const { return static_cast<unsigned>(mhtJetEtThreshold_/htLsb_); }


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

