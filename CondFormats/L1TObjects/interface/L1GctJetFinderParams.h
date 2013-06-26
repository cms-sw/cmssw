#ifndef L1GCTJETFINDERPARAMS_H_
#define L1GCTJETFINDERPARAMS_H_

#include <vector>
#include <stdint.h>
#include <iosfwd>

class L1GctJetFinderParams
{
 public:

  static const unsigned NUMBER_ETA_VALUES;     ///< Number of eta bins used in correction
  static const unsigned N_CENTRAL_ETA_VALUES;     ///< Number of eta bins used in correction

  L1GctJetFinderParams();

  L1GctJetFinderParams(double rgnEtLsb,
		       double htLsb,
		       double cJetSeed,
                       double fJetSeed,
                       double tJetSeed,
		       double tauIsoEtThresh,
		       double htJetEtThresh,
		       double mhtJetEtThresh,
                       unsigned etaBoundary, 
		       unsigned corrType,
		       const std::vector< std::vector<double> >& jetCorrCoeffs,
		       const std::vector< std::vector<double> >& tauCorrCoeffs,
		       bool convertToEnergy,
		       const std::vector<double>& energyConvCoeffs);

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
  bool getConvertToEnergy() const { return convertToEnergy_; }

  // get integers
  unsigned getCenJetEtSeedGct() const { return static_cast<unsigned>(cenJetEtSeed_/rgnEtLsb_); }
  unsigned getForJetEtSeedGct() const { return static_cast<unsigned>(forJetEtSeed_/rgnEtLsb_); }
  unsigned getTauJetEtSeedGct() const { return static_cast<unsigned>(tauJetEtSeed_/rgnEtLsb_); }
  unsigned getTauIsoEtThresholdGct() const { return static_cast<unsigned>(tauIsoEtThreshold_/rgnEtLsb_); }
  unsigned getHtJetEtThresholdGct() const { return static_cast<unsigned>(htJetEtThreshold_/htLsb_); }
  unsigned getMHtJetEtThresholdGct() const { return static_cast<unsigned>(mhtJetEtThreshold_/htLsb_); }

  // set methods
  void setRegionEtLsb (const double rgnEtLsb);
  void setSlidingWindowParams(const double cJetSeed,
			      const double fJetSeed,
			      const double tJetSeed,
			      const unsigned etaBoundary);
  void setJetEtCalibrationParams(const unsigned corrType,
				 const std::vector< std::vector<double> >& jetCorrCoeffs,
				 const std::vector< std::vector<double> >& tauCorrCoeffs);
  void setJetEtConvertToEnergyOn(const std::vector<double>& energyConvCoeffs);
  void setJetEtConvertToEnergyOff();
  void setHtSumParams(const double htLsb,
		      const double htJetEtThresh,
		      const double mhtJetEtThresh);
  void setTauAlgorithmParams(const double tauIsoEtThresh);
  void setParams(const double rgnEtLsb,
		 const double htLsb,
		 const double cJetSeed,
		 const double fJetSeed,
		 const double tJetSeed,
		 const double tauIsoEtThresh,
		 const double htJetEtThresh,
		 const double mhtJetEtThresh,
		 const unsigned etaBoundary,
		 const unsigned corrType,
		 const std::vector< std::vector<double> >& jetCorrCoeffs,
		 const std::vector< std::vector<double> >& tauCorrCoeffs);

  // correct jet Et
  /// Eta takes a value from 0-10, corresponding to jet regions running from eta=0.0 to eta=5.0
  double correctedEtGeV(const double et, const unsigned eta, const bool tauVeto) const;
  
  /// Convert the corrected Et value to a linear Et for Ht summing
  uint16_t correctedEtGct(const double correctedEt) const;
  
  /// Access to jet Et calibration parameters
  unsigned getCorrType() const { return corrType_; }
  const std::vector< std::vector<double> >& getJetCorrCoeffs() const { return jetCorrCoeffs_; }
  const std::vector< std::vector<double> >& getTauCorrCoeffs() const { return tauCorrCoeffs_; }

 private:

  // correct the et
  double correctionFunction(const double Et, const std::vector<double>& coeffs) const;

  // different jet correction functions
  double findCorrectedEt       (const double Et, const std::vector<double>& coeffs) const;
  double powerSeriesCorrect    (const double Et, const std::vector<double>& coeffs) const;
  double orcaStyleCorrect      (const double Et, const std::vector<double>& coeffs) const;
  double simpleCorrect         (const double Et, const std::vector<double>& coeffs) const;
  double piecewiseCubicCorrect (const double Et, const std::vector<double>& coeffs) const;
  double pfCorrect             (const double Et, const std::vector<double>& coeffs) const;
  
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

  // jet Et corrections
  unsigned corrType_; 
  std::vector< std::vector<double> > jetCorrCoeffs_;
  std::vector< std::vector<double> > tauCorrCoeffs_;

  // convert Et to E
  bool convertToEnergy_;
  std::vector<double> energyConversionCoeffs_;

};

/// Overload << operator
std::ostream& operator << (std::ostream& os, const L1GctJetFinderParams& fn);

#endif /*L1GCTJETPARAMS_H_*/

