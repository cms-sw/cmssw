#ifndef CondFormats_SiStripObjects_PedestalsAnalysis_H
#define CondFormats_SiStripObjects_PedestalsAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/** 
    @class PedestalsAnalysis
    @author M. Wingham, R.Bainbridge
    @brief Histogram-based analysis for pedestal run.
*/
class PedestalsAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors ----------

  PedestalsAnalysis(const uint32_t& key);

  PedestalsAnalysis();

  ~PedestalsAnalysis() override { ; }

  friend class PedestalsAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

  // Pedestal, noise and raw noise (128-strip vector per APV)
  inline const VVFloat& peds() const;
  inline const VVFloat& noise() const;
  inline const VVFloat& raw() const;

  // Dead and noisy strips (vector per APV)
  inline const VVInt& dead() const;
  inline const VVInt& noisy() const;

  // Mean and rms spread (value per APV)
  inline const VFloat& pedsMean() const;
  inline const VFloat& pedsSpread() const;
  inline const VFloat& noiseMean() const;
  inline const VFloat& noiseSpread() const;
  inline const VFloat& rawMean() const;
  inline const VFloat& rawSpread() const;

  // Max and min values (value per APV)
  inline const VFloat& pedsMax() const;
  inline const VFloat& pedsMin() const;
  inline const VFloat& noiseMax() const;
  inline const VFloat& noiseMin() const;
  inline const VFloat& rawMax() const;
  inline const VFloat& rawMin() const;

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t apv_number = 0) override;

  /** Overrides base method. */
  void summary(std::stringstream&) const override;

  /** Resets analysis member data. */
  void reset() override;

  // ---------- private member data ----------

private:
  // VVFloats means: 1 vector per APV, 1 value per strip.

  /** Peds values. */
  VVFloat peds_;

  /** Noise values. */
  VVFloat noise_;

  /** Raw noise values. */
  VVFloat raw_;

  // VVInts means: 1 vector per APV, values are strip numbers.

  /** Dead strips. */
  VVInt dead_;

  /** Noisy strips. */
  VVInt noisy_;

  // VFloat: 1 value per APV

  /** Mean peds value. */
  VFloat pedsMean_;

  /** Rms spread in peds. */
  VFloat pedsSpread_;

  /** Mean noise value. */
  VFloat noiseMean_;

  /** Rms spread in noise. */
  VFloat noiseSpread_;

  /** Mean raw noise value. */
  VFloat rawMean_;

  /** Rms spread in raw noise. */
  VFloat rawSpread_;

  /** Max peds value. */
  VFloat pedsMax_;

  /** Min peds value. */
  VFloat pedsMin_;

  /** Max noise value. */
  VFloat noiseMax_;

  /** Min noise value. */
  VFloat noiseMin_;

  /** Max raw noise value. */
  VFloat rawMax_;

  /** Min raw noise value. */
  VFloat rawMin_;

  // true if legacy histogram naming is used
  bool legacy_;
};

// ---------- Inline methods ----------

const PedestalsAnalysis::VVFloat& PedestalsAnalysis::peds() const { return peds_; }
const PedestalsAnalysis::VVFloat& PedestalsAnalysis::noise() const { return noise_; }
const PedestalsAnalysis::VVFloat& PedestalsAnalysis::raw() const { return raw_; }

const PedestalsAnalysis::VVInt& PedestalsAnalysis::dead() const { return dead_; }
const PedestalsAnalysis::VVInt& PedestalsAnalysis::noisy() const { return noisy_; }

const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMean() const { return pedsMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsSpread() const { return pedsSpread_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMean() const { return noiseMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseSpread() const { return noiseSpread_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMean() const { return rawMean_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawSpread() const { return rawSpread_; }

const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMax() const { return pedsMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::pedsMin() const { return pedsMin_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMax() const { return noiseMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::noiseMin() const { return noiseMin_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMax() const { return rawMax_; }
const PedestalsAnalysis::VFloat& PedestalsAnalysis::rawMin() const { return rawMin_; }

#endif  // CondFormats_SiStripObjects_PedestalsAnalysis_H
