#ifndef CondFormats_SiStripObjects_DaqScopeModeAnalysis_H
#define CondFormats_SiStripObjects_DaqScopeModeAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/**
   @class DaqScopeModeAnalysis
   @author R.Bainbridge
   @brief Analysis for scope mode data.
*/

class DaqScopeModeAnalysis : public CommissioningAnalysis {
public:
  DaqScopeModeAnalysis(const uint32_t& key);

  DaqScopeModeAnalysis();

  ~DaqScopeModeAnalysis() override { ; }

  friend class DaqScopeModeAlgorithm;

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;
  /** Identifies if tick mark is found or not. */
  bool foundTickMark() const;
  /** FED frame-finding threshold [ADC] (returns 65535 if invalid). */
  uint16_t frameFindingThreshold() const;

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

  /** Height of tick mark [ADC]. */
  inline const float& height() const;
  /** Baseline level of tick mark [ADC]. */
  inline const float& base() const;
  /** Level of tick mark top [ADC]. */
  inline const float& peak() const;

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t apv_number = 0) override;

  /** Adds error codes for analysis (overrides private base). */
  inline void addErrorCode(const std::string& error) override;

  /** Overrides base method. */
  void summary(std::stringstream&) const override;

  /** Resets analysis member data. */
  void reset() override;

  /** Threshold defining minimum tick mark height [ADC]. */
  static const float tickMarkHeightThreshold_;

  /** Threshold for FED frame finding (fraction of tick height). */
  static const float frameFindingThreshold_;

private:
  /** Height of tick mark [ADC]. */
  float height_;
  /** Baseline level of tick mark [ADC]. */
  float base_;
  /** Level of tick mark top [ADC]. */
  float peak_;

  /** Peds values. */
  VVFloat peds_;
  /** Noise values. */
  VVFloat noise_;
  /** Raw noise values. */
  VVFloat raw_;

  /** Dead strips. */
  VVInt dead_;
  /** Noisy strips. */
  VVInt noisy_;

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

const DaqScopeModeAnalysis::VVFloat& DaqScopeModeAnalysis::peds() const { return peds_; }
const DaqScopeModeAnalysis::VVFloat& DaqScopeModeAnalysis::noise() const { return noise_; }
const DaqScopeModeAnalysis::VVFloat& DaqScopeModeAnalysis::raw() const { return raw_; }

const DaqScopeModeAnalysis::VVInt& DaqScopeModeAnalysis::dead() const { return dead_; }
const DaqScopeModeAnalysis::VVInt& DaqScopeModeAnalysis::noisy() const { return noisy_; }

const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::pedsMean() const { return pedsMean_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::pedsSpread() const { return pedsSpread_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::noiseMean() const { return noiseMean_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::noiseSpread() const { return noiseSpread_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::rawMean() const { return rawMean_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::rawSpread() const { return rawSpread_; }

const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::pedsMax() const { return pedsMax_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::pedsMin() const { return pedsMin_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::noiseMax() const { return noiseMax_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::noiseMin() const { return noiseMin_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::rawMax() const { return rawMax_; }
const DaqScopeModeAnalysis::VFloat& DaqScopeModeAnalysis::rawMin() const { return rawMin_; }

const float& DaqScopeModeAnalysis::height() const { return height_; }
const float& DaqScopeModeAnalysis::base() const { return base_; }
const float& DaqScopeModeAnalysis::peak() const { return peak_; }
void DaqScopeModeAnalysis::addErrorCode(const std::string& error) { CommissioningAnalysis::addErrorCode(error); }

#endif  // CondFormats_SiStripObjects_DaqScopeModeAnalysis_H
