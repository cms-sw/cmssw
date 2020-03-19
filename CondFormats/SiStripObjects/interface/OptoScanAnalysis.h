#ifndef CondFormats_SiStripObjects_OptoScanAnalysis_H
#define CondFormats_SiStripObjects_OptoScanAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/** 
   @class OptoScanAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Histogram-based analysis for opto bias/gain scan.
*/
class OptoScanAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors ----------

  OptoScanAnalysis(const uint32_t& key);

  OptoScanAnalysis();

  ~OptoScanAnalysis() override { ; }

  friend class OptoScanAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

  /** Optimum LLD gain setting */
  inline const uint16_t& gain() const;

  /** LLD bias value for each gain setting */
  inline const VInt& bias() const;

  /** Measured gains for each setting [V/V]. */
  inline const VFloat& measGain() const;

  /** "Zero light" levels [ADC] */
  inline const VFloat& zeroLight() const;

  /** Noise value at "zero light" levels [ADC] */
  inline const VFloat& linkNoise() const;

  /** Baseline "lift-off" values [mA] */
  inline const VFloat& liftOff() const;

  /** Laser thresholds [mA] */
  inline const VFloat& threshold() const;

  /** Tick mark heights [ADC] */
  inline const VFloat& tickHeight() const;

  /** Baseline slope [ADC/I2C] */
  inline const VFloat& baseSlope() const;

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t gain_setting = sistrip::invalid_) override;

  /** Overrides base method. */
  void summary(std::stringstream&) const override;

  /** Resets analysis member data. */
  void reset() override;

  // ---------- public static data ----------

  /** Default LLD gain setting if analysis fails. */
  static const uint16_t defaultGainSetting_;

  /** Default LLD bias setting if analysis fails. */
  static const uint16_t defaultBiasSetting_;

  /** Peak-to-peak voltage for FED A/D converter [V/ADC]. */
  static const float fedAdcGain_;

  // ---------- private member data ----------

private:
  /** Optimum LLD gain setting */
  uint16_t gain_;

  /** LLD bias value for each gain setting */
  VInt bias_;

  /** Measured gains for each setting [V/V]. */
  VFloat measGain_;

  /** "Zero light" levels [ADC] */
  VFloat zeroLight_;

  /** Noise value at "zero light" levels [ADC] */
  VFloat linkNoise_;

  /** Baseline "lift-off" values [mA] */
  VFloat liftOff_;

  /** Laser thresholds [mA] */
  VFloat threshold_;

  /** Tick mark heights [ADC] */
  VFloat tickHeight_;

  /** Slope of baseline [ADC/I2C] */
  VFloat baseSlope_;
};

// ---------- Inline methods ----------

const uint16_t& OptoScanAnalysis::gain() const { return gain_; }
const OptoScanAnalysis::VInt& OptoScanAnalysis::bias() const { return bias_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::measGain() const { return measGain_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::zeroLight() const { return zeroLight_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::linkNoise() const { return linkNoise_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::liftOff() const { return liftOff_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::threshold() const { return threshold_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::tickHeight() const { return tickHeight_; }
const OptoScanAnalysis::VFloat& OptoScanAnalysis::baseSlope() const { return baseSlope_; }

#endif  // CondFormats_SiStripObjects_OptoScanAnalysis_H
