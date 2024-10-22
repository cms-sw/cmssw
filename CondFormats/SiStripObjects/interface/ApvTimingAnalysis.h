#ifndef CondFormats_SiStripObjects_ApvTimingAnalysis_H
#define CondFormats_SiStripObjects_ApvTimingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/**
   @class ApvTimingAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Analysis for timing run using APV tick marks.
*/
class ApvTimingAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors ----------

  ApvTimingAnalysis(const uint32_t& key);

  ApvTimingAnalysis();

  ~ApvTimingAnalysis() override { ; }

  friend class ApvTimingAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

  /** Identifies if tick mark is found or not. */
  bool foundTickMark() const;

  /** Time of tick mark rising edge [ns]. */
  inline const float& time() const;

  /** Error on time of tick mark rising edge [ns]. */
  inline const float& error() const;

  /** Optimum sampling point, defined w.r.t. rising edge [ns]. */
  inline const float& optimumSamplingPoint() const;

  /** Sampling point of "reference" tick mark [ns]. */
  inline const float& refTime() const;

  /** Static method to set global reference time [ns]. */
  void refTime(const float& time, const float& targetDelay = -1);

  /** Delay required to sync w.r.t. reference tick mark [ns]. */
  inline const float& delay() const;

  /** Height of tick mark [ADC]. */
  inline const float& height() const;

  /** Baseline level of tick mark [ADC]. */
  inline const float& base() const;

  /** Level of tick mark top [ADC]. */
  inline const float& peak() const;

  /** FED frame-finding threshold [ADC] (returns 65535 if invalid). */
  uint16_t frameFindingThreshold() const;

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t not_used = 0) override;

  /** Adds error codes for analysis (overrides private base). */
  inline void addErrorCode(const std::string& error) override;

  /** Resets analysis member data. */
  void reset() override;

  // ---------- public static data ----------

  /** Optimum sampling point, defined w.r.t. rising edge [ns]. */
  static const float optimumSamplingPoint_;

  /** Threshold defining minimum tick mark height [ADC]. */
  static const float tickMarkHeightThreshold_;

  /** Threshold for FED frame finding (fraction of tick height). */
  static const float frameFindingThreshold_;

  // ---------- private member data ----------

private:
  /** Time of tick mark rising edge [ns]. */
  float time_;

  /** Error on time of tick mark rising edge [ns]. */
  float error_;

  /** Sampling point of "reference" tick mark [ns]. */
  static float refTime_;

  /** Delay required to sync w.r.t. reference tick mark [ns]. */
  float delay_;

  /** Height of tick mark [ADC]. */
  float height_;

  /** Baseline level of tick mark [ADC]. */
  float base_;

  /** Level of tick mark top [ADC]. */
  float peak_;

  /** Checks synchronization to ref time is done only once. */
  bool synchronized_;
};

// ---------- Inline methods ----------

const float& ApvTimingAnalysis::time() const { return time_; }
const float& ApvTimingAnalysis::error() const { return error_; }
const float& ApvTimingAnalysis::optimumSamplingPoint() const { return optimumSamplingPoint_; }
const float& ApvTimingAnalysis::refTime() const { return refTime_; }
const float& ApvTimingAnalysis::delay() const { return delay_; }
const float& ApvTimingAnalysis::height() const { return height_; }
const float& ApvTimingAnalysis::base() const { return base_; }
const float& ApvTimingAnalysis::peak() const { return peak_; }
void ApvTimingAnalysis::addErrorCode(const std::string& error) { CommissioningAnalysis::addErrorCode(error); }

#endif  // CondFormats_SiStripObjects_ApvTimingAnalysis_H
