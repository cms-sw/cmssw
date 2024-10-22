#ifndef CondFormats_SiStripObjects_FastFedCablingAnalysis_H
#define CondFormats_SiStripObjects_FastFedCablingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <map>
#include <cstdint>

/** 
   @class FastFedCablingAnalysis
   @author R.Bainbridge
   @brief Histogram-based analysis for connection loop.
*/
class FastFedCablingAnalysis : public CommissioningAnalysis {
public:
  // ---------- con(de)structors and typedefs ----------

  FastFedCablingAnalysis(const uint32_t& key);

  FastFedCablingAnalysis();

  ~FastFedCablingAnalysis() override { ; }

  typedef std::map<uint32_t, uint16_t> Candidates;

  friend class FastFedCablingAlgorithm;

  // ---------- public interface ----------

  /** Identifies if analysis is valid or not. */
  bool isValid() const override;

  /** Identifies if fibre is dirty or not. */
  bool isDirty() const;

  /** Identifies if TrimDAQ setting is valid or not. */
  bool badTrimDac() const;

  /** DCU hardware id (32-bits). */
  inline const uint32_t& dcuHardId() const;

  /** Linear Laser Driver channel. */
  inline const uint16_t& lldCh() const;

  /** "High" light level [ADC]. */
  inline const float& highLevel() const;

  /** Spread in "high" ligh level [ADC]. */
  inline const float& highRms() const;

  /** "Low" light level [ADC]. */
  inline const float& lowLevel() const;

  /** Spread in "low" ligh level [ADC]. */
  inline const float& lowRms() const;

  /** Maximum light level in data [ADC]. */
  inline const float& max() const;

  /** Minimum light level in data [ADC]. */
  inline const float& min() const;

  // ---------- misc ----------

  /** Prints analysis results. */
  void print(std::stringstream&, uint32_t not_used = 0) override;

  /** Header information for analysis print(). */
  void header(std::stringstream&) const override;

  /** Overrides base method. */
  void summary(std::stringstream&) const override;

  /** Resets analysis member data. */
  void reset() override;

  // ---------- public static data ----------

public:
  /** Threshold to identify digital high from digital low. */
  static const float threshold_;

  /** Level [ADC] below which fibre is defined as "dirty". */
  static const float dirtyThreshold_;

  /** Level [ADC] below which TrimDAC setting is defined as "bad". */
  static const float trimDacThreshold_;

  /** */
  static const uint16_t nBitsForDcuId_;

  /** */
  static const uint16_t nBitsForLldCh_;

  // ---------- private member data ----------

private:
  /** Extracted DCU id. */
  uint32_t dcuHardId_;

  /** Extracted LLD channel. */
  uint16_t lldCh_;

  /** */
  float highMedian_;

  /** */
  float highMean_;

  /** */
  float highRms_;

  /** */
  float lowMedian_;

  /** */
  float lowMean_;

  /** */
  float lowRms_;

  /** */
  float range_;

  /** */
  float midRange_;

  /** */
  float max_;

  /** */
  float min_;
};

// ---------- Inline methods ----------

const uint32_t& FastFedCablingAnalysis::dcuHardId() const { return dcuHardId_; }
const uint16_t& FastFedCablingAnalysis::lldCh() const { return lldCh_; }
const float& FastFedCablingAnalysis::highLevel() const { return highMean_; }
const float& FastFedCablingAnalysis::highRms() const { return highRms_; }
const float& FastFedCablingAnalysis::lowLevel() const { return lowMean_; }
const float& FastFedCablingAnalysis::lowRms() const { return lowRms_; }
const float& FastFedCablingAnalysis::max() const { return max_; }
const float& FastFedCablingAnalysis::min() const { return min_; }

#endif  // CondFormats_SiStripObjects_FastFedCablingAnalysis_H
