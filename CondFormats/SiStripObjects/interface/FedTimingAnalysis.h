#ifndef CondFormats_SiStripObjects_FedTimingAnalysis_H
#define CondFormats_SiStripObjects_FedTimingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/**
   @class FedTimingAnalysis
   @author M. Wingham, R.Bainbridge
   @brief Analysis for timing run using APV tick marks.
*/

class FedTimingAnalysis : public CommissioningAnalysis {
public:
  FedTimingAnalysis(const uint32_t& key);

  FedTimingAnalysis();

  ~FedTimingAnalysis() override { ; }

  friend class FedTimingAlgorithm;

  inline const float& time() const;

  inline const float& max() const;

  inline const float& delay() const;

  inline const float& error() const;

  inline const float& base() const;

  inline const float& peak() const;

  inline const float& height() const;

  void max(const float&);

  void print(std::stringstream&, uint32_t not_used = 0) override;

  void reset() override;

private:
  /** Time of tick mark rising edge [ns] */
  float time_;

  /** Maximum time set [ns] */
  float max_;

  /** Delay required, relative to maximum time [ns] */
  float delay_;

  /** Error on time delay [ns] */
  float error_;

  /** Level of tick mark "base" [adc] */
  float base_;

  /** Level of tick mark "peak" [adc] */
  float peak_;

  /** Tick mark height [adc] */
  float height_;

  /** */
  float optimumSamplingPoint_;
};

const float& FedTimingAnalysis::time() const { return time_; }
const float& FedTimingAnalysis::max() const { return max_; }
const float& FedTimingAnalysis::delay() const { return delay_; }
const float& FedTimingAnalysis::error() const { return error_; }
const float& FedTimingAnalysis::base() const { return base_; }
const float& FedTimingAnalysis::peak() const { return peak_; }
const float& FedTimingAnalysis::height() const { return height_; }

#endif  // CondFormats_SiStripObjects_FedTimingAnalysis_H
