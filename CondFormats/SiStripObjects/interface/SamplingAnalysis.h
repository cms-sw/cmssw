#ifndef CondFormats_SiStripObjects_SamplingAnalysis_H
#define CondFormats_SiStripObjects_SamplingAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include <sstream>
#include <vector>
#include <cstdint>

/**
   @class SamplingAnalysis
   @author C. Delaere
   @brief Analysis for latency run
*/

class SamplingAnalysis : public CommissioningAnalysis {
public:
  SamplingAnalysis(const uint32_t& key);

  SamplingAnalysis();

  ~SamplingAnalysis() override { ; }

  friend class SamplingAlgorithm;

  inline const float& maximum() const { return max_; }

  inline const float& error() const { return error_; }

  inline void setSoNcut(const float sOnCut) { sOnCut_ = sOnCut; }

  float getSoNcut() const { return sOnCut_; }

  void print(std::stringstream&, uint32_t not_used = 0) override;

  void reset() override;

  float limit(float SoNcut) const;

  float correctMeasurement(float mean, float SoNcut = 3.) const;

  sistrip::Granularity granularity() const { return granularity_; }

private:
  /** s/n cut to be used */
  float sOnCut_;

  /** Delay corresponding to the maximum of the pulse shape */
  float max_;

  /** Error on the position ( from the fit) */
  float error_;

  /** reconstruction mode */
  sistrip::RunType runType_;

  /** granularity */
  sistrip::Granularity granularity_;
};

#endif  // CondFormats_SiStripObjects_SamplingAnalysis_H
