#ifndef CondFormats_SiStripObjects_CalibrationAnalysis_H
#define CondFormats_SiStripObjects_CalibrationAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <cstdint>

/**
   @class CalibrationAnalysis
   @author C. Delaere
   @brief Analysis for calibration runs
*/

class CalibrationAnalysis : public CommissioningAnalysis {
public:
  CalibrationAnalysis(const uint32_t& key, const bool& deconv);
  CalibrationAnalysis(const bool& deconv);

  ~CalibrationAnalysis() override { ; }

  friend class CalibrationAlgorithm;

  // values per strip and per APV
  inline const VVFloat& amplitude() { return amplitude_; }
  inline const VVFloat& tail() { return tail_; }
  inline const VVFloat& riseTime() { return riseTime_; }
  inline const VVFloat& decayTime() { return decayTime_; }
  inline const VVFloat& turnOn() { return turnOn_; }
  inline const VVFloat& peakTime() { return peakTime_; }
  inline const VVFloat& undershoot() { return undershoot_; }
  inline const VVFloat& baseline() { return baseline_; }
  inline const VVFloat& smearing() { return smearing_; }
  inline const VVFloat& chi2() { return chi2_; }

  inline const VVBool isValidStrip() { return isvalid_; }  // analysis validity
  bool isValid() const override;

  // mean values per APV
  inline const VFloat& amplitudeMean() { return mean_amplitude_; }
  inline const VFloat& tailMean() { return mean_tail_; }
  inline const VFloat& riseTimeMean() { return mean_riseTime_; }
  inline const VFloat& decayTimeMean() { return mean_decayTime_; }
  inline const VFloat& smearingMean() { return mean_smearing_; }
  inline const VFloat& turnOnMean() { return mean_turnOn_; }
  inline const VFloat& peakTimeMean() { return mean_peakTime_; }
  inline const VFloat& undershootMean() { return mean_undershoot_; }
  inline const VFloat& baselineMean() { return mean_baseline_; }
  inline const VFloat& chi2Mean() { return mean_chi2_; }

  // spread, min and max
  inline const VFloat& amplitudeSpread() { return spread_amplitude_; }
  inline const VFloat& amplitudeMin() { return min_amplitude_; }
  inline const VFloat& amplitudeMax() { return max_amplitude_; }

  inline const VFloat& tailSpread() { return spread_tail_; }
  inline const VFloat& tailMin() { return min_tail_; }
  inline const VFloat& tailMax() { return max_tail_; }

  inline const VFloat& riseTimeSpread() { return spread_riseTime_; }
  inline const VFloat& riseTimeMin() { return min_riseTime_; }
  inline const VFloat& riseTimeMax() { return max_riseTime_; }

  inline const VFloat& decayTimeSpread() { return spread_decayTime_; }
  inline const VFloat& decayTimeMin() { return min_decayTime_; }
  inline const VFloat& decayTimeMax() { return max_decayTime_; }

  inline const VFloat& smearingSpread() { return spread_smearing_; }
  inline const VFloat& smearingMin() { return min_smearing_; }
  inline const VFloat& smearingMax() { return max_smearing_; }

  inline const VFloat& turnOnSpread() { return spread_turnOn_; }
  inline const VFloat& turnOnMin() { return min_turnOn_; }
  inline const VFloat& turnOnMax() { return max_turnOn_; }

  inline const VFloat& peakTimeSpread() { return spread_peakTime_; }
  inline const VFloat& peakTimeMin() { return min_peakTime_; }
  inline const VFloat& peakTimeMax() { return max_peakTime_; }

  inline const VFloat& undershootSpread() { return spread_undershoot_; }
  inline const VFloat& undershootMin() { return min_undershoot_; }
  inline const VFloat& undershootMax() { return max_undershoot_; }

  inline const VFloat& baselineSpread() { return spread_baseline_; }
  inline const VFloat& baselineMin() { return min_baseline_; }
  inline const VFloat& baselineMax() { return max_baseline_; }

  inline const VFloat& chi2Spread() { return spread_chi2_; }
  inline const VFloat& chi2Min() { return min_chi2_; }
  inline const VFloat& chi2Max() { return max_chi2_; }

  inline int calChan() { return calChan_; }
  inline bool deconvMode() { return deconv_; }

  void print(std::stringstream&, uint32_t not_used = 0) override;
  void reset() override;

private:
  /** Values for quality cuts */
  static const float minAmplitudeThreshold_;
  static const float minBaselineThreshold_;
  static const float maxBaselineThreshold_;
  static const float maxChi2Threshold_;

  static const float minDecayTimeThreshold_;
  static const float maxDecayTimeThreshold_;
  static const float minPeakTimeThreshold_;
  static const float maxPeakTimeThreshold_;
  static const float minRiseTimeThreshold_;
  static const float maxRiseTimeThreshold_;
  static const float minTurnOnThreshold_;
  static const float maxTurnOnThreshold_;

  static const float minDecayTimeThresholdDeco_;
  static const float maxDecayTimeThresholdDeco_;
  static const float minPeakTimeThresholdDeco_;
  static const float maxPeakTimeThresholdDeco_;
  static const float minRiseTimeThresholdDeco_;
  static const float maxRiseTimeThresholdDeco_;
  static const float minTurnOnThresholdDeco_;
  static const float maxTurnOnThresholdDeco_;

private:
  /** Parameters extracted from the fit of pulse shape */
  VVFloat amplitude_, tail_, riseTime_, decayTime_, turnOn_, peakTime_, undershoot_, baseline_, smearing_, chi2_;
  VVBool isvalid_;
  VFloat mean_amplitude_, mean_tail_, mean_riseTime_, mean_decayTime_, mean_turnOn_, mean_peakTime_, mean_undershoot_,
      mean_baseline_, mean_smearing_, mean_chi2_;
  VFloat min_amplitude_, min_tail_, min_riseTime_, min_decayTime_, min_turnOn_, min_peakTime_, min_undershoot_,
      min_baseline_, min_smearing_, min_chi2_;
  VFloat max_amplitude_, max_tail_, max_riseTime_, max_decayTime_, max_turnOn_, max_peakTime_, max_undershoot_,
      max_baseline_, max_smearing_, max_chi2_;
  VFloat spread_amplitude_, spread_tail_, spread_riseTime_, spread_decayTime_, spread_turnOn_, spread_peakTime_,
      spread_undershoot_, spread_baseline_, spread_smearing_, spread_chi2_;

  /** fit mode: deconv or not ? */
  bool deconv_;

  /** calchan value */
  int calChan_;
};

#endif  // CondFormats_SiStripObjects_CalibrationAnalysis_H
