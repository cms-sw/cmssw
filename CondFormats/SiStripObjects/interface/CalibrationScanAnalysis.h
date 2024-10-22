#ifndef CondFormats_SiStripObjects_CalibrationScanAnalysis_H
#define CondFormats_SiStripObjects_CalibrationScanAnalysis_H

#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>
#include <map>
#include "TGraph.h"
#include "TGraph2D.h"
#include <cstdint>

/**
   @class CalibrationScanAnalysis
   @author R. Gerosa
   @brief Analysis for calibration scans
*/

class CalibrationScanAnalysis : public CommissioningAnalysis {
public:
  CalibrationScanAnalysis(const uint32_t& key, const bool& deconv);
  CalibrationScanAnalysis(const bool& deconv);

  ~CalibrationScanAnalysis() override { ; }

  void addOneCalibrationPoint(const std::string& key);

  friend class CalibrationScanAlgorithm;

  inline const VBool isValid(const std::string& key) { return isvalid_[key]; }  // analysis validity
  bool isValid() const override;

  inline const VFloat& amplitude(const std::string& key) {
    return amplitude_[key];
  }  // key stands for isha_%d_vfs_%d values
  inline const VFloat& tail(const std::string& key) { return tail_[key]; }
  inline const VFloat& riseTime(const std::string& key) { return riseTime_[key]; }
  inline const VFloat& decayTime(const std::string& key) { return decayTime_[key]; }
  inline const VFloat& turnOn(const std::string& key) { return turnOn_[key]; }
  inline const VFloat& peakTime(const std::string& key) { return peakTime_[key]; }
  inline const VFloat& undershoot(const std::string& key) { return undershoot_[key]; }
  inline const VFloat& baseline(const std::string& key) { return baseline_[key]; }
  inline const VFloat& smearing(const std::string& key) { return smearing_[key]; }
  inline const VFloat& chi2(const std::string& key) { return chi2_[key]; }

  inline const VFloat& tunedAmplitude() { return tunedAmplitude_; }
  inline const VFloat& tunedTail() { return tunedTail_; }
  inline const VFloat& tunedRiseTime() { return tunedRiseTime_; }
  inline const VFloat& tunedDecayTime() { return tunedDecayTime_; }
  inline const VFloat& tunedTurnOn() { return tunedTurnOn_; }
  inline const VFloat& tunedPeakTime() { return tunedPeakTime_; }
  inline const VFloat& tunedUndershoot() { return tunedUndershoot_; }
  inline const VFloat& tunedBaseline() { return tunedBaseline_; }
  inline const VFloat& tunedSmearing() { return tunedSmearing_; }
  inline const VFloat& tunedChi2() { return tunedChi2_; }

  inline const VInt& tunedISHA() { return tunedISHA_; }
  inline const VInt& tunedVFS() { return tunedVFS_; }

  inline const std::vector<TGraph*>& decayTimeVsVFS() { return decayTime_vs_vfs_; }
  inline const std::vector<TGraph*>& riseTimeVsISHA() { return riseTime_vs_isha_; }
  inline const std::vector<TGraph2D*>& decayTimeVsISHAVsVFS() { return decayTime_vs_isha_vfs_; }
  inline const std::vector<TGraph2D*>& riseTimeVsISHAVsVFS() { return riseTime_vs_isha_vfs_; }

  inline const VInt& bestISHA() { return isha_; }
  inline const VInt& bestVFS() { return vfs_; }

  inline bool deconvMode() { return deconv_; }

  void print(std::stringstream&, uint32_t not_used = 0) override;
  void reset() override;

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
  static const float minISHAforVFSTune_;
  static const float maxISHAforVFSTune_;
  static const float VFSrange_;

private:
  /** Parameters extracted from the fit of pulse shape */
  std::map<std::string, VFloat> amplitude_;
  std::map<std::string, VFloat> tail_;
  std::map<std::string, VFloat> riseTime_;
  std::map<std::string, VFloat> decayTime_;
  std::map<std::string, VFloat> turnOn_;
  std::map<std::string, VFloat> peakTime_;
  std::map<std::string, VFloat> undershoot_;
  std::map<std::string, VFloat> baseline_;
  std::map<std::string, VFloat> smearing_;
  std::map<std::string, VFloat> chi2_;
  std::map<std::string, VBool> isvalid_;

  bool deconv_;

  /** Best isha and vfs values --> one per APV --> interpolate linearly allows a better evaluation compared to the point scanned */
  std::vector<TGraph*> decayTime_vs_vfs_;
  std::vector<TGraph*> riseTime_vs_isha_;
  std::vector<TGraph2D*> riseTime_vs_isha_vfs_;
  std::vector<TGraph2D*> decayTime_vs_isha_vfs_;

  VInt isha_;
  VInt vfs_;

  /** properties of pulse shapes closes to the optimal ISHA and VFS values */
  VFloat tunedAmplitude_, tunedTail_;
  VFloat tunedRiseTime_, tunedDecayTime_;
  VFloat tunedTurnOn_, tunedPeakTime_;
  VFloat tunedUndershoot_, tunedBaseline_;
  VFloat tunedSmearing_, tunedChi2_;
  VInt tunedISHA_, tunedVFS_;
};

#endif  // CondFormats_SiStripObjects_CalibrationScanAnalysis_H
