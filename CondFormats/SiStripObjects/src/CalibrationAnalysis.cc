#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

const float CalibrationAnalysis::minAmplitudeThreshold_ = 50;
const float CalibrationAnalysis::minBaselineThreshold_ = -50;
const float CalibrationAnalysis::maxBaselineThreshold_ = 50;
const float CalibrationAnalysis::maxChi2Threshold_ = 3;
const float CalibrationAnalysis::minDecayTimeThreshold_ = 30;
const float CalibrationAnalysis::maxDecayTimeThreshold_ = 250;
const float CalibrationAnalysis::minPeakTimeThreshold_ = 40;
const float CalibrationAnalysis::maxPeakTimeThreshold_ = 130;
const float CalibrationAnalysis::minRiseTimeThreshold_ = 10;
const float CalibrationAnalysis::maxRiseTimeThreshold_ = 100;
const float CalibrationAnalysis::minTurnOnThreshold_ = 5;
const float CalibrationAnalysis::maxTurnOnThreshold_ = 40;

const float CalibrationAnalysis::minDecayTimeThresholdDeco_ = 10;
const float CalibrationAnalysis::maxDecayTimeThresholdDeco_ = 100;
const float CalibrationAnalysis::minPeakTimeThresholdDeco_ = 40;
const float CalibrationAnalysis::maxPeakTimeThresholdDeco_ = 130;
const float CalibrationAnalysis::minRiseTimeThresholdDeco_ = 10;
const float CalibrationAnalysis::maxRiseTimeThresholdDeco_ = 100;
const float CalibrationAnalysis::minTurnOnThresholdDeco_ = 10;
const float CalibrationAnalysis::maxTurnOnThresholdDeco_ = 80;

CalibrationAnalysis::CalibrationAnalysis(const uint32_t& key, const bool& deconv)
    : CommissioningAnalysis(key, "CalibrationAnalysis"),
      amplitude_(2, VFloat(128, sistrip::invalid_)),
      tail_(2, VFloat(128, sistrip::invalid_)),
      riseTime_(2, VFloat(128, sistrip::invalid_)),
      decayTime_(2, VFloat(128, sistrip::invalid_)),
      turnOn_(2, VFloat(128, sistrip::invalid_)),
      peakTime_(2, VFloat(128, sistrip::invalid_)),
      undershoot_(2, VFloat(128, sistrip::invalid_)),
      baseline_(2, VFloat(128, sistrip::invalid_)),
      smearing_(2, VFloat(128, sistrip::invalid_)),
      chi2_(2, VFloat(128, sistrip::invalid_)),
      isvalid_(2, VBool(128, sistrip::invalid_)),
      mean_amplitude_(2, sistrip::invalid_),
      mean_tail_(2, sistrip::invalid_),
      mean_riseTime_(2, sistrip::invalid_),
      mean_decayTime_(2, sistrip::invalid_),
      mean_turnOn_(2, sistrip::invalid_),
      mean_peakTime_(2, sistrip::invalid_),
      mean_undershoot_(2, sistrip::invalid_),
      mean_baseline_(2, sistrip::invalid_),
      mean_smearing_(2, sistrip::invalid_),
      mean_chi2_(2, sistrip::invalid_),
      min_amplitude_(2, sistrip::invalid_),
      min_tail_(2, sistrip::invalid_),
      min_riseTime_(2, sistrip::invalid_),
      min_decayTime_(2, sistrip::invalid_),
      min_turnOn_(2, sistrip::invalid_),
      min_peakTime_(2, sistrip::invalid_),
      min_undershoot_(2, sistrip::invalid_),
      min_baseline_(2, sistrip::invalid_),
      min_smearing_(2, sistrip::invalid_),
      min_chi2_(2, sistrip::invalid_),
      max_amplitude_(2, sistrip::invalid_),
      max_tail_(2, sistrip::invalid_),
      max_riseTime_(2, sistrip::invalid_),
      max_decayTime_(2, sistrip::invalid_),
      max_turnOn_(2, sistrip::invalid_),
      max_peakTime_(2, sistrip::invalid_),
      max_undershoot_(2, sistrip::invalid_),
      max_baseline_(2, sistrip::invalid_),
      max_smearing_(2, sistrip::invalid_),
      max_chi2_(2, sistrip::invalid_),
      spread_amplitude_(2, sistrip::invalid_),
      spread_tail_(2, sistrip::invalid_),
      spread_riseTime_(2, sistrip::invalid_),
      spread_decayTime_(2, sistrip::invalid_),
      spread_turnOn_(2, sistrip::invalid_),
      spread_peakTime_(2, sistrip::invalid_),
      spread_undershoot_(2, sistrip::invalid_),
      spread_baseline_(2, sistrip::invalid_),
      spread_smearing_(2, sistrip::invalid_),
      spread_chi2_(2, sistrip::invalid_),
      deconv_(deconv),
      calChan_(0) {
  ;
}

// ----------------------------------------------------------------------------
//
CalibrationAnalysis::CalibrationAnalysis(const bool& deconv)
    : CommissioningAnalysis("CalibrationAnalysis"),
      amplitude_(2, VFloat(128, sistrip::invalid_)),
      tail_(2, VFloat(128, sistrip::invalid_)),
      riseTime_(2, VFloat(128, sistrip::invalid_)),
      decayTime_(2, VFloat(128, sistrip::invalid_)),
      turnOn_(2, VFloat(128, sistrip::invalid_)),
      peakTime_(2, VFloat(128, sistrip::invalid_)),
      undershoot_(2, VFloat(128, sistrip::invalid_)),
      baseline_(2, VFloat(128, sistrip::invalid_)),
      smearing_(2, VFloat(128, sistrip::invalid_)),
      chi2_(2, VFloat(128, sistrip::invalid_)),
      isvalid_(2, VBool(128, sistrip::invalid_)),
      mean_amplitude_(2, sistrip::invalid_),
      mean_tail_(2, sistrip::invalid_),
      mean_riseTime_(2, sistrip::invalid_),
      mean_decayTime_(2, sistrip::invalid_),
      mean_turnOn_(2, sistrip::invalid_),
      mean_peakTime_(2, sistrip::invalid_),
      mean_undershoot_(2, sistrip::invalid_),
      mean_baseline_(2, sistrip::invalid_),
      mean_smearing_(2, sistrip::invalid_),
      mean_chi2_(2, sistrip::invalid_),
      min_amplitude_(2, sistrip::invalid_),
      min_tail_(2, sistrip::invalid_),
      min_riseTime_(2, sistrip::invalid_),
      min_decayTime_(2, sistrip::invalid_),
      min_turnOn_(2, sistrip::invalid_),
      min_peakTime_(2, sistrip::invalid_),
      min_undershoot_(2, sistrip::invalid_),
      min_baseline_(2, sistrip::invalid_),
      min_smearing_(2, sistrip::invalid_),
      min_chi2_(2, sistrip::invalid_),
      max_amplitude_(2, sistrip::invalid_),
      max_tail_(2, sistrip::invalid_),
      max_riseTime_(2, sistrip::invalid_),
      max_decayTime_(2, sistrip::invalid_),
      max_turnOn_(2, sistrip::invalid_),
      max_peakTime_(2, sistrip::invalid_),
      max_undershoot_(2, sistrip::invalid_),
      max_baseline_(2, sistrip::invalid_),
      max_smearing_(2, sistrip::invalid_),
      max_chi2_(2, sistrip::invalid_),
      spread_amplitude_(2, sistrip::invalid_),
      spread_tail_(2, sistrip::invalid_),
      spread_riseTime_(2, sistrip::invalid_),
      spread_decayTime_(2, sistrip::invalid_),
      spread_turnOn_(2, sistrip::invalid_),
      spread_peakTime_(2, sistrip::invalid_),
      spread_undershoot_(2, sistrip::invalid_),
      spread_baseline_(2, sistrip::invalid_),
      spread_smearing_(2, sistrip::invalid_),
      spread_chi2_(2, sistrip::invalid_),
      deconv_(deconv),
      calChan_(0) {
  ;
}

// ----------------------------------------------------------------------------
//
void CalibrationAnalysis::reset() {
  calChan_ = 0;

  amplitude_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  tail_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  riseTime_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  decayTime_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  turnOn_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  peakTime_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  undershoot_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  baseline_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  smearing_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  chi2_ = VVFloat(2, VFloat(128, sistrip::invalid_));
  isvalid_ = VVBool(2, VBool(128, sistrip::invalid_));

  mean_amplitude_ = VFloat(2, sistrip::invalid_);
  mean_tail_ = VFloat(2, sistrip::invalid_);
  mean_riseTime_ = VFloat(2, sistrip::invalid_);
  mean_decayTime_ = VFloat(2, sistrip::invalid_);
  mean_turnOn_ = VFloat(2, sistrip::invalid_);
  mean_peakTime_ = VFloat(2, sistrip::invalid_);
  mean_undershoot_ = VFloat(2, sistrip::invalid_);
  mean_baseline_ = VFloat(2, sistrip::invalid_);
  mean_smearing_ = VFloat(2, sistrip::invalid_);
  mean_chi2_ = VFloat(2, sistrip::invalid_);

  min_amplitude_ = VFloat(2, sistrip::invalid_);
  min_tail_ = VFloat(2, sistrip::invalid_);
  min_riseTime_ = VFloat(2, sistrip::invalid_);
  min_decayTime_ = VFloat(2, sistrip::invalid_);
  min_turnOn_ = VFloat(2, sistrip::invalid_);
  min_peakTime_ = VFloat(2, sistrip::invalid_);
  min_undershoot_ = VFloat(2, sistrip::invalid_);
  min_baseline_ = VFloat(2, sistrip::invalid_);
  min_smearing_ = VFloat(2, sistrip::invalid_);
  min_chi2_ = VFloat(2, sistrip::invalid_);

  max_amplitude_ = VFloat(2, sistrip::invalid_);
  max_tail_ = VFloat(2, sistrip::invalid_);
  max_riseTime_ = VFloat(2, sistrip::invalid_);
  max_decayTime_ = VFloat(2, sistrip::invalid_);
  max_turnOn_ = VFloat(2, sistrip::invalid_);
  max_peakTime_ = VFloat(2, sistrip::invalid_);
  max_undershoot_ = VFloat(2, sistrip::invalid_);
  max_baseline_ = VFloat(2, sistrip::invalid_);
  max_smearing_ = VFloat(2, sistrip::invalid_);
  max_chi2_ = VFloat(2, sistrip::invalid_);

  spread_amplitude_ = VFloat(2, sistrip::invalid_);
  spread_tail_ = VFloat(2, sistrip::invalid_);
  spread_riseTime_ = VFloat(2, sistrip::invalid_);
  spread_decayTime_ = VFloat(2, sistrip::invalid_);
  spread_turnOn_ = VFloat(2, sistrip::invalid_);
  spread_peakTime_ = VFloat(2, sistrip::invalid_);
  spread_undershoot_ = VFloat(2, sistrip::invalid_);
  spread_baseline_ = VFloat(2, sistrip::invalid_);
  spread_smearing_ = VFloat(2, sistrip::invalid_);
  spread_chi2_ = VFloat(2, sistrip::invalid_);
}

// ----------------------------------------------------------------------------
//
void CalibrationAnalysis::print(std::stringstream& ss, uint32_t iapv) {
  header(ss);
  ss << " Monitorables for APV number     : " << iapv;
  if (iapv == 0) {
    ss << " (first of pair)";
  } else if (iapv == 1) {
    ss << " (second of pair)";
  }
  ss << std::endl;
  ss << " Mean Amplitude of the pulse : " << mean_amplitude_[iapv] << std::endl
     << " Mean Tail amplitude after 150ns : " << mean_tail_[iapv] << std::endl
     << " Mean Rise time : " << mean_riseTime_[iapv] << std::endl
     << " Mean Time constant : " << mean_decayTime_[iapv] << std::endl
     << " Mean Turn on time  : " << mean_turnOn_[iapv] << std::endl
     << " Mean peak time : " << mean_peakTime_[iapv] << std::endl
     << " Mean undershoot amplitude : " << mean_undershoot_[iapv] << std::endl
     << " Mean baseline amplitude : " << mean_baseline_[iapv] << std::endl
     << " Mean Smearing parameter : " << mean_smearing_[iapv] << std::endl
     << " Mean Chi2 of the fit : " << mean_chi2_[iapv] << std::endl;
  if (deconvMode()) {
    ss << "Data obtained in deconvolution mode." << std::endl;
  } else {
    ss << "Data obtained in peak mode." << std::endl;
  }
}

bool CalibrationAnalysis::isValid() const { return true; }
