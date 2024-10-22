#include "CondFormats/SiStripObjects/interface/CalibrationScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>

using namespace sistrip;

const float CalibrationScanAnalysis::minAmplitudeThreshold_ = 50;
const float CalibrationScanAnalysis::minBaselineThreshold_ = -50;
const float CalibrationScanAnalysis::maxBaselineThreshold_ = 50;
const float CalibrationScanAnalysis::maxChi2Threshold_ = 3;
const float CalibrationScanAnalysis::minDecayTimeThreshold_ = 30;
const float CalibrationScanAnalysis::maxDecayTimeThreshold_ = 250;
const float CalibrationScanAnalysis::minPeakTimeThreshold_ = 40;
const float CalibrationScanAnalysis::maxPeakTimeThreshold_ = 130;
const float CalibrationScanAnalysis::minRiseTimeThreshold_ = 10;
const float CalibrationScanAnalysis::maxRiseTimeThreshold_ = 100;
const float CalibrationScanAnalysis::minTurnOnThreshold_ = 5;
const float CalibrationScanAnalysis::maxTurnOnThreshold_ = 40;
const float CalibrationScanAnalysis::minISHAforVFSTune_ = 30;
const float CalibrationScanAnalysis::maxISHAforVFSTune_ = 110;
const float CalibrationScanAnalysis::VFSrange_ = 20;

CalibrationScanAnalysis::CalibrationScanAnalysis(const uint32_t& key, const bool& deconv)
    : CommissioningAnalysis(key, "CalibrationScanAnalysis"), deconv_(deconv) {
  isha_ = VInt(2, sistrip::invalid_);
  vfs_ = VInt(2, sistrip::invalid_);
  tunedAmplitude_ = VFloat(2, sistrip::invalid_);
  tunedTail_ = VFloat(2, sistrip::invalid_);
  tunedRiseTime_ = VFloat(2, sistrip::invalid_);
  tunedDecayTime_ = VFloat(2, sistrip::invalid_);
  tunedTurnOn_ = VFloat(2, sistrip::invalid_);
  tunedPeakTime_ = VFloat(2, sistrip::invalid_);
  tunedUndershoot_ = VFloat(2, sistrip::invalid_);
  tunedBaseline_ = VFloat(2, sistrip::invalid_);
  tunedSmearing_ = VFloat(2, sistrip::invalid_);
  tunedChi2_ = VFloat(2, sistrip::invalid_);
  tunedISHA_ = VInt(2, sistrip::invalid_);
  tunedVFS_ = VInt(2, sistrip::invalid_);
}

// ----------------------------------------------------------------------------
//
CalibrationScanAnalysis::CalibrationScanAnalysis(const bool& deconv)
    : CommissioningAnalysis("CalibrationScanAnalysis"), deconv_(deconv) {
  isha_ = VInt(2, sistrip::invalid_);
  vfs_ = VInt(2, sistrip::invalid_);

  tunedAmplitude_ = VFloat(2, sistrip::invalid_);
  tunedTail_ = VFloat(2, sistrip::invalid_);
  tunedRiseTime_ = VFloat(2, sistrip::invalid_);
  tunedDecayTime_ = VFloat(2, sistrip::invalid_);
  tunedTurnOn_ = VFloat(2, sistrip::invalid_);
  tunedPeakTime_ = VFloat(2, sistrip::invalid_);
  tunedUndershoot_ = VFloat(2, sistrip::invalid_);
  tunedBaseline_ = VFloat(2, sistrip::invalid_);
  tunedSmearing_ = VFloat(2, sistrip::invalid_);
  tunedChi2_ = VFloat(2, sistrip::invalid_);
  tunedISHA_ = VInt(2, sistrip::invalid_);
  tunedVFS_ = VInt(2, sistrip::invalid_);
}
// ----------------------------------------------------------------------------
//
void CalibrationScanAnalysis::addOneCalibrationPoint(const std::string& key)  // key of form isha_%d_vfs_%d
{
  amplitude_[key] = VFloat(2, sistrip::invalid_);
  tail_[key] = VFloat(2, sistrip::invalid_);
  riseTime_[key] = VFloat(2, sistrip::invalid_);
  turnOn_[key] = VFloat(2, sistrip::invalid_);
  peakTime_[key] = VFloat(2, sistrip::invalid_);
  undershoot_[key] = VFloat(2, sistrip::invalid_);
  baseline_[key] = VFloat(2, sistrip::invalid_);
  smearing_[key] = VFloat(2, sistrip::invalid_);
  chi2_[key] = VFloat(2, sistrip::invalid_);
  decayTime_[key] = VFloat(2, sistrip::invalid_);
  isvalid_[key] = VBool(2, sistrip::invalid_);
}

// ----------------------------------------------------------------------------
//
void CalibrationScanAnalysis::reset() {
  isha_ = VInt(2, sistrip::invalid_);
  vfs_ = VInt(2, sistrip::invalid_);
  tunedAmplitude_ = VFloat(2, sistrip::invalid_);
  tunedTail_ = VFloat(2, sistrip::invalid_);
  tunedRiseTime_ = VFloat(2, sistrip::invalid_);
  tunedDecayTime_ = VFloat(2, sistrip::invalid_);
  tunedTurnOn_ = VFloat(2, sistrip::invalid_);
  tunedPeakTime_ = VFloat(2, sistrip::invalid_);
  tunedUndershoot_ = VFloat(2, sistrip::invalid_);
  tunedBaseline_ = VFloat(2, sistrip::invalid_);
  tunedSmearing_ = VFloat(2, sistrip::invalid_);
  tunedChi2_ = VFloat(2, sistrip::invalid_);
  tunedISHA_ = VInt(2, sistrip::invalid_);
  tunedVFS_ = VInt(2, sistrip::invalid_);

  for (const auto& key : amplitude_) {
    amplitude_[key.first] = VFloat(2, sistrip::invalid_);
    tail_[key.first] = VFloat(2, sistrip::invalid_);
    riseTime_[key.first] = VFloat(2, sistrip::invalid_);
    turnOn_[key.first] = VFloat(2, sistrip::invalid_);
    peakTime_[key.first] = VFloat(2, sistrip::invalid_);
    undershoot_[key.first] = VFloat(2, sistrip::invalid_);
    baseline_[key.first] = VFloat(2, sistrip::invalid_);
    smearing_[key.first] = VFloat(2, sistrip::invalid_);
    chi2_[key.first] = VFloat(2, sistrip::invalid_);
    decayTime_[key.first] = VFloat(2, sistrip::invalid_);
  }
}

// ----------------------------------------------------------------------------
//
void CalibrationScanAnalysis::print(std::stringstream& ss, uint32_t iapv) {
  header(ss);
  ss << " Monitorables for APV number     : " << iapv;
  if (iapv == 0) {
    ss << " (first of pair)";
  } else if (iapv == 1) {
    ss << " (second of pair)";
  }
  ss << std::endl;
  ss << " Looking at key " << amplitude_.begin()->first << std::endl;
  ss << " Amplitude of the pulse : " << amplitude_[amplitude_.begin()->first][iapv] << std::endl
     << " Baseline : " << baseline_[amplitude_.begin()->first][iapv] << std::endl
     << " Rise time : " << riseTime_[amplitude_.begin()->first][iapv] << std::endl
     << " Turn-on time : " << turnOn_[amplitude_.begin()->first][iapv] << std::endl
     << " Peak time : " << peakTime_[amplitude_.begin()->first][iapv] << std::endl
     << " Undershoot : " << undershoot_[amplitude_.begin()->first][iapv] << std::endl
     << " Time constant : " << decayTime_[amplitude_.begin()->first][iapv] << std::endl
     << " Smearing : " << smearing_[amplitude_.begin()->first][iapv] << std::endl
     << " Chi2 : " << chi2_[amplitude_.begin()->first][iapv] << std::endl;
  if (deconvMode()) {
    ss << "Data obtained in deconvolution mode." << std::endl;
  } else {
    ss << "Data obtained in peak mode." << std::endl;
  }
}

bool CalibrationScanAnalysis::isValid() const { return true; }
