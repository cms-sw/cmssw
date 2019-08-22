#include "DQM/SiStripCommissioningSummary/interface/CalibrationScanSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/CalibrationScanAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void CalibrationScanSummaryFactory::extract(Iterator iter) {
  CalibrationScanAnalysis* anal = dynamic_cast<CalibrationScanAnalysis*>(iter->second);
  if (!anal) {
    return;
  }

  SiStripFecKey lldKey = SiStripFecKey(iter->first);

  uint32_t key1 = SiStripFecKey(lldKey.fecCrate(),
                                lldKey.fecSlot(),
                                lldKey.fecRing(),
                                lldKey.ccuAddr(),
                                lldKey.ccuChan(),
                                lldKey.lldChan(),
                                lldKey.i2cAddr(lldKey.lldChan(), true))
                      .key();

  uint32_t key2 = SiStripFecKey(lldKey.fecCrate(),
                                lldKey.fecSlot(),
                                lldKey.fecRing(),
                                lldKey.ccuAddr(),
                                lldKey.ccuChan(),
                                lldKey.lldChan(),
                                lldKey.i2cAddr(lldKey.lldChan(), false))
                      .key();

  std::vector<float> value(2, 1 * sistrip::invalid_);

  // search for the best isha and VFS values cose to the optimal ones
  if (mon_ == sistrip::CALIBRATION_AMPLITUDE_TUNED) {
    value[0] = anal->tunedAmplitude()[0];
    value[1] = anal->tunedAmplitude()[1];
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_TUNED) {
    value[0] = anal->tunedBaseline()[0];
    value[1] = anal->tunedBaseline()[1];
  } else if (mon_ == sistrip::CALIBRATION_TURNON_TUNED) {
    value[0] = anal->tunedTurnOn()[0];
    value[1] = anal->tunedTurnOn()[1];
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_TUNED) {
    value[0] = anal->tunedRiseTime()[0];
    value[1] = anal->tunedRiseTime()[1];
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_TUNED) {
    value[0] = anal->tunedDecayTime()[0];
    value[1] = anal->tunedDecayTime()[1];
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_TUNED) {
    value[0] = anal->tunedPeakTime()[0];
    value[1] = anal->tunedPeakTime()[1];
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_TUNED) {
    value[0] = anal->tunedUndershoot()[0];
    value[1] = anal->tunedUndershoot()[1];
  } else if (mon_ == sistrip::CALIBRATION_TAIL_TUNED) {
    value[0] = anal->tunedTail()[0];
    value[1] = anal->tunedTail()[1];
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_TUNED) {
    value[0] = anal->tunedSmearing()[0];
    value[1] = anal->tunedSmearing()[1];
  } else if (mon_ == sistrip::CALIBRATION_CHI2_TUNED) {
    value[0] = anal->tunedChi2()[0];
    value[1] = anal->tunedChi2()[1];
  } else if (mon_ == sistrip::CALIBRATION_ISHA_TUNED) {
    value[0] = anal->tunedISHA()[0];
    value[1] = anal->tunedISHA()[1];
  } else if (mon_ == sistrip::CALIBRATION_VFS_TUNED) {
    value[0] = anal->tunedVFS()[0];
    value[1] = anal->tunedVFS()[1];
  } else if (mon_ == sistrip::CALIBRATION_ISHA_BEST) {
    value[0] = anal->bestISHA()[0];
    value[1] = anal->bestISHA()[1];
  } else if (mon_ == sistrip::CALIBRATION_VFS_BEST) {
    value[0] = anal->bestVFS()[0];
    value[1] = anal->bestVFS()[1];
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected monitorable: "
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
    return;
  }

  SummaryPlotFactoryBase::generator_->fillMap(
      SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key1, value[0]);

  SummaryPlotFactoryBase::generator_->fillMap(
      SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key2, value[1]);

  // set the x-axis
  format();
}

//------------------------------------------------------------------------------
//
void CalibrationScanSummaryFactory::format() {
  // Histogram formatting
  if (mon_ == sistrip::CALIBRATION_AMPLITUDE_TUNED)
    generator_->axisLabel("Amplitude (ADC)");

  else if (mon_ == sistrip::CALIBRATION_BASELINE_TUNED)
    generator_->axisLabel("Baseline (ADC)");

  else if (mon_ == sistrip::CALIBRATION_TURNON_TUNED)
    generator_->axisLabel("Turn-On (ns)");

  else if (mon_ == sistrip::CALIBRATION_TAIL_TUNED)
    generator_->axisLabel("Tail (%)");

  else if (mon_ == sistrip::CALIBRATION_RISETIME_TUNED)
    generator_->axisLabel("Rise Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_PEAKTIME_TUNED)
    generator_->axisLabel("Peak Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_DECAYTIME_TUNED)
    generator_->axisLabel("Decay Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_SMEARING_TUNED)
    generator_->axisLabel("Smearing (ns)");

  else if (mon_ == sistrip::CALIBRATION_CHI2_TUNED)
    generator_->axisLabel("Chi2/ndf");

  else if (mon_ == sistrip::CALIBRATION_VFS_TUNED or mon_ == sistrip::CALIBRATION_VFS_BEST)
    generator_->axisLabel("VFS");

  else if (mon_ == sistrip::CALIBRATION_ISHA_TUNED or mon_ == sistrip::CALIBRATION_ISHA_BEST)
    generator_->axisLabel("ISHA");
  else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected SummaryHisto value:"
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
  }
}
