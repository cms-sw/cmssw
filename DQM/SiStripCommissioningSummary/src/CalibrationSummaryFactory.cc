#include "DQM/SiStripCommissioningSummary/interface/CalibrationSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/CalibrationAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void CalibrationSummaryFactory::extract(Iterator iter) {
  CalibrationAnalysis* anal = dynamic_cast<CalibrationAnalysis*>(iter->second);
  if (!anal) {
    return;
  }

  std::vector<float> temp(128, 1. * sistrip::invalid_);
  std::vector<std::vector<float> > amplitude(2, temp);
  std::vector<std::vector<float> > baseline(2, temp);
  std::vector<std::vector<float> > riseTime(2, temp);
  std::vector<std::vector<float> > turnOn(2, temp);
  std::vector<std::vector<float> > peakTime(2, temp);
  std::vector<std::vector<float> > undershoot(2, temp);
  std::vector<std::vector<float> > tail(2, temp);
  std::vector<std::vector<float> > decayTime(2, temp);
  std::vector<std::vector<float> > smearing(2, temp);
  std::vector<std::vector<float> > chi2(2, temp);

  std::vector<std::vector<float> > value(2, temp);

  amplitude[0] = anal->amplitude()[0];
  amplitude[1] = anal->amplitude()[1];
  baseline[0] = anal->baseline()[0];
  baseline[1] = anal->baseline()[1];
  tail[0] = anal->tail()[0];
  tail[1] = anal->tail()[1];
  riseTime[0] = anal->riseTime()[0];
  riseTime[1] = anal->riseTime()[1];
  decayTime[0] = anal->decayTime()[0];
  decayTime[1] = anal->decayTime()[1];
  peakTime[0] = anal->peakTime()[0];
  peakTime[1] = anal->peakTime()[1];
  turnOn[0] = anal->turnOn()[0];
  turnOn[1] = anal->turnOn()[1];
  undershoot[0] = anal->undershoot()[0];
  undershoot[1] = anal->undershoot()[1];
  smearing[0] = anal->smearing()[0];
  smearing[1] = anal->smearing()[1];
  chi2[0] = anal->chi2()[0];
  chi2[1] = anal->chi2()[1];

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

  bool all_strips = false;
  if (mon_ == sistrip::CALIBRATION_AMPLITUDE_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = amplitude[amplitude[0].size() < amplitude[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = amplitude[0][i];
      value[1][i] = amplitude[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = baseline[baseline[0].size() < baseline[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = baseline[0][i];
      value[1][i] = baseline[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_TURNON_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = turnOn[turnOn[0].size() < turnOn[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = turnOn[0][i];
      value[1][i] = turnOn[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = riseTime[riseTime[0].size() < riseTime[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = riseTime[0][i];
      value[1][i] = riseTime[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = decayTime[decayTime[0].size() < decayTime[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = decayTime[0][i];
      value[1][i] = decayTime[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = peakTime[peakTime[0].size() < peakTime[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = peakTime[0][i];
      value[1][i] = peakTime[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = undershoot[undershoot[0].size() < undershoot[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = undershoot[0][i];
      value[1][i] = undershoot[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_TAIL_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = tail[tail[0].size() < tail[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = tail[0][i];
      value[1][i] = tail[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = smearing[smearing[0].size() < smearing[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = smearing[0][i];
      value[1][i] = smearing[1][i];
    }
  } else if (mon_ == sistrip::CALIBRATION_CHI2_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = chi2[chi2[0].size() < chi2[1].size() ? 1 : 0].size();
    for (uint16_t i = 0; i < bins; i++) {
      value[0][i] = chi2[0][i];
      value[1][i] = chi2[1][i];
    }
  }
  //////
  else if (mon_ == sistrip::CALIBRATION_AMPLITUDE_MEAN) {
    value[0][0] = anal->amplitudeMean()[0];
    value[1][0] = anal->amplitudeMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_MEAN) {
    value[0][0] = anal->baselineMean()[0];
    value[1][0] = anal->baselineMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_TURNON_MEAN) {
    value[0][0] = anal->turnOnMean()[0];
    value[1][0] = anal->turnOnMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_MEAN) {
    value[0][0] = anal->riseTimeMean()[0];
    value[1][0] = anal->riseTimeMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_MEAN) {
    value[0][0] = anal->decayTimeMean()[0];
    value[1][0] = anal->decayTimeMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_MEAN) {
    value[0][0] = anal->peakTimeMean()[0];
    value[1][0] = anal->peakTimeMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_MEAN) {
    value[0][0] = anal->undershootMean()[0];
    value[1][0] = anal->undershootMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_TAIL_MEAN) {
    value[0][0] = anal->tailMean()[0];
    value[1][0] = anal->tailMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_MEAN) {
    value[0][0] = anal->smearingMean()[0];
    value[1][0] = anal->smearingMean()[1];
  } else if (mon_ == sistrip::CALIBRATION_CHI2_MEAN) {
    value[0][0] = anal->chi2Mean()[0];
    value[1][0] = anal->chi2Mean()[1];
  }
  //////
  else if (mon_ == sistrip::CALIBRATION_AMPLITUDE_MIN) {
    value[0][0] = anal->amplitudeMin()[0];
    value[1][0] = anal->amplitudeMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_MIN) {
    value[0][0] = anal->baselineMin()[0];
    value[1][0] = anal->baselineMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_TURNON_MIN) {
    value[0][0] = anal->turnOnMin()[0];
    value[1][0] = anal->turnOnMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_MIN) {
    value[0][0] = anal->riseTimeMin()[0];
    value[1][0] = anal->riseTimeMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_MIN) {
    value[0][0] = anal->decayTimeMin()[0];
    value[1][0] = anal->decayTimeMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_MIN) {
    value[0][0] = anal->peakTimeMin()[0];
    value[1][0] = anal->peakTimeMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_MIN) {
    value[0][0] = anal->undershootMin()[0];
    value[1][0] = anal->undershootMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_TAIL_MIN) {
    value[0][0] = anal->tailMin()[0];
    value[1][0] = anal->tailMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_MIN) {
    value[0][0] = anal->smearingMin()[0];
    value[1][0] = anal->smearingMin()[1];
  } else if (mon_ == sistrip::CALIBRATION_CHI2_MIN) {
    value[0][0] = anal->chi2Min()[0];
    value[1][0] = anal->chi2Min()[1];
  }
  //////
  else if (mon_ == sistrip::CALIBRATION_AMPLITUDE_MAX) {
    value[0][0] = anal->amplitudeMax()[0];
    value[1][0] = anal->amplitudeMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_MAX) {
    value[0][0] = anal->baselineMax()[0];
    value[1][0] = anal->baselineMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_TURNON_MAX) {
    value[0][0] = anal->turnOnMax()[0];
    value[1][0] = anal->turnOnMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_MAX) {
    value[0][0] = anal->riseTimeMax()[0];
    value[1][0] = anal->riseTimeMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_MAX) {
    value[0][0] = anal->decayTimeMax()[0];
    value[1][0] = anal->decayTimeMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_MAX) {
    value[0][0] = anal->peakTimeMax()[0];
    value[1][0] = anal->peakTimeMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_MAX) {
    value[0][0] = anal->undershootMax()[0];
    value[1][0] = anal->undershootMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_TAIL_MAX) {
    value[0][0] = anal->tailMax()[0];
    value[1][0] = anal->tailMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_MAX) {
    value[0][0] = anal->smearingMax()[0];
    value[1][0] = anal->smearingMax()[1];
  } else if (mon_ == sistrip::CALIBRATION_CHI2_MAX) {
    value[0][0] = anal->chi2Max()[0];
    value[1][0] = anal->chi2Max()[1];
  }
  //////
  else if (mon_ == sistrip::CALIBRATION_AMPLITUDE_SPREAD) {
    value[0][0] = anal->amplitudeSpread()[0];
    value[1][0] = anal->amplitudeSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_BASELINE_SPREAD) {
    value[0][0] = anal->baselineSpread()[0];
    value[1][0] = anal->baselineSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_TURNON_SPREAD) {
    value[0][0] = anal->turnOnSpread()[0];
    value[1][0] = anal->turnOnSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_RISETIME_SPREAD) {
    value[0][0] = anal->riseTimeSpread()[0];
    value[1][0] = anal->riseTimeSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_DECAYTIME_SPREAD) {
    value[0][0] = anal->decayTimeSpread()[0];
    value[1][0] = anal->decayTimeSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_PEAKTIME_SPREAD) {
    value[0][0] = anal->peakTimeSpread()[0];
    value[1][0] = anal->peakTimeSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_SPREAD) {
    value[0][0] = anal->undershootSpread()[0];
    value[1][0] = anal->undershootSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_TAIL_SPREAD) {
    value[0][0] = anal->tailSpread()[0];
    value[1][0] = anal->tailSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_SMEARING_SPREAD) {
    value[0][0] = anal->smearingSpread()[0];
    value[1][0] = anal->smearingSpread()[1];
  } else if (mon_ == sistrip::CALIBRATION_CHI2_SPREAD) {
    value[0][0] = anal->chi2Spread()[0];
    value[1][0] = anal->chi2Spread()[1];
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected monitorable: "
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
    return;
  }

  if (!all_strips) {
    SummaryPlotFactoryBase::generator_->fillMap(
        SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key1, value[0][0]);

    SummaryPlotFactoryBase::generator_->fillMap(
        SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key2, value[1][0]);
  } else {
    for (uint16_t istr = 0; istr < value[0].size(); istr++)
      SummaryPlotFactoryBase::generator_->fillMap(
          SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key1, value[0][istr]);

    for (uint16_t istr = 0; istr < value[1].size(); istr++)
      SummaryPlotFactoryBase::generator_->fillMap(
          SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, key2, value[1][istr]);
  }

  format();
}

//------------------------------------------------------------------------------
//
void CalibrationSummaryFactory::format() {
  // Histogram formatting
  if (mon_ == sistrip::CALIBRATION_AMPLITUDE_MEAN or mon_ == sistrip::CALIBRATION_AMPLITUDE_ALL_STRIPS or
      mon_ == sistrip::CALIBRATION_AMPLITUDE_MIN or mon_ == sistrip::CALIBRATION_AMPLITUDE_MAX or
      mon_ == sistrip::CALIBRATION_AMPLITUDE_SPREAD)
    generator_->axisLabel("Amplitude (ADC)");

  else if (mon_ == sistrip::CALIBRATION_BASELINE_MEAN or mon_ == sistrip::CALIBRATION_BASELINE_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_BASELINE_MIN or mon_ == sistrip::CALIBRATION_BASELINE_MAX or
           mon_ == sistrip::CALIBRATION_BASELINE_SPREAD)
    generator_->axisLabel("Baseline (ADC)");

  else if (mon_ == sistrip::CALIBRATION_TURNON_MEAN or mon_ == sistrip::CALIBRATION_TURNON_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_TURNON_MIN or mon_ == sistrip::CALIBRATION_TURNON_MAX or
           mon_ == sistrip::CALIBRATION_TURNON_SPREAD)
    generator_->axisLabel("Turn-On (ns)");

  else if (mon_ == sistrip::CALIBRATION_TAIL_MEAN or mon_ == sistrip::CALIBRATION_TAIL_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_TAIL_MIN or mon_ == sistrip::CALIBRATION_TAIL_MAX or
           mon_ == sistrip::CALIBRATION_TAIL_SPREAD)
    generator_->axisLabel("Tail (%)");

  else if (mon_ == sistrip::CALIBRATION_RISETIME_MEAN or mon_ == sistrip::CALIBRATION_RISETIME_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_RISETIME_MIN or mon_ == sistrip::CALIBRATION_RISETIME_MAX or
           mon_ == sistrip::CALIBRATION_RISETIME_SPREAD)
    generator_->axisLabel("Rise Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_PEAKTIME_MEAN or mon_ == sistrip::CALIBRATION_PEAKTIME_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_PEAKTIME_MIN or mon_ == sistrip::CALIBRATION_PEAKTIME_MAX or
           mon_ == sistrip::CALIBRATION_PEAKTIME_SPREAD)
    generator_->axisLabel("Peak Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_DECAYTIME_MEAN or mon_ == sistrip::CALIBRATION_DECAYTIME_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_DECAYTIME_MIN or mon_ == sistrip::CALIBRATION_DECAYTIME_MAX or
           mon_ == sistrip::CALIBRATION_DECAYTIME_SPREAD)
    generator_->axisLabel("Decay Time (ns)");

  else if (mon_ == sistrip::CALIBRATION_SMEARING_MEAN or mon_ == sistrip::CALIBRATION_SMEARING_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_SMEARING_MIN or mon_ == sistrip::CALIBRATION_SMEARING_MAX or
           mon_ == sistrip::CALIBRATION_SMEARING_SPREAD)
    generator_->axisLabel("Smearing (ns)");

  else if (mon_ == sistrip::CALIBRATION_CHI2_MEAN or mon_ == sistrip::CALIBRATION_CHI2_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_CHI2_MIN or mon_ == sistrip::CALIBRATION_CHI2_MAX or
           mon_ == sistrip::CALIBRATION_CHI2_SPREAD)
    generator_->axisLabel("Chi2/ndf");

  else if (mon_ == sistrip::CALIBRATION_UNDERSHOOT_MEAN or mon_ == sistrip::CALIBRATION_UNDERSHOOT_ALL_STRIPS or
           mon_ == sistrip::CALIBRATION_UNDERSHOOT_MIN or mon_ == sistrip::CALIBRATION_UNDERSHOOT_MAX or
           mon_ == sistrip::CALIBRATION_UNDERSHOOT_SPREAD)
    generator_->axisLabel("Undershoot (%)");

  else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected SummaryHisto value:"
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
  }
}
