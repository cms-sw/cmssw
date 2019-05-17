#include "DQM/SiStripCommissioningSummary/interface/PedsOnlySummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/PedsOnlyAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void PedsOnlySummaryFactory::extract(Iterator iter) {
  PedsOnlyAnalysis* anal = dynamic_cast<PedsOnlyAnalysis*>(iter->second);
  if (!anal) {
    return;
  }

  std::vector<float> temp(128, 1. * sistrip::invalid_);
  std::vector<std::vector<float> > value(2, temp);
  std::vector<std::vector<float> > peds(2, temp);
  std::vector<std::vector<float> > noise(2, temp);
  peds[0] = anal->peds()[0];
  peds[1] = anal->peds()[1];
  noise[0] = anal->raw()[0];  //@@ raw noise
  noise[1] = anal->raw()[1];  //@@ raw noise

  bool all_strips = false;
  if (mon_ == sistrip::PEDESTALS_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = peds[0].size();
    if (peds[0].size() < peds[1].size()) {
      bins = peds[1].size();
    }
    for (uint16_t iped = 0; iped < bins; iped++) {
      value[0][iped] = peds[0][iped];
      value[1][iped] = peds[1][iped];
    }
  } else if (mon_ == sistrip::PEDESTALS_MEAN) {
    value[0][0] = anal->pedsMean()[0];
    value[1][0] = anal->pedsMean()[1];
  } else if (mon_ == sistrip::PEDESTALS_SPREAD) {
    value[0][0] = anal->pedsSpread()[0];
    value[1][0] = anal->pedsSpread()[1];
  } else if (mon_ == sistrip::PEDESTALS_MAX) {
    value[0][0] = anal->pedsMax()[0];
    value[1][0] = anal->pedsMax()[1];
  } else if (mon_ == sistrip::PEDESTALS_MIN) {
    value[0][0] = anal->pedsMin()[0];
    value[1][0] = anal->pedsMin()[1];
  } else if (mon_ == sistrip::NOISE_ALL_STRIPS) {
    all_strips = true;
    uint16_t bins = noise[0].size();
    if (noise[0].size() < noise[1].size()) {
      bins = noise[1].size();
    }
    for (uint16_t inoise = 0; inoise < bins; inoise++) {
      value[0][inoise] = noise[0][inoise];
      value[1][inoise] = noise[1][inoise];
    }
  } else if (mon_ == sistrip::NOISE_MEAN) {
    value[0][0] = anal->rawMean()[0];
    value[1][0] = anal->rawMean()[1];
  } else if (mon_ == sistrip::NOISE_SPREAD) {
    value[0][0] = anal->rawSpread()[0];
    value[1][0] = anal->rawSpread()[1];
  } else if (mon_ == sistrip::NOISE_MAX) {
    value[0][0] = anal->rawMax()[0];
    value[1][0] = anal->rawMax()[1];
  } else if (mon_ == sistrip::NOISE_MIN) {
    value[0][0] = anal->rawMin()[0];
    value[1][0] = anal->rawMin()[1];
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected monitorable: "
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
    return;
  }

  if (!all_strips) {
    SummaryPlotFactoryBase::generator_->fillMap(
        SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value[0][0]);

    SummaryPlotFactoryBase::generator_->fillMap(
        SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value[1][0]);

  } else {
    for (uint16_t istr = 0; istr < value[0].size(); istr++) {
      SummaryPlotFactoryBase::generator_->fillMap(
          SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value[0][istr]);
    }

    for (uint16_t istr = 0; istr < value[1].size(); istr++) {
      SummaryPlotFactoryBase::generator_->fillMap(
          SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value[1][istr]);
    }
  }
}

// -----------------------------------------------------------------------------
//
void PedsOnlySummaryFactory::format() {
  if (mon_ == sistrip::PEDESTALS_ALL_STRIPS) {
    generator_->axisLabel("Pedestal value [adc]");
  } else if (mon_ == sistrip::PEDESTALS_MEAN) {
  } else if (mon_ == sistrip::PEDESTALS_SPREAD) {
  } else if (mon_ == sistrip::PEDESTALS_MAX) {
  } else if (mon_ == sistrip::PEDESTALS_MIN) {
  } else if (mon_ == sistrip::NOISE_ALL_STRIPS) {
    generator_->axisLabel("Noise [adc]");
  } else if (mon_ == sistrip::NOISE_MEAN) {
  } else if (mon_ == sistrip::NOISE_SPREAD) {
  } else if (mon_ == sistrip::NOISE_MAX) {
  } else if (mon_ == sistrip::NOISE_MIN) {
  } else if (mon_ == sistrip::NUM_OF_DEAD) {
  } else if (mon_ == sistrip::NUM_OF_NOISY) {
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected SummaryHisto value:"
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
  }
}
