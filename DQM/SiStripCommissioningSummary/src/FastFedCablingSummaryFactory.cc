#include "DQM/SiStripCommissioningSummary/interface/FastFedCablingSummaryFactory.h"
#include "CondFormats/SiStripObjects/interface/FastFedCablingAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
void FastFedCablingSummaryFactory::extract(Iterator iter) {
  FastFedCablingAnalysis* anal = dynamic_cast<FastFedCablingAnalysis*>(iter->second);
  if (!anal) {
    return;
  }

  float value = 1. * sistrip::invalid_;
  float error = 1. * sistrip::invalid_;

  if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_HIGH_LEVEL) {
    value = anal->highLevel();
    error = anal->highRms();
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_LOW_LEVEL) {
    value = anal->lowLevel();
    error = anal->lowRms();
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_MAX) {
    value = anal->max();
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_MIN) {
    value = anal->min();
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_CONNS_PER_FED) {
    value = 1. * static_cast<uint16_t>(anal->isValid());
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected monitorable: "
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
    return;
  }

  SummaryPlotFactoryBase::generator_->fillMap(
      SummaryPlotFactoryBase::level_, SummaryPlotFactoryBase::gran_, iter->first, value, error);
}

// -----------------------------------------------------------------------------
//
void FastFedCablingSummaryFactory::format() {
  if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_HIGH_LEVEL) {
    SummaryPlotFactoryBase::generator_->axisLabel("\"High\" light level [ADC]");
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_LOW_LEVEL) {
    SummaryPlotFactoryBase::generator_->axisLabel("\"Low\" light level [ADC]");
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_MAX) {
    SummaryPlotFactoryBase::generator_->axisLabel("Maximum light level [ADC]");
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_MIN) {
    SummaryPlotFactoryBase::generator_->axisLabel("Minumum light level [ADC]");
  } else if (SummaryPlotFactoryBase::mon_ == sistrip::FAST_CABLING_CONNS_PER_FED) {
    SummaryPlotFactoryBase::generator_->axisLabel("Connected channels per FED");
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactory::" << __func__ << "]"
                                     << " Unexpected SummaryHisto value:"
                                     << SiStripEnumsAndStrings::monitorable(SummaryPlotFactoryBase::mon_);
  }
}
