#include "DQM/SiStripCommissioningSummary/interface/FedTimingSummaryFactory.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<FedTimingAnalysis>::SummaryHistogramFactory()
    : mon_(sistrip::UNKNOWN_MONITORABLE),
      pres_(sistrip::UNKNOWN_PRESENTATION),
      view_(sistrip::UNKNOWN_VIEW),
      level_(sistrip::root_),
      gran_(sistrip::UNKNOWN_GRAN),
      generator_(nullptr) {
  ;
}

// -----------------------------------------------------------------------------
//
SummaryHistogramFactory<FedTimingAnalysis>::~SummaryHistogramFactory() {
  if (generator_) {
    delete generator_;
  }
}

// -----------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedTimingAnalysis>::init(const sistrip::Monitorable& mon,
                                                      const sistrip::Presentation& pres,
                                                      const sistrip::View& view,
                                                      const std::string& top_level_dir,
                                                      const sistrip::Granularity& gran) {
  LogTrace(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]";
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = top_level_dir;
  gran_ = gran;

  // Retrieve utility class used to generate summary histograms
  if (generator_) {
    delete generator_;
    generator_ = nullptr;
  }
  generator_ = SummaryGenerator::instance(view);
}

//------------------------------------------------------------------------------
//
uint32_t SummaryHistogramFactory<FedTimingAnalysis>::extract(const std::map<uint32_t, FedTimingAnalysis>& data) {
  // Check if data are present
  if (data.empty()) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " No data in monitorables std::map!";
    return 0;
  }

  // Check if instance of generator class exists
  if (!generator_) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " NULL pointer to SummaryGenerator object!";
    return 0;
  }

  // Transfer appropriate monitorables info to generator object
  generator_->clearMap();
  std::map<uint32_t, FedTimingAnalysis>::const_iterator iter = data.begin();
  for (; iter != data.end(); iter++) {
    if (mon_ == sistrip::FED_TIMING_TIME) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.time());
    } else if (mon_ == sistrip::FED_TIMING_MAX_TIME) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.max());
    } else if (mon_ == sistrip::FED_TIMING_DELAY) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.delay());
    } else if (mon_ == sistrip::FED_TIMING_ERROR) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.error());
    } else if (mon_ == sistrip::FED_TIMING_BASE) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.base());
    } else if (mon_ == sistrip::FED_TIMING_PEAK) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.peak());
    } else if (mon_ == sistrip::FED_TIMING_HEIGHT) {
      generator_->fillMap(level_, gran_, iter->first, iter->second.height());
    } else {
      edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                       << " Unexpected SummaryHisto value:"
                                       << SiStripEnumsAndStrings::monitorable(mon_);
      continue;
    }
  }
  return generator_->size();
}

//------------------------------------------------------------------------------
//
void SummaryHistogramFactory<FedTimingAnalysis>::fill(TH1& summary_histo) {
  // Check if instance of generator class exists
  if (!generator_) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if std::map is filled
  if (!generator_->size()) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " No data in the monitorables std::map!";
    return;
  }

  // Generate appropriate summary histogram
  if (pres_ == sistrip::HISTO_1D) {
    generator_->histo1D(summary_histo);
  } else if (pres_ == sistrip::HISTO_2D_SUM) {
    generator_->histo2DSum(summary_histo);
  } else if (pres_ == sistrip::HISTO_2D_SCATTER) {
    generator_->histo2DScatter(summary_histo);
  } else if (pres_ == sistrip::PROFILE_1D) {
    generator_->profile1D(summary_histo);
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " Unexpected SummaryType value:" << SiStripEnumsAndStrings::presentation(pres_);
    return;
  }

  // Histogram formatting
  if (mon_ == sistrip::FED_TIMING_TIME) {
  } else if (mon_ == sistrip::FED_TIMING_MAX_TIME) {
  } else if (mon_ == sistrip::FED_TIMING_DELAY) {
  } else if (mon_ == sistrip::FED_TIMING_ERROR) {
  } else if (mon_ == sistrip::FED_TIMING_BASE) {
  } else if (mon_ == sistrip::FED_TIMING_PEAK) {
  } else if (mon_ == sistrip::FED_TIMING_HEIGHT) {
  } else {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryHistogramFactory::" << __func__ << "]"
                                     << " Unexpected SummaryHisto value:" << SiStripEnumsAndStrings::monitorable(mon_);
  }
  generator_->format(sistrip::FED_TIMING, mon_, pres_, view_, level_, gran_, summary_histo);
}
