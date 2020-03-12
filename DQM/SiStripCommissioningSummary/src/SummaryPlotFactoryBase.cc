#include "DQM/SiStripCommissioningSummary/interface/SummaryPlotFactoryBase.h"
#include "DQM/SiStripCommissioningSummary/interface/SummaryGenerator.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryPlotFactoryBase::SummaryPlotFactoryBase()
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
SummaryPlotFactoryBase::~SummaryPlotFactoryBase() {
  if (generator_) {
    delete generator_;
  }
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::init(const sistrip::Monitorable& mon,
                                  const sistrip::Presentation& pres,
                                  const sistrip::View& view,
                                  const std::string& level,
                                  const sistrip::Granularity& gran) {
  // Create generator object
  if (generator_) {
    delete generator_;
    generator_ = nullptr;
  }
  generator_ = SummaryGenerator::instance(view);

  // Check if generator object exists
  if (!generator_) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " NULL pointer to generator object!";
    return;
  }

  // Clear map used to build histogram
  generator_->clearMap();

  // Set parameters
  mon_ = mon;
  pres_ = pres;
  view_ = view;
  level_ = level;
  gran_ = gran;

  // Some checks
  if (mon_ == sistrip::UNKNOWN_MONITORABLE || mon_ == sistrip::UNDEFINED_MONITORABLE) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected monitorable: " << SiStripEnumsAndStrings::monitorable(mon_);
  }

  if (pres_ == sistrip::UNKNOWN_PRESENTATION || pres_ == sistrip::UNDEFINED_PRESENTATION) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected presentation: " << SiStripEnumsAndStrings::presentation(pres_);
  }

  if (view_ == sistrip::UNKNOWN_VIEW || view_ == sistrip::UNDEFINED_VIEW) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected view: " << SiStripEnumsAndStrings::view(view_);
  }

  if (level_.empty() || level_.find(sistrip::unknownView_) != std::string::npos ||
      level_.find(sistrip::undefinedView_) != std::string::npos) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected top-level directory: \"" << level_ << "\"";
  }

  if ((gran_ == sistrip::UNKNOWN_GRAN || gran_ == sistrip::UNDEFINED_GRAN) && pres != sistrip::HISTO_1D) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected granularity: " << SiStripEnumsAndStrings::granularity(gran_);
  }

  //   ss << "[SummaryPlotFactoryBase::" << __func__ << "]"
  //      << " Dump of parameters defining summary plot:" << std::endl
  //      << " Monitorable   : " << SiStripEnumsAndStrings::monitorable( mon_ ) << std::endl
  //      << " Presentation  : " << SiStripEnumsAndStrings::presentation( pres_ ) << std::endl
  //      << " Logical view  : " << SiStripEnumsAndStrings::view( view_ ) << std::endl
  //      << " Top level dir : " << level_ << std::endl
  //      << " Granularity   : " << SiStripEnumsAndStrings::granularity( gran_ );
  //   LogTrace(mlSummaryPlots_) << ss.str();
}

// -----------------------------------------------------------------------------
//
void SummaryPlotFactoryBase::fill(TH1& summary_histo) {
  // Check if instance of generator class exists
  if (!generator_) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " NULL pointer to SummaryGenerator object!";
    return;
  }

  // Check if map is filled
  if (!generator_->nBins()) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Zero bins returned by SummaryGenerator!";
    return;
  }

  // Print contents of map for histogram
  //generator_->printMap();

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
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlotFactoryBase::" << __func__ << "]"
                                     << " Unexpected presentation type: "
                                     << SiStripEnumsAndStrings::presentation(pres_);
    return;
  }

  // Histogram formatting
  generator_->format(sistrip::UNKNOWN_RUN_TYPE,  //@@ not used
                     mon_,
                     pres_,
                     view_,
                     level_,
                     gran_,
                     summary_histo);
}
