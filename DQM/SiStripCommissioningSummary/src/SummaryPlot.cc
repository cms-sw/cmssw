#include "DQM/SiStripCommissioningSummary/interface/SummaryPlot.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryPlot::SummaryPlot(const std::string& monitorable,
                         const std::string& presentation,
                         const std::string& granularity,
                         const std::string& level)
    : mon_(sistrip::UNKNOWN_MONITORABLE),
      pres_(sistrip::UNKNOWN_PRESENTATION),
      view_(sistrip::UNKNOWN_VIEW),
      gran_(sistrip::UNKNOWN_GRAN),
      level_(""),
      isValid_(false) {
  // Extract enums from strings
  mon_ = SiStripEnumsAndStrings::monitorable(monitorable);
  pres_ = SiStripEnumsAndStrings::presentation(presentation);
  gran_ = SiStripEnumsAndStrings::granularity(granularity);
  level_ = level;

  // Extract view and perform checks
  check();
  isValid_ = true;

  // Checks on member data
  if (mon_ == sistrip::UNKNOWN_MONITORABLE || mon_ == sistrip::UNDEFINED_MONITORABLE) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected monitorable \"" << SiStripEnumsAndStrings::monitorable(mon_)
                                     << "\" based on input string \"" << monitorable << "\"";
    isValid_ = false;
  }

  if (pres_ == sistrip::UNKNOWN_PRESENTATION || pres_ == sistrip::UNDEFINED_PRESENTATION) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected presentation \"" << SiStripEnumsAndStrings::presentation(pres_)
                                     << "\" based on input string \"" << presentation << "\"";
    isValid_ = false;
  }

  if (view_ == sistrip::UNKNOWN_VIEW || view_ == sistrip::UNDEFINED_VIEW) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected view \"" << SiStripEnumsAndStrings::view(level_)
                                     << "\" based on input string \"" << level << "\"";
    isValid_ = false;
  }

  if (level_.empty() || level_.find(sistrip::unknownView_) != std::string::npos ||
      level_.find(sistrip::undefinedView_) != std::string::npos) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected top-level directory: \"" << level_ << "\"";
    isValid_ = false;
  }

  if (gran_ == sistrip::UNKNOWN_GRAN || (gran_ == sistrip::UNDEFINED_GRAN && pres_ != sistrip::HISTO_1D)) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected granularity: \"" << SiStripEnumsAndStrings::granularity(gran_)
                                     << "\" based on input string \"" << granularity << "\"";
    isValid_ = false;
  }
}

// -----------------------------------------------------------------------------
//
SummaryPlot::SummaryPlot(const SummaryPlot& input)
    : mon_(input.monitorable()),
      pres_(input.presentation()),
      view_(input.view()),
      gran_(input.granularity()),
      level_(input.level()),
      isValid_(input.isValid()) {
  ;
}

// -----------------------------------------------------------------------------
//
SummaryPlot::SummaryPlot()
    : mon_(sistrip::UNKNOWN_MONITORABLE),
      pres_(sistrip::UNKNOWN_PRESENTATION),
      view_(sistrip::UNKNOWN_VIEW),
      gran_(sistrip::UNKNOWN_GRAN),
      level_(""),
      isValid_(false) {
  ;
}

// -----------------------------------------------------------------------------
//
void SummaryPlot::reset() {
  mon_ = sistrip::UNKNOWN_MONITORABLE;
  pres_ = sistrip::UNKNOWN_PRESENTATION;
  view_ = sistrip::UNKNOWN_VIEW;
  gran_ = sistrip::UNKNOWN_GRAN;
  level_ = "";
  isValid_ = false;
}

// -----------------------------------------------------------------------------
//
void SummaryPlot::check() {
  // Remove end "/" from level_ if it exists
  if (!level_.empty()) {
    std::string slash = level_.substr(level_.size() - 1, 1);
    if (slash == sistrip::dir_) {
      level_ = level_.substr(0, level_.size() - 1);
    }
  }

  // Extract and check view
  sistrip::View check = SiStripEnumsAndStrings::view(level_);
  view_ = check;
  if (check == sistrip::UNKNOWN_VIEW || check == sistrip::UNDEFINED_VIEW) {
    edm::LogWarning(mlSummaryPlots_) << "[SummaryPlot::" << __func__ << "]"
                                     << " Unexpected view \"" << SiStripEnumsAndStrings::view(check) << "\"";
  }

  // Prepend sistrip::root_ to level_ if not found
  if (level_.find(sistrip::root_) == std::string::npos) {
    if (check == sistrip::UNKNOWN_VIEW) {
      level_ = std::string(sistrip::root_) + sistrip::dir_ + sistrip::unknownView_ + sistrip::dir_ + level_;
    } else if (check == sistrip::UNDEFINED_VIEW) {
      level_ = std::string(sistrip::root_) + sistrip::dir_ + sistrip::undefinedView_ + sistrip::dir_ + level_;
    } else {
      level_ = std::string(sistrip::root_) + sistrip::dir_ + level_;
    }
  }
}

// -----------------------------------------------------------------------------
//
void SummaryPlot::print(std::stringstream& ss) const {
  ss << "[SummaryPlot::" << __func__ << "]" << std::endl
     << " Monitorable:  " << SiStripEnumsAndStrings::monitorable(mon_) << std::endl
     << " Presentation: " << SiStripEnumsAndStrings::presentation(pres_) << std::endl
     << " View:         " << SiStripEnumsAndStrings::view(view_) << std::endl
     << " TopLevelDir:  " << level_ << std::endl
     << " Granularity:  " << SiStripEnumsAndStrings::granularity(gran_) << std::endl;
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<<(std::ostream& os, const SummaryPlot& summary) {
  std::stringstream ss;
  summary.print(ss);
  os << ss.str();
  return os;
}
