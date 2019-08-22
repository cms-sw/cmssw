#include "DQM/SiStripCommissioningSummary/interface/SummaryGeneratorReadoutView.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SummaryGeneratorReadoutView::SummaryGeneratorReadoutView() : SummaryGenerator("SummaryGeneratorReadoutView") { ; }

// -----------------------------------------------------------------------------
//
void SummaryGeneratorReadoutView::fill(const std::string& top_level_dir,
                                       const sistrip::Granularity& granularity,
                                       const uint32_t& device_key,
                                       const float& value,
                                       const float& error) {
  // Check granularity is recognised
  std::string gran = SiStripEnumsAndStrings::granularity(granularity);

  if (granularity != sistrip::UNDEFINED_GRAN && granularity != sistrip::FE_DRIVER && granularity != sistrip::FE_UNIT &&
      granularity != sistrip::FE_CHAN && granularity != sistrip::APV) {
    std::string temp = SiStripEnumsAndStrings::granularity(sistrip::FE_CHAN);
    edm::LogWarning(mlSummaryPlots_) << "[SummaryGeneratorReadoutView::" << __func__ << "]"
                                     << " Unexpected granularity requested: " << gran;
    return;
  }

  // Create key representing "top level" directory
  SiStripFedKey top(top_level_dir);

  // Path and std::string for "present working directory" as defined by device key
  SiStripFedKey path(device_key);
  const std::string& pwd = path.path();

  // Check path is "within" top-level directory structure
  if (top.isValid() && path.isValid() && (path.fedId() == top.fedId() || !top.fedId()) &&
      (path.feUnit() == top.feUnit() || !top.feUnit()) && (path.feChan() == top.feChan() || !top.feChan())) {
    // Extract path and std::string corresponding to "top-level down to granularity"
    std::string sub_dir = pwd;
    size_t pos = pwd.find(gran);
    if (pos != std::string::npos) {
      sub_dir = pwd.substr(0, pwd.find(sistrip::dir_, pos));
    } else if (granularity == sistrip::UNKNOWN_GRAN) {
      sub_dir = pwd;
    }

    SiStripFedKey sub_path(sub_dir);

    //     LogTrace(mlTest_)
    //       << "TEST "
    //       << "sub_path " << sub_path;

    // Construct bin label
    std::stringstream bin;
    if (sub_path.fedId() && sub_path.fedId() != sistrip::invalid_) {
      bin << std::setw(3) << std::setfill('0') << sub_path.fedId();
    }
    if (sub_path.feUnit() && sub_path.feUnit() != sistrip::invalid_) {
      bin << sistrip::dir_ << std::setw(1) << std::setfill('0') << sub_path.feUnit();
    }
    if (sub_path.feChan() && sub_path.feChan() != sistrip::invalid_) {
      bin << sistrip::dir_ << std::setw(2) << std::setfill('0') << sub_path.feChan();
    }
    if (sub_path.fedApv() && sub_path.fedApv() != sistrip::invalid_) {
      bin << sistrip::dir_ << std::setw(1) << std::setfill('0') << sub_path.fedApv();
    }
    //     if ( granularity == sistrip::APV &&
    // 	 path.fedApv() != sistrip::invalid_ ) { bin << sistrip::dot_ << path.fedApv(); }

    // Store "value" in appropriate std::vector within std::map (key is bin label)
    map_[bin.str()].push_back(Data(value, error));
    entries_ += value;
    //     LogTrace(mlTest_)
    //       << "TEST "
    //       << " filling " << bin.str()
    //       << " " << value
    //       << " " << error;

  } else {
    //     std::stringstream ss;
    //     ss << "[SummaryGeneratorReadoutView::" << __func__ << "]"
    //        << " Path for 'pwd' is not within top-level directory!" << std::endl
    //        << "Top-level: " << top << std::endl
    //        << "Path: " << path << std::endl;
    //     edm::LogWarning(mlSummaryPlots_) << ss.str();
  }
}
