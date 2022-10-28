// user includes
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// system includes
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/**
   @class testSiStripEnumsAndStrings 
   @author R.Bainbridge
   @brief Simple class that tests SiStripEnumsAndStrings.
*/
class testSiStripEnumsAndStrings : public edm::one::EDAnalyzer<> {
public:
  testSiStripEnumsAndStrings(const edm::ParameterSet&);
  ~testSiStripEnumsAndStrings();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
};

using namespace sistrip;

// -----------------------------------------------------------------------------
//
testSiStripEnumsAndStrings::testSiStripEnumsAndStrings(const edm::ParameterSet& pset) {
  LogTrace(mlDqmCommon_) << "[testSiStripEnumsAndStrings::" << __func__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
testSiStripEnumsAndStrings::~testSiStripEnumsAndStrings() {
  LogTrace(mlDqmCommon_) << "[testSiStripEnumsAndStrings::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void testSiStripEnumsAndStrings::beginJob() {
  // sistrip::View
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::View...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::View in = static_cast<sistrip::View>(cntr);
      std::string str = SiStripEnumsAndStrings::view(in);
      sistrip::View out = SiStripEnumsAndStrings::view(str);
      if (out != sistrip::UNKNOWN_VIEW || (out == sistrip::UNKNOWN_VIEW && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
      }
    }
  }

  // sistrip::RunType
  std::vector<sistrip::RunType> run_types;
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::RunType...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::RunType in = static_cast<sistrip::RunType>(cntr);
      std::string str = SiStripEnumsAndStrings::runType(in);
      sistrip::RunType out = SiStripEnumsAndStrings::runType(str);
      if (out != sistrip::UNKNOWN_RUN_TYPE || (out == sistrip::UNKNOWN_RUN_TYPE && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
        if (out != sistrip::UNKNOWN_RUN_TYPE) {
          run_types.push_back(in);
        }
      }
    }
  }

  // sistrip::KeyType
  std::vector<sistrip::KeyType> key_types;
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::KeyType...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::KeyType in = static_cast<sistrip::KeyType>(cntr);
      std::string str = SiStripEnumsAndStrings::keyType(in);
      sistrip::KeyType out = SiStripEnumsAndStrings::keyType(str);
      if (out != sistrip::UNKNOWN_KEY || (out == sistrip::UNKNOWN_KEY && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
        if (out != sistrip::UNKNOWN_KEY) {
          key_types.push_back(in);
        }
      }
    }
  }

  // sistrip::Granularity
  std::vector<sistrip::Granularity> grans;
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::Granularity...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::Granularity in = static_cast<sistrip::Granularity>(cntr);
      std::string str = SiStripEnumsAndStrings::granularity(in);
      sistrip::Granularity out = SiStripEnumsAndStrings::granularity(str);
      if (out != sistrip::UNKNOWN_GRAN || (out == sistrip::UNKNOWN_GRAN && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
        if (out != sistrip::UNKNOWN_GRAN) {
          grans.push_back(in);
        }
      }
    }
  }

  // sistrip::ApvReadoutMode
  std::vector<sistrip::ApvReadoutMode> apv_modes;
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::ApvReadoutMode...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::ApvReadoutMode in = static_cast<sistrip::ApvReadoutMode>(cntr);
      std::string str = SiStripEnumsAndStrings::apvReadoutMode(in);
      sistrip::ApvReadoutMode out = SiStripEnumsAndStrings::apvReadoutMode(str);
      if (out != sistrip::UNKNOWN_APV_READOUT_MODE || (out == sistrip::UNKNOWN_APV_READOUT_MODE && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
        if (out != sistrip::UNKNOWN_APV_READOUT_MODE) {
          apv_modes.push_back(in);
        }
      }
    }
  }

  // sistrip::FedReadoutMode
  std::vector<sistrip::FedReadoutMode> fed_modes;
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::FedReadoutMode...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::FedReadoutMode in = static_cast<sistrip::FedReadoutMode>(cntr);
      std::string str = SiStripEnumsAndStrings::fedReadoutMode(in);
      sistrip::FedReadoutMode out = SiStripEnumsAndStrings::fedReadoutMode(str);
      if (out != sistrip::UNKNOWN_FED_READOUT_MODE || (out == sistrip::UNKNOWN_FED_READOUT_MODE && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
        if (out != sistrip::UNKNOWN_FED_READOUT_MODE) {
          fed_modes.push_back(in);
        }
      }
    }
  }

  // sistrip::Monitorable
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::Monitorable...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::Monitorable in = static_cast<sistrip::Monitorable>(cntr);
      std::string str = SiStripEnumsAndStrings::monitorable(in);
      sistrip::Monitorable out = SiStripEnumsAndStrings::monitorable(str);
      if (out != sistrip::UNKNOWN_MONITORABLE || (out == sistrip::UNKNOWN_MONITORABLE && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
      }
    }
  }

  // sistrip::Presentation
  {
    LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                           << " Checking sistrip::Presentation...";
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::Presentation in = static_cast<sistrip::Presentation>(cntr);
      std::string str = SiStripEnumsAndStrings::presentation(in);
      sistrip::Presentation out = SiStripEnumsAndStrings::presentation(str);
      if (out != sistrip::UNKNOWN_PRESENTATION || (out == sistrip::UNKNOWN_PRESENTATION && first)) {
        first = false;
        std::stringstream ss;
        ss << "[testSiStripEnumsAndStrings::" << __func__ << "]"
           << " cntr: " << std::setw(8) << cntr << "  in: " << std::setw(8) << in << "  out: " << std::setw(8) << out
           << "  str: " << str;
        LogTrace(mlDqmCommon_) << ss.str();
      }
    }
  }
}

// -----------------------------------------------------------------------------
//
void testSiStripEnumsAndStrings::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlDqmCommon_) << "[SiStripEnumsAndStrings::" << __func__ << "]"
                         << " Analyzing run/event " << event.id().run() << "/" << event.id().event();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripEnumsAndStrings);
