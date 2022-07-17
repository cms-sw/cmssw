// system includes
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

// user includes
#include "DataFormats/SiStripCommon/interface/Constants.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoTitle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**
   @class testSiStripHistoTitle 
   @author R.Bainbridge
   @brief Simple class that tests SiStripHistoTitle.
*/
class testSiStripHistoTitle : public edm::one::EDAnalyzer<> {
public:
  testSiStripHistoTitle(const edm::ParameterSet&);
  ~testSiStripHistoTitle();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
};

using namespace sistrip;

// -----------------------------------------------------------------------------
//
testSiStripHistoTitle::testSiStripHistoTitle(const edm::ParameterSet& pset) {
  LogTrace(mlDqmCommon_) << "[testSiStripHistoTitle::" << __func__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
testSiStripHistoTitle::~testSiStripHistoTitle() {
  LogTrace(mlDqmCommon_) << "[testSiStripHistoTitle::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void testSiStripHistoTitle::beginJob() {
  LogTrace(mlDqmCommon_) << "[SiStripHistoTitle::" << __func__ << "]"
                         << " Checking HistoTitle...";

  // Build vectors containing HistoType, RunType, KeyType, Granularity

  std::vector<sistrip::HistoType> histo_types;
  {
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::HistoType in = static_cast<sistrip::HistoType>(cntr);
      std::string str = SiStripEnumsAndStrings::histoType(in);
      sistrip::HistoType out = SiStripEnumsAndStrings::histoType(str);
      if (out != sistrip::UNKNOWN_HISTO_TYPE || (out == sistrip::UNKNOWN_HISTO_TYPE && first)) {
        first = false;
        if (out != sistrip::UNKNOWN_HISTO_TYPE) {
          histo_types.push_back(in);
        }
      }
    }
  }

  std::vector<sistrip::RunType> run_types;
  {
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::RunType in = static_cast<sistrip::RunType>(cntr);
      std::string str = SiStripEnumsAndStrings::runType(in);
      sistrip::RunType out = SiStripEnumsAndStrings::runType(str);
      if (out != sistrip::UNKNOWN_RUN_TYPE || (out == sistrip::UNKNOWN_RUN_TYPE && first)) {
        first = false;
        if (out != sistrip::UNKNOWN_RUN_TYPE) {
          run_types.push_back(in);
        }
      }
    }
  }

  std::vector<sistrip::KeyType> key_types;
  {
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::KeyType in = static_cast<sistrip::KeyType>(cntr);
      std::string str = SiStripEnumsAndStrings::keyType(in);
      sistrip::KeyType out = SiStripEnumsAndStrings::keyType(str);
      if (out != sistrip::UNKNOWN_KEY || (out == sistrip::UNKNOWN_KEY && first)) {
        first = false;
        if (out != sistrip::UNKNOWN_KEY) {
          key_types.push_back(in);
        }
      }
    }
  }

  std::vector<sistrip::Granularity> grans;
  {
    bool first = true;
    for (uint32_t cntr = 0; cntr <= sistrip::invalid_; cntr++) {
      sistrip::Granularity in = static_cast<sistrip::Granularity>(cntr);
      std::string str = SiStripEnumsAndStrings::granularity(in);
      sistrip::Granularity out = SiStripEnumsAndStrings::granularity(str);
      if (out != sistrip::UNKNOWN_GRAN || (out == sistrip::UNKNOWN_GRAN && first)) {
        first = false;
        if (out != sistrip::UNKNOWN_GRAN) {
          grans.push_back(in);
        }
      }
    }
  }

  // Test SiStripHistoTitle class

  std::vector<sistrip::HistoType>::const_iterator ihisto = histo_types.begin();
  for (; ihisto != histo_types.end(); ihisto++) {
    std::vector<sistrip::RunType>::const_iterator irun = run_types.begin();
    for (; irun != run_types.end(); irun++) {
      std::vector<sistrip::KeyType>::const_iterator ikey = key_types.begin();
      for (; ikey != key_types.end(); ikey++) {
        if (*ikey != sistrip::FED_KEY) {
          continue;
        }

        // FED ids
        for (uint16_t ifed = 0; ifed <= sistrip::CMS_FED_ID_MAX + 1; ifed++) {
          if (ifed > 1 && ifed != sistrip::FED_ID_MIN + 1 && ifed < sistrip::CMS_FED_ID_MAX - 1) {
            continue;
          }

          // FE units
          for (uint16_t ife = 0; ife <= sistrip::FEUNITS_PER_FED + 1; ife++) {
            if (ife > 1 && ife < sistrip::FEUNITS_PER_FED) {
              continue;
            }

            // FE channels
            for (uint16_t ichan = 0; ichan <= sistrip::FEDCH_PER_FEUNIT + 1; ichan++) {
              if (ichan > 1 && ichan < sistrip::FEDCH_PER_FEUNIT) {
                continue;
              }

              // APVs
              for (uint16_t iapv = 0; iapv <= sistrip::APVS_PER_FEDCH + 1; iapv++) {
                if (iapv > 1 && iapv < sistrip::APVS_PER_FEDCH) {
                  continue;
                }

                SiStripFedKey key(ifed, ife, ichan, iapv);

                for (uint16_t iextra = 0; iextra < 2; iextra++) {
                  std::string extra = "";
                  if (iextra) {
                    extra = "SomeInfo";
                  }

                  // Generate title objects

                  SiStripHistoTitle in1(*ihisto, *irun, *ikey, key.key(), key.granularity(), key.channel(), extra);

                  SiStripHistoTitle in2(*ihisto, *irun, key, extra);

                  SiStripHistoTitle out1 = SiStripHistoTitle(in1.title());
                  SiStripHistoTitle out2 = SiStripHistoTitle(in2.title());

                  // Debug info

                  std::stringstream ss;

                  ss << std::endl
                     << "[testSiStripHistoTitle::" << __func__ << "]"
                     << " >>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl
                     << std::endl;

                  ss << "[testSiStripHistoTitle::" << __func__ << "]"
                     << " HistoType/RunType/KeyType: " << *ihisto << "/" << *irun << "/" << *ikey
                     << " HistoType: " << SiStripEnumsAndStrings::histoType(*ihisto)
                     << " RunType: " << SiStripEnumsAndStrings::runType(*irun)
                     << " KeyType: " << SiStripEnumsAndStrings::keyType(*ikey) << std::endl;

                  ss << "[testSiStripHistoTitle::" << __func__ << "]"
                     << " FedId/FeUnit/FeChan/APV: " << ifed << "/" << ife << "/" << ichan << "/" << iapv << std::endl;

                  ss << "[testSiStripHistoTitle::" << __func__ << "]"
                     << " Gran/Channel: " << SiStripEnumsAndStrings::granularity(key.granularity()) << "/"
                     << key.channel() << " ExtraInfo: ";
                  if (extra == "") {
                    ss << "(none)" << std::endl;
                  } else {
                    ss << extra << std::endl;
                  }

                  ss << key << std::endl;

                  ss << "IN1:  " << in1 << std::endl;
                  ss << "IN2:  " << in2 << std::endl;
                  ss << "OUT1: " << out1 << std::endl;
                  ss << "OUT2: " << out2 << std::endl;

                  LogTrace(mlDqmCommon_) << ss.str();
                }
              }
            }
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
//
void testSiStripHistoTitle::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlDqmCommon_) << "[SiStripHistoTitle::" << __func__ << "]"
                         << " Analyzing run/event " << event.id().run() << "/" << event.id().event();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripHistoTitle);
