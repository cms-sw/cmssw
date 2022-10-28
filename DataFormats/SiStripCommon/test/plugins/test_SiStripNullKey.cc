// system includes
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>

// user includes
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

/**
   @class testSiStripNullKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripNullKey.
*/
class testSiStripNullKey : public edm::one::EDAnalyzer<> {
public:
  testSiStripNullKey(const edm::ParameterSet&);
  ~testSiStripNullKey();

  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
};

using namespace sistrip;

// -----------------------------------------------------------------------------
//
testSiStripNullKey::testSiStripNullKey(const edm::ParameterSet& pset) {
  LogTrace(mlDqmCommon_) << "[testSiStripNullKey::" << __func__ << "]"
                         << " Constructing object...";
}

// -----------------------------------------------------------------------------
//
testSiStripNullKey::~testSiStripNullKey() {
  LogTrace(mlDqmCommon_) << "[testSiStripNullKey::" << __func__ << "]"
                         << " Destructing object...";
}

// -----------------------------------------------------------------------------
//
void testSiStripNullKey::beginJob() {
  // Tests for utility methods

  SiStripNullKey invalid;
  SiStripNullKey same(invalid);
  SiStripNullKey equal = invalid;
  SiStripNullKey equals;
  equals = invalid;
  SiStripKey* base = static_cast<SiStripKey*>(&invalid);

  std::stringstream ss;

  ss << "[SiStripNullKey::" << __func__ << "]"
     << " Tests for utility methods..." << std::endl;

  ss << ">>>> invalid: " << invalid << std::endl
     << ">>>> same:    " << same << std::endl
     << ">>>> equal:   " << equal << std::endl
     << ">>>> equals:  " << equals << std::endl
     << ">>>> base:    " << *base << std::endl;

  ss << ">>>> invalid.isInvalid: " << invalid.isInvalid() << std::endl
     << ">>>> invalid.isValid:   " << invalid.isValid() << std::endl;

  ss << ">>>> invalid.isEqual(invalid):      " << invalid.isEqual(invalid) << std::endl
     << ">>>> invalid.isConsistent(invalid): " << invalid.isConsistent(invalid) << std::endl
     << ">>>> invalid.isEqual(same):         " << invalid.isEqual(same) << std::endl
     << ">>>> invalid.isEqual(equal):        " << invalid.isEqual(equal) << std::endl
     << ">>>> invalid.isEqual(equals):       " << invalid.isEqual(equals) << std::endl;
  if (base) {
    ss << ">>>> base->isEqual(invalid):        " << base->isEqual(invalid) << std::endl
       << ">>>> base->isConsistent(invalid):   " << base->isConsistent(invalid) << std::endl;
  }

  LogTrace(mlDqmCommon_) << ss.str();
}

// -----------------------------------------------------------------------------
//
void testSiStripNullKey::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  LogTrace(mlDqmCommon_) << "[SiStripNullKey::" << __func__ << "]"
                         << " Analyzing run/event " << event.id().run() << "/" << event.id().event();
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testSiStripNullKey);
