// Last commit: $Id: test_SiStripNullKey.cc,v 1.3 2008/01/14 09:18:17 bainbrid Exp $

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripNullKey.h"
#include "FWCore/Framework/interface/Event.h" 
#include "DataFormats/SiStripCommon/interface/SiStripNullKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <time.h>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
testSiStripNullKey::testSiStripNullKey( const edm::ParameterSet& pset ) 
{
  LogTrace(mlDqmCommon_)
    << "[testSiStripNullKey::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
testSiStripNullKey::~testSiStripNullKey() {
  LogTrace(mlDqmCommon_)
    << "[testSiStripNullKey::" << __func__ << "]"
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
  if ( base ) {
    ss << ">>>> base->isEqual(invalid):        " << base->isEqual(invalid) << std::endl
       << ">>>> base->isConsistent(invalid):   " << base->isConsistent(invalid) << std::endl;
  }
  
  LogTrace(mlDqmCommon_) << ss.str();

}

// -----------------------------------------------------------------------------
// 
void testSiStripNullKey::analyze( const edm::Event& event, 
				  const edm::EventSetup& setup ) {
  LogTrace(mlDqmCommon_) 
    << "[SiStripNullKey::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}


