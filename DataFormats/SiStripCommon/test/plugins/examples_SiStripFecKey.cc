// Last commit: $Id: examples_SiStripFecKey.cc,v 1.2 2010/01/07 11:20:41 lowette Exp $

#include "DataFormats/SiStripCommon/test/plugins/examples_SiStripFecKey.h"
#include "FWCore/Framework/interface/Event.h" 
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <time.h>
#include <algorithm>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
examplesSiStripFecKey::examplesSiStripFecKey( const edm::ParameterSet& pset ) 
{
  LogTrace(mlDqmCommon_)
    << "[examplesSiStripFecKey::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
examplesSiStripFecKey::~examplesSiStripFecKey() {
  LogTrace(mlDqmCommon_)
    << "[examplesSiStripFecKey::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void examplesSiStripFecKey::beginJob() {
  
  SiStripFecKey invalid;
  SiStripFecKey inv(sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_);
  SiStripFecKey valid(1,2,1,1,16,1,32);
  SiStripFecKey all(0,0,0,0,0,0,0);
  SiStripFecKey same(valid);
  SiStripFecKey equal = valid;
  SiStripFecKey equals; 
  equals = valid;
  SiStripFecKey to_gran(valid,sistrip::CCU_CHAN); 

  std::stringstream ss;

  ss << "[SiStripFecKey::" << __func__ << "]"
     << " Tests for utility methods..." << std::endl;

  ss << ">>>> invalid.path: " << invalid << std::endl
     << ">>>> inv.path:     " << inv << std::endl
     << ">>>> valid.path:   " << valid << std::endl
     << ">>>> all.path:     " << all << std::endl
     << ">>>> same.path:    " << same << std::endl
     << ">>>> equal.path:   " << equal << std::endl
     << ">>>> equals.path:  " << equals << std::endl
     << ">>>> to_gran.path: " << to_gran << std::endl;
  
  ss << std::hex
     << ">>>> invalid.key:  " << invalid.key() << std::endl
     << ">>>> valid.key:    " << valid.key() << std::endl
     << ">>>> all.key:      " << all.key() << std::endl
     << std::dec;
  
  ss << ">>>> invalid.isInvalid: " << invalid.isInvalid() << std::endl
     << ">>>> invalid.isValid:   " << invalid.isValid() << std::endl
     << ">>>> valid.isInvalid:   " << valid.isInvalid() << std::endl
     << ">>>> valid.isValid:     " << valid.isValid() << std::endl
     << ">>>> all.isInvalid:     " << all.isInvalid() << std::endl
     << ">>>> all.isValid:       " << all.isValid() << std::endl;

  ss << ">>>> valid.isEqual(valid):        " << valid.isEqual(valid) << std::endl
     << ">>>> valid.isConsistent(valid):   " << valid.isConsistent(valid) << std::endl
     << ">>>> valid.isEqual(invalid):      " << valid.isEqual(invalid) << std::endl
     << ">>>> valid.isConsistent(invalid): " << valid.isConsistent(invalid) << std::endl
     << ">>>> valid.isEqual(all):          " << valid.isEqual(all) << std::endl
     << ">>>> valid.isConsistent(all):     " << valid.isConsistent(all) << std::endl
     << ">>>> valid.isEqual(same):         " << valid.isEqual(same) << std::endl
     << ">>>> valid.isEqual(equal):        " << valid.isEqual(equal) << std::endl
     << ">>>> valid.isEqual(equals):       " << valid.isEqual(equals) << std::endl;
  
  LogTrace(mlDqmCommon_) << ss.str();
  
}

// -----------------------------------------------------------------------------
// 
void examplesSiStripFecKey::analyze( const edm::Event& event, 
				  const edm::EventSetup& setup ) {
  LogTrace(mlDqmCommon_) 
    << "[SiStripFecKey::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}
