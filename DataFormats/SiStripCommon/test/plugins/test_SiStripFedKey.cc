// Last commit: $Id: test_SiStripFedKey.cc,v 1.5 2010/01/07 11:20:59 lowette Exp $

#include "DataFormats/SiStripCommon/test/plugins/test_SiStripFedKey.h"
#include "FWCore/Framework/interface/Event.h" 
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "DataFormats/SiStripCommon/interface/Constants.h" 
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
testSiStripFedKey::testSiStripFedKey( const edm::ParameterSet& pset ) 
{
  LogTrace(mlDqmCommon_)
    << "[testSiStripFedKey::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
testSiStripFedKey::~testSiStripFedKey() {
  LogTrace(mlDqmCommon_)
    << "[testSiStripFedKey::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void testSiStripFedKey::beginJob() {
  
  uint32_t cntr = 0;
  uint32_t start = time(NULL);

  edm::LogInfo(mlDqmCommon_)
    << "[SiStripFedKey::" << __func__ << "]"
    << " Tests the generation of keys...";
  
  // FED ids
  for ( uint16_t ifed = 0; ifed <= sistrip::CMS_FED_ID_MAX+1; ifed++ ) {
    if ( ifed > 1 && 
	 ifed != sistrip::FED_ID_MIN+1 &&
	 ifed < sistrip::CMS_FED_ID_MAX-1 ) { continue; }

    // FE units
    for ( uint16_t ife = 0; ife <= sistrip::FEUNITS_PER_FED+1; ife++ ) {
      if ( ife > 1 && ife < sistrip::FEUNITS_PER_FED ) { continue; }

      // FE channels
      for ( uint16_t ichan = 0; ichan <= sistrip::FEDCH_PER_FEUNIT+1; ichan++ ) {
	if ( ichan > 1 && ichan < sistrip::FEDCH_PER_FEUNIT ) { continue; }

	// APVs 
	for ( uint16_t iapv = 0; iapv <= sistrip::APVS_PER_FEDCH+1; iapv++ ) {
	  if ( iapv > 1 && iapv < sistrip::APVS_PER_FEDCH ) { continue; }

	  // FED channel
	  uint16_t channel = 12 * ife + ichan;
	  
	  // Some debug
	  if ( !(cntr%1000) ) {
	    LogTrace(mlDqmCommon_)
	      << "[SiStripFedKey::" << __func__ << "]"
	      << " Cntr: " << cntr;
	  }
	  cntr++;
	  
	  // Print out FED id/channel
	  std::stringstream ss;
	  ss << std::endl
	     << "[SiStripFedKey::" << __func__ << "]"
	     << " FedId/FeUnit/FeChan/FedCh/APV: "
	     << ifed << "/"
	     << ife << "/"
	     << ichan << "/"
	     << channel << "/"
	     << iapv << std::endl << std::endl;
	  
	  SiStripFedKey tmp1( ifed, ife, ichan, iapv );
	  SiStripFedKey tmp2 = SiStripFedKey( tmp1.key() );
	  SiStripFedKey tmp3 = SiStripFedKey( tmp1.path() );
	  SiStripFedKey tmp4 = SiStripFedKey( tmp1 );
	  SiStripFedKey tmp5; tmp5 = tmp1;
	  
	  ss << ">>> original       : "; tmp1.terse(ss); ss << std::endl;
	  ss << ">>> from FED key   : "; tmp2.terse(ss); ss << std::endl;
	  ss << ">>> from directory : "; tmp3.terse(ss); ss << std::endl;
	  ss << ">>> directory      : " << tmp1.path() << std::endl;
	  ss << ">>> isValid        : " << tmp1.isValid()
	     << " " << tmp1.isValid( tmp1.granularity() )
	     << " " << tmp1.isValid( sistrip::FED_APV ) << std::endl
	     << ">>> isInvalid      : " << tmp1.isInvalid()
	     << " " << tmp1.isInvalid( tmp1.granularity() )
	     << " " << tmp1.isInvalid( sistrip::FED_APV );
	  LogTrace(mlDqmCommon_) << ss.str();
	  
	}
      }
    }
  }
  
  edm::LogVerbatim(mlDqmCommon_)
    << "[SiStripFedKey::" << __func__ << "]"
    << " Processed " << cntr
    << " FedKeys in " << (time(NULL)-start)
    << " seconds at an average rate of " << (cntr*1.) / ((time(NULL)-start)*1.)
    << " per second...";

  // Tests for utility methods

  SiStripFedKey invalid;
  SiStripFedKey inv(sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_,
		    sistrip::invalid_);
  SiStripFedKey valid(51,1,1,1);
  SiStripFedKey all(0,0,0,0);
  SiStripFedKey same(valid);
  SiStripFedKey equal = valid;
  SiStripFedKey equals; 
  equals = valid;

  std::stringstream ss;

  ss << "[SiStripFedKey::" << __func__ << "]"
     << " Tests for utility methods..." << std::endl;

  ss << ">>>> invalid.path: " << invalid << std::endl
     << ">>>> inv.path:     " << inv << std::endl
     << ">>>> valid.path:   " << valid << std::endl
     << ">>>> all.path:     " << all << std::endl
     << ">>>> same.path:    " << same << std::endl
     << ">>>> equal.path:   " << equal << std::endl
     << ">>>> equals.path:  " << equals << std::endl;
  
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
void testSiStripFedKey::analyze( const edm::Event& event, 
				  const edm::EventSetup& setup ) {
  LogTrace(mlDqmCommon_) 
    << "[SiStripFedKey::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}


