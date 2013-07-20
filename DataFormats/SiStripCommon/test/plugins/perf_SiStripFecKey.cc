// Last commit: $Id: perf_SiStripFecKey.cc,v 1.4 2010/07/20 02:58:29 wmtan Exp $

#include "DataFormats/SiStripCommon/test/plugins/perf_SiStripFecKey.h"
#include "FWCore/Framework/interface/Event.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
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
perfSiStripFecKey::perfSiStripFecKey( const edm::ParameterSet& pset ) :
  loops_( pset.getUntrackedParameter<uint32_t>( "Loops", 1 ) )
{
  LogTrace(mlDqmCommon_)
    << "[perfSiStripFecKey::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
// 
perfSiStripFecKey::~perfSiStripFecKey() {
  LogTrace(mlDqmCommon_)
    << "[perfSiStripFecKey::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::beginJob() {
  
  edm::LogInfo(mlDqmCommon_)
    << "[SiStripFecKey::" << __func__ << "]"
    << " Tests the generation of keys...";

  std::vector<Value> values;
  std::vector<uint32_t> keys;
  std::vector<std::string> paths;
  std::vector<SiStripFecKey> derived;
  std::vector<SiStripKey> base;

  build(values,keys,paths,derived,base);
  
  build(values);
  build(keys);
  build(paths);
  build(derived);
  build(base);

  test(derived);
  
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::analyze( const edm::Event& event, 
				 const edm::EventSetup& setup ) {
  LogTrace(mlDqmCommon_) 
    << "[SiStripFecKey::" << __func__ << "]"
    << " Analyzing run/event "
    << event.id().run() << "/"
    << event.id().event();
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( const std::vector<Value>& input ) const {
  std::vector<Value>::const_iterator iter = input.begin();
  std::vector<Value>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { SiStripFecKey( iter->crate_,
						  iter->slot_,
						  iter->ring_,
						  iter->ccu_,
						  iter->module_,
						  iter->lld_,
						  iter->i2c_ ); }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( const std::vector<uint32_t>& input ) const {
  std::vector<uint32_t>::const_iterator iter = input.begin();
  std::vector<uint32_t>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { SiStripFecKey key( *iter ); }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( const std::vector<std::string>& input ) const { 
  std::vector<std::string>::const_iterator iter = input.begin();
  std::vector<std::string>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { SiStripFecKey key( *iter ); }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( const std::vector<SiStripFecKey>& input ) const { 
  std::vector<SiStripFecKey>::const_iterator iter = input.begin();
  std::vector<SiStripFecKey>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { SiStripFecKey key( *iter ); }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( const std::vector<SiStripKey>& input ) const { 
  std::vector<SiStripKey>::const_iterator iter = input.begin();
  std::vector<SiStripKey>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { SiStripFecKey key( *iter ); }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::test( const std::vector<SiStripFecKey>& input ) const { 
  std::vector<SiStripFecKey>::const_iterator iter = input.begin();
  std::vector<SiStripFecKey>::const_iterator jter = input.end();
  for ( ; iter != jter; ++iter ) { 
    iter->fecCrate(); 
    iter->fecSlot(); 
    iter->fecRing(); 
    iter->ccuAddr(); 
    iter->ccuChan(); 
    iter->lldChan(); 
    iter->i2cAddr(); 
  }
}

// -----------------------------------------------------------------------------
// 
void perfSiStripFecKey::build( std::vector<Value>& values,
			       std::vector<uint32_t>& keys, 
			       std::vector<std::string>& paths, 
			       std::vector<SiStripFecKey>& derived, 
			       std::vector<SiStripKey>& base ) {
  
  values.clear();
  keys.clear();
  paths.clear();
  derived.clear();
  base.clear();
  
  for ( uint16_t iloop = 0; iloop <= loops_; ++iloop ) {

    if ( !(iloop%10) ) { LogTrace("TEST") << "Nloop: " << iloop; }

    for ( uint16_t icrate = 0; icrate <= sistrip::FEC_CRATE_MAX+1; ++icrate ) {
      if ( icrate > 1 && icrate < sistrip::FEC_CRATE_MAX ) { continue; }

      for ( uint16_t ifec = 0; ifec <= sistrip::SLOTS_PER_CRATE+1; ++ifec ) {
	if ( ifec > 1 && ifec < sistrip::SLOTS_PER_CRATE ) { continue; }

	for ( uint16_t iring = 0; iring <= sistrip::FEC_RING_MAX+1; ++iring ) {
	  if ( iring > 1 && iring < sistrip::FEC_RING_MAX ) { continue; }

	  for ( uint16_t iccu = 0; iccu <= sistrip::CCU_ADDR_MAX+1; ++iccu ) {
	    if ( iccu > 1 && iccu < sistrip::CCU_ADDR_MAX ) { continue; }
	  
	    for ( uint16_t ichan = 0; ichan <= sistrip::CCU_CHAN_MAX+1; ++ichan ) {
	      if ( ichan > 1 && 
		   ichan != sistrip::CCU_CHAN_MIN &&
		   ichan < sistrip::CCU_CHAN_MAX-1 ) { continue; }
	    
	      for ( uint16_t illd = 0; illd <= sistrip::LLD_CHAN_MAX+1; ++illd ) {
		if ( illd > 1 && illd < sistrip::LLD_CHAN_MAX ) { continue; }
	      
		for ( uint16_t iapv = 0; iapv <= sistrip::APV_I2C_MAX+1; ++iapv ) {
		  if ( iapv > 1 && 
		       iapv != sistrip::APV_I2C_MIN &&
		       iapv < sistrip::APV_I2C_MAX ) { continue; }
		
		  SiStripFecKey key( icrate, ifec, iring, iccu, ichan, illd, iapv );
		  values.push_back( Value( key.fecCrate(),
					   key.fecSlot(),
					   key.fecRing(),
					   key.ccuAddr(),
					   key.ccuChan(),
					   key.lldChan(),
					   key.i2cAddr() ) );
		  keys.push_back( key.key() );
		  paths.push_back( key.path() );
		  derived.push_back( key );
		  base.push_back( SiStripKey( key.key() ) );
		}
	      }
	    }
	  }
	}
      }
    }
  }

//   std::sort( values.begin(), values.end() );
//   std::sort( keys.begin(), keys.end() );
//   std::sort( path.begin(), path.end() );
//   std::sort( derived.begin(), derived.end() );
//   std::sort( base.begin(), base.end() );
  
}
