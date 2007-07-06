// Last commit: $Id: PedestalsHistosUsingDb.cc,v 1.2 2007/03/21 16:55:07 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/PedestalsHistosUsingDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
PedestalsHistosUsingDb::PedestalsHistosUsingDb( MonitorUserInterface* mui,
						const DbParams& params )
  : PedestalsHistograms( mui ),
    CommissioningHistosUsingDb( params )
{
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistosUsingDb::PedestalsHistosUsingDb( DaqMonitorBEInterface* bei,
						SiStripConfigDb* const db ) 
  : PedestalsHistograms( bei ),
    CommissioningHistosUsingDb( db )
{
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedestalsHistosUsingDb::~PedestalsHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistosUsingDb::uploadToConfigDb() {

  if ( !db_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[PedestalsHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update FED descriptions with new peds/noise values
  db_->resetFedDescriptions();
  const SiStripConfigDb::FedDescriptions& devices = db_->getFedDescriptions(); 
  update( const_cast<SiStripConfigDb::FedDescriptions&>(devices) );
  if ( !test_ ) { db_->uploadFedDescriptions(false); }
  LogTrace(mlDqmClient_) 
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << "Upload of peds/noise constants to DB finished!";
  
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {
 
  // Iterate through feds and update fed descriptions
  SiStripConfigDb::FedDescriptions::iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Retrieve FEC key from FED-FEC map
      uint32_t fec_key = 0;
      uint32_t fed_key = SiStripFedKey( static_cast<uint16_t>((*ifed)->getFedId()), 
					SiStripFedKey::feUnit(ichan),
					SiStripFedKey::feChan(ichan) ).key();
      FedToFecMap::const_iterator ifec = mapping().find(fed_key);
      if ( ifec != mapping().end() ) { fec_key = ifec->second; }
      else {
	edm::LogWarning(mlDqmClient_)
	  << "[PedestalsHistosUsingDb::" << __func__ << "]"
	  << " Unable to find FEC key for FED id/ch: "
	  << (*ifed)->getFedId() << "/" << ichan;
	continue; //@@ write defaults here?... 
      }
      
      map<uint32_t,PedestalsAnalysis>::const_iterator iter = data_.find( fec_key );
      if ( iter != data_.end() ) {

	// Iterate through APVs and strips
	for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
	  for ( uint16_t istr = 0; istr < iter->second.peds()[iapv].size(); istr++ ) { 

	    static float high_threshold = 5.;
	    static float low_threshold  = 5.;
	    static bool  disable_strip  = false;
	    Fed9U::Fed9UStripDescription data( static_cast<uint32_t>( iter->second.peds()[iapv][istr] ), 
					       high_threshold, 
					       low_threshold, 
					       iter->second.noise()[iapv][istr],
					       disable_strip );
	    Fed9U::Fed9UAddress addr( ichan, iapv, istr );
	    (*ifed)->getFedStrips().setStrip( addr, data );

	  }
	}
      
      } else {
	SiStripFecKey path( fec_key );
	edm::LogWarning(mlDqmClient_)
	  << "[PedestalsHistosUsingDb::" << __func__ << "]"
	  << " Unable to find ticker thresholds for FED id/ch: " 
	  << (*ifed)->getFedId() << "/"
	  << ichan << "/"
	  << " and device with at FEC/slot/ring/CCU/LLD channel: " 
	  << path.fecCrate_ << "/"
	     << path.fecSlot_ << "/"
	  << path.fecRing_ << "/"
	  << path.ccuAddr_ << "/"
	  << path.ccuChan_ << "/"
	  << path.channel();
      }
    }
  }

}
