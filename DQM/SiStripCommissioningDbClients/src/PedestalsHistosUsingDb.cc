// Last commit: $Id: PedestalsHistosUsingDb.cc,v 1.5 2007/06/19 12:30:37 bainbrid Exp $

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
PedestalsHistosUsingDb::PedestalsHistosUsingDb( MonitorUserInterface* mui,
						SiStripConfigDb* const db )
  : PedestalsHistograms( mui ),
    CommissioningHistosUsingDb( db )
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
  const SiStripConfigDb::FedDescriptions& feds = db_->getFedDescriptions(); 
  update( const_cast<SiStripConfigDb::FedDescriptions&>(feds) );
  if ( !test_ ) { 
    LogTrace(mlDqmClient_) 
      << "[PedestalsHistosUsingDb::" << __func__ << "]"
      << " Uploading pedestals/noise to DB...";
    db_->uploadFedDescriptions(true); 
    LogTrace(mlDqmClient_) 
      << "[PedestalsHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size() 
      << " FED descriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedestalsHistosUsingDb::" << __func__ << "]"
      << " TEST only! No pedestals/noise values will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void PedestalsHistosUsingDb::update( SiStripConfigDb::FedDescriptions& feds ) {
 
  // Iterate through feds and update fed descriptions
  uint16_t updated = 0;
  SiStripConfigDb::FedDescriptions::iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling_->connection( (*ifed)->getFedId(), ichan );
      if ( conn.fecCrate() == sistrip::invalid_ ||
	   conn.fecSlot() == sistrip::invalid_ ||
	   conn.fecRing() == sistrip::invalid_ ||
	   conn.ccuAddr() == sistrip::invalid_ ||
	   conn.ccuChan() == sistrip::invalid_ ||
	   conn.lldChannel() == sistrip::invalid_ ) { continue; }
      SiStripFedKey fed_key( conn.fedId(), 
			     SiStripFedKey::feUnit( conn.fedCh() ),
			     SiStripFedKey::feChan( conn.fedCh() ) );
      SiStripFecKey fec_key( conn.fecCrate(), 
			     conn.fecSlot(), 
			     conn.fecRing(), 
			     conn.ccuAddr(), 
			     conn.ccuChan(), 
			     conn.lldChannel() );

      // Locate appropriate analysis object 
      map<uint32_t,PedestalsAnalysis*>::const_iterator iter = data_.find( fec_key.key() );
      if ( iter != data_.end() ) {

	// Check if analysis is valid
	if ( !iter->second->isValid() ) { continue; }
	
	// Iterate through APVs and strips
	for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
	  for ( uint16_t istr = 0; istr < iter->second->peds()[iapv].size(); istr++ ) { 

	    static float high_threshold = 5.;
	    static float low_threshold  = 5.;
	    static bool  disable_strip  = false;
	    Fed9U::Fed9UStripDescription data( static_cast<uint32_t>( iter->second->peds()[iapv][istr] ), 
					       high_threshold, 
					       low_threshold, 
					       iter->second->noise()[iapv][istr],
					       disable_strip );
	    Fed9U::Fed9UAddress addr( ichan, iapv, istr );
	    (*ifed)->getFedStrips().setStrip( addr, data );

	  }
	}
	updated++;
      
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[PedestalsHistosUsingDb::" << __func__ << "]"
	  << " Unable to find pedestals/noise for FedKey/Id/Ch: " 
	  << hex << setw(8) << setfill('0') << fed_key.key() << dec << "/"
	  << (*ifed)->getFedId() << "/"
	  << ichan
	  << " and device with FEC/slot/ring/CCU/LLD " 
	  << fec_key.fecCrate() << "/"
	  << fec_key.fecSlot() << "/"
	  << fec_key.fecRing() << "/"
	  << fec_key.ccuAddr() << "/"
	  << fec_key.ccuChan() << "/"
	  << fec_key.channel();
      }
    }
  }

  edm::LogVerbatim(mlDqmClient_) 
    << "[PedestalsHistosUsingDb::" << __func__ << "]"
    << " Updated FED pedestals/noise for " 
    << updated << " channels";

}
