// Last commit: $Id: PedsOnlyHistosUsingDb.cc,v 1.8 2012/08/09 17:19:21 eulisse Exp $

#include "DQM/SiStripCommissioningDbClients/interface/PedsOnlyHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/PedsOnlyAnalysis.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
PedsOnlyHistosUsingDb::PedsOnlyHistosUsingDb( const edm::ParameterSet & pset,
                                              DQMStore* bei,
                                              SiStripConfigDb* const db ) 
  : CommissioningHistograms( pset.getParameter<edm::ParameterSet>("PedsOnlyParameters"),
                             bei,
                             sistrip::PEDS_ONLY ),
    CommissioningHistosUsingDb( db,
                                sistrip::PEDS_ONLY ),
    PedsOnlyHistograms( pset.getParameter<edm::ParameterSet>("PedsOnlyParameters"),
                        bei )
{
  LogTrace(mlDqmClient_) 
    << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
PedsOnlyHistosUsingDb::~PedsOnlyHistosUsingDb() {
  LogTrace(mlDqmClient_) 
    << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void PedsOnlyHistosUsingDb::uploadConfigurations() {
  LogTrace(mlDqmClient_) 
    << "[PedsOnlyHistosUsingDb::" << __func__ << "]";

  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Update FED descriptions with new peds/noise values
  SiStripConfigDb::FedDescriptionsRange feds = db()->getFedDescriptions(); 
  update( feds );
  if ( doUploadConf() ) { 
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
      << " Uploading pedestals/noise to DB...";
    db()->uploadFedDescriptions(); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
      << " Completed database upload of " << feds.size() 
      << " FED descriptions!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
      << " TEST only! No pedestals/noise values will be uploaded to DB...";
  }
  
}

// -----------------------------------------------------------------------------
/** */
void PedsOnlyHistosUsingDb::update( SiStripConfigDb::FedDescriptionsRange feds ) {
 
  // Iterate through feds and update fed descriptions
  uint16_t updated = 0;
  SiStripConfigDb::FedDescriptionsV::const_iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    
    for ( uint16_t ichan = 0; ichan < sistrip::FEDCH_PER_FED; ichan++ ) {

      // Build FED and FEC keys
      const FedChannelConnection& conn = cabling()->connection( (*ifed)->getFedId(), ichan );
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
      Analyses::const_iterator iter = data().find( fec_key.key() );
      if ( iter != data().end() ) {

	// Check if analysis is valid
	if ( !iter->second->isValid() ) { continue; }
	
	PedsOnlyAnalysis* anal = dynamic_cast<PedsOnlyAnalysis*>( iter->second );
	if ( !anal ) { 
	  edm::LogError(mlDqmClient_)
	    << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
	    << " NULL pointer to analysis object!";
	  continue; 
	}
	
        // Determine the pedestal shift to apply
        uint32_t pedshift = 127;
        for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
          uint32_t pedmin = (uint32_t) anal->pedsMin()[iapv];
          pedshift = pedmin < pedshift ? pedmin : pedshift;
        }

	// Iterate through APVs and strips
	for ( uint16_t iapv = 0; iapv < sistrip::APVS_PER_FEDCH; iapv++ ) {
	  for ( uint16_t istr = 0; istr < anal->peds()[iapv].size(); istr++ ) { 
	    
	    constexpr float high_threshold = 5.;
	    constexpr float low_threshold  = 2.;
	    constexpr bool  disable_strip  = false;
	    Fed9U::Fed9UStripDescription data( static_cast<uint32_t>( anal->peds()[iapv][istr]-pedshift ), 
					       high_threshold, 
					       low_threshold, 
					       anal->raw()[iapv][istr], //@@ raw noise!
					       disable_strip );
	    Fed9U::Fed9UAddress addr( ichan, iapv, istr );
	    (*ifed)->getFedStrips().setStrip( addr, data );
	    
	  }
	}
	updated++;
      
      } else {
	edm::LogWarning(mlDqmClient_) 
	  << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
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
    << "[PedsOnlyHistosUsingDb::" << __func__ << "]"
    << " Updated FED pedestals/noise for " 
    << updated << " channels";

}

// -----------------------------------------------------------------------------
/** */
void PedsOnlyHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptionsV& desc,
				    Analysis analysis ) {

  PedsOnlyAnalysis* anal = dynamic_cast<PedsOnlyAnalysis*>( analysis->second );
  if ( !anal ) { return; }
  
  SiStripFecKey fec_key( anal->fecKey() );
  SiStripFedKey fed_key( anal->fedKey() );
  
  for ( uint16_t iapv = 0; iapv < 2; ++iapv ) {
    
    // Create description
    PedestalsAnalysisDescription* tmp;
    tmp = new PedestalsAnalysisDescription( std::vector<uint16_t>(0,0), //@@
					    std::vector<uint16_t>(0,0), //@@
					    anal->pedsMean()[iapv],
					    anal->pedsSpread()[iapv],
					    1.*sistrip::invalid_, //@@
					    1.*sistrip::invalid_, //@@
					    anal->rawMean()[iapv],
					    anal->rawSpread()[iapv],
					    anal->pedsMax()[iapv], 
					    anal->pedsMin()[iapv], 
					    1.*sistrip::invalid_, //@@
					    1.*sistrip::invalid_, //@@
					    anal->rawMax()[iapv],
					    anal->rawMin()[iapv],
					    fec_key.fecCrate(),
					    fec_key.fecSlot(),
					    fec_key.fecRing(),
					    fec_key.ccuAddr(),
					    fec_key.ccuChan(),
					    SiStripFecKey::i2cAddr( fec_key.lldChan(), !iapv ), 
					    db()->dbParams().partitions().begin()->second.partitionName(),
					    db()->dbParams().partitions().begin()->second.runNumber(),
					    anal->isValid(),
					    "",
					    fed_key.fedId(),
					    fed_key.feUnit(),
					    fed_key.feChan(),
					    fed_key.fedApv() );
    
    // Add comments
    typedef std::vector<std::string> Strings;
    Strings errors = anal->getErrorCodes();
    Strings::const_iterator istr = errors.begin();
    Strings::const_iterator jstr = errors.end();
    for ( ; istr != jstr; ++istr ) { tmp->addComments( *istr ); }

    // Store description
    desc.push_back( tmp );
      
  }

}

