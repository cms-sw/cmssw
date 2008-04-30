// Last commit: $Id: SiStripNoiseBuilderFromDb.cc,v 1.4 2008/03/04 16:42:04 giordano Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripNoiseBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripNoiseBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Fed9UUtils.hh"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripNoiseBuilderFromDb::SiStripNoiseBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripNoiseESSource( pset ),
    db_(0)
{
  LogTrace(mlESSources_) 
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripNoiseBuilderFromDb::~SiStripNoiseBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripNoises* SiStripNoiseBuilderFromDb::makeNoise() {
  LogTrace(mlESSources_) 
    << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Constructing Noise object...";
  
  // Create Noise object 
  SiStripNoises* noise = new SiStripNoises();
  
  // Build and retrieve SiStripConfigDb object using service
  db_ = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE! 
  
  // Check if DB connection is made 
  if ( db_ ) { 

    if ( db_->deviceFactory() ) { 
      
      // Build FEC cabling object
      SiStripFecCabling fec_cabling;
      SiStripFedCablingBuilderFromDb::buildFecCabling( db_, 
						       fec_cabling, 
						       sistrip::CABLING_FROM_CONNS );
      
      // Retrieve DET cabling (should be improved)
      SiStripFedCabling fed_cabling;
      SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, fed_cabling );
      SiStripDetCabling det_cabling( fed_cabling );
      
      // Populate Noise object
      buildNoise( db_, det_cabling, *noise );
      
      // Call virtual method that writes FED cabling object to conditions DB
      //writeNoiseToCondDb( *noise );
      
    } else {
      edm::LogWarning(mlESSources_)
	<< "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build Noise object!";
    }
  } else {
    edm::LogWarning(mlESSources_)
      << "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build Noise object!";
  }
  
  return noise;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripNoiseBuilderFromDb::buildNoise( SiStripConfigDb* const db,
					    const SiStripDetCabling& det_cabling,
					    SiStripNoises& noise ) {
  
  // Retrieve FedDescriptions from configuration database
  SiStripConfigDb::FedDescriptions::range descriptions = db->getFedDescriptions();
  if ( descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripNoiseBuilderFromDb::" << __func__ << "]"
      << " Unable to build Noise object!"
      << " No FED descriptions found!";
    return;
  }
  
  // Retrieve list of active DetIds
  vector<uint32_t> det_ids;
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripNoiseBuilderFromDb::" << __func__ << "]"
      << " Unable to build Noise object!"
      << " No DetIds found!";
    return;
  }  
  LogTrace(mlESSources_)
    << "SiStripNoiseBuilderFromDb::" << __func__ << "]"
    << " Found " << det_ids.size() << " active DetIds";

  // Iterate through active DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {
    
    // Ignore NULL DetIds
    if ( !(*det_id) ) { continue; }
    if ( *det_id == sistrip::invalid32_ ) { continue; }
    
    // Iterate through connections for given DetId and fill peds container
    vector<int16_t> noi;
    const vector<FedChannelConnection>& conns = det_cabling.getConnections(*det_id);
    vector<FedChannelConnection>::const_iterator ipair = conns.begin();
    for ( ; ipair != conns.end(); ipair++ ) {
      
      // Check if the ApvPair is connected
      if ( !(ipair->fedId()) ) {
	edm::LogWarning(mlESSources_)
	  << "SiStripNoiseBuilderFromDb::" << __func__ << "]"
	  << " DetId " << ipair->detId() 
	  << " is missing APV pair number " << ipair->apvPairNumber() 
	  << " out of " << ipair->nApvPairs() << " APV pairs";
	// Fill Noise object with default values
	for ( uint16_t istrip = 0;istrip < sistrip::STRIPS_PER_FEDCH; istrip++ ){
	  noise.setData( 0., noi );
	}
	continue;
      }
      
      // Check if description exists for given FED id 
      SiStripConfigDb::FedDescriptionV::const_iterator description = descriptions.begin();
      while ( description != descriptions.end() ) {
	if ( (*description) && (*description)->getFedId() == ipair->fedId() ) { break; }
	description++;
      }
      if ( description == descriptions.end() ) { 
	edm::LogWarning(mlESSources_)
	  << "SiStripNoiseBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED description for FED id: " << ipair->fedId();
	continue; 
      }
      
      // Retrieve Fed9UStrips object from FED description
      const Fed9U::Fed9UStrips& strips = (*description)->getFedStrips();
      
      // Retrieve StripDescriptions for each APV
      for ( uint16_t iapv = 2*ipair->fedCh(); iapv < 2*ipair->fedCh()+2; iapv++ ) {
	
	// Get StripDescriptions for the given APV
	Fed9U::Fed9UAddress addr;
	addr.setFedApv(iapv);
	vector<Fed9U::Fed9UStripDescription> strip = strips.getApvStrips(addr);
	    
	vector<Fed9U::Fed9UStripDescription>::const_iterator istrip = strip.begin();
	for ( ; istrip != strip.end(); istrip++ ) {
	  
	  noise.setData( istrip->getNoise(),
			 noi );
	  
	} // strip loop
      } // apv loop
    } // connection loop
    
    // Insert pedestal values into Noise object
    if ( !noise.put( *det_id, noi ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripNoiseBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripNoises object!"
	<< " DetId already exists!";
    }
    
  } // det id loop

}
