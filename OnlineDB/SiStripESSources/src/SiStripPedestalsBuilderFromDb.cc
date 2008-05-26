// Last commit: $Id: SiStripPedestalsBuilderFromDb.cc,v 1.7 2008/05/16 15:30:07 bainbrid Exp $
// Latest tag:  $Name: V02-00-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripPedestalsBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripPedestalsBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
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
SiStripPedestalsBuilderFromDb::SiStripPedestalsBuilderFromDb( const edm::ParameterSet& pset ) 
  : SiStripPedestalsESSource( pset ),
    db_(0)
{
  LogTrace(mlESSources_) 
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripPedestalsBuilderFromDb::~SiStripPedestalsBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripPedestals* SiStripPedestalsBuilderFromDb::makePedestals() {
  LogTrace(mlESSources_) 
    << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Constructing Pedestals object...";
  
  // Create Pedestals object 
  SiStripPedestals* pedestals = new SiStripPedestals();
  
  // Build and retrieve SiStripConfigDb object using service
  db_ = edm::Service<SiStripConfigDb>().operator->(); //@@ NOT GUARANTEED TO BE THREAD SAFE! 

  LogTrace(mlConfigDb_) 
    << "TEST db: " << db_;
  
  // Check if DB connection is made 
  if ( db_ ) { 
    
    LogTrace(mlConfigDb_) 
      << "TEST dv: " << db_->deviceFactory();
    
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
      
      // Populate Pedestals object
      LogTrace(mlConfigDb_) 
	<< "TEST db1: " << db_;
      buildPedestals( db_, det_cabling, *pedestals );
      
      // Call virtual method that writes FED cabling object to conditions DB
      //writePedestalsToCondDb( *pedestals );
      
    } else {
      edm::LogWarning(mlESSources_)
	<< "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build Pedestals object!";
    }
  } else {
    edm::LogWarning(mlESSources_)
      << "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build Pedestals object!";
  }
  
  return pedestals;
  
}

// -----------------------------------------------------------------------------
/** */
void SiStripPedestalsBuilderFromDb::buildPedestals( SiStripConfigDb* const db,
						    const SiStripDetCabling& det_cabling,
						    SiStripPedestals& pedestals ) {
  
  // Retrieve FedDescriptions from configuration database
  SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();
  if ( descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripPedestalsBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No FED descriptions found!";
    return;
  }
  
  // Retrieve list of active DetIds
  vector<uint32_t> det_ids;
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripPedestalsBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No DetIds found!";
    return;
  }  
  LogTrace(mlESSources_)
    << "SiStripPedestalsBuilderFromDb::" << __func__ << "]"
    << " Found " << det_ids.size() << " active DetIds";

  // Iterate through active DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {
    
    // Ignore NULL DetIds
    if ( !(*det_id) ) { continue; }
    if ( *det_id == sistrip::invalid32_ ) { continue; }
    
    // Iterate through connections for given DetId and fill peds container
    SiStripPedestals::InputVector peds;
    const vector<FedChannelConnection>& conns = det_cabling.getConnections(*det_id);
    vector<FedChannelConnection>::const_iterator ipair = conns.begin();
    for ( ; ipair != conns.end(); ipair++ ) {
      
      // Check if the ApvPair is connected
      if ( !(ipair->fedId()) ) {
	edm::LogWarning(mlESSources_)
	  << "SiStripPedestalsBuilderFromDb::" << __func__ << "]"
	  << " DetId " << ipair->detId() 
	  << " is missing APV pair number " << ipair->apvPairNumber() 
	  << " out of " << ipair->nApvPairs() << " APV pairs";
	// Fill Pedestals object with default values
	for ( uint16_t istrip = 0;istrip < sistrip::STRIPS_PER_FEDCH; istrip++ ){
	  pedestals.setData(0.,peds );
	}
	continue;
      }
      
      // Check if description exists for given FED id 
      SiStripConfigDb::FedDescriptionsV::const_iterator description = descriptions.begin();
      while ( description != descriptions.end() ) {
	if ( (*description) && (*description)->getFedId() == ipair->fedId() ) { break; }
	description++;
      }
      if ( description == descriptions.end() ) { 
	edm::LogWarning(mlESSources_)
	  << "SiStripPedestalsBuilderFromDb::" << __func__ << "]"
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
	  
	  pedestals.setData( istrip->getPedestal(),
			     peds );
	  
	} // strip loop
      } // apv loop
    } // connection loop
    
    // Insert pedestal values into Pedestals object
    if ( !pedestals.put( *det_id, peds ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripPedestalsBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripPedestals object!"
	<< " DetId already exists!";
    }
    
  } // det id loop

}
