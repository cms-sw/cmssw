// Last commit: $Id: SiStripPedestalsBuilderFromDb.cc,v 1.3 2007/11/09 14:40:44 bainbrid Exp $
// Latest tag:  $Name: V01-01-00 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripPedestalsBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
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
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb() 
  : db_(0)
{
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::~SiStripCondObjBuilderFromDb() {
  LogTrace(mlESSources_)
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildCondObj() {
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]";
  
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
      SiStripConfigDb::DcuDetIdMap dcu_detid;
      SiStripFedCablingBuilderFromDb::buildFecCabling( db_, 
						       fec_cabling, 
						       dcu_detid, 
						       sistrip::CABLING_FROM_CONNS );
      
      // Retrieve DET cabling (should be improved)
      fed_cabling_=new SiStripFedCabling;
      SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, *fed_cabling_ );
      SiStripDetCabling det_cabling( *fed_cabling_ );
      
      // Populate Pedestals object
      LogTrace(mlConfigDb_) 
	<< "TEST db1: " << db_;

      buildStripRelatedObjects( db_, det_cabling);
      
      // Call virtual method that writes FED cabling object to conditions DB
      //writePedestalsToCondDb( *pedestals );
      
    } else {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build Pedestals object!";
    }
  } else {
    edm::LogWarning(mlESSources_)
      << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build Pedestals object!";
  }
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildStripRelatedObjects( SiStripConfigDb* const db,
							    const SiStripDetCabling& det_cabling){
  
  // Retrieve FedDescriptions from configuration database
  const SiStripConfigDb::FedDescriptions& descriptions = db->getFedDescriptions();
  if ( descriptions.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No FED descriptions found!";
    return;
  }
  
  // Retrieve list of active DetIds
  vector<uint32_t> det_ids;
  det_cabling.addActiveDetectorsRawIds(det_ids);
  if ( det_ids.empty() ) {
    edm::LogWarning(mlESSources_)
      << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
      << " Unable to build Pedestals object!"
      << " No DetIds found!";
    return;
  }  
  LogTrace(mlESSources_)
    << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Found " << det_ids.size() << " active DetIds";

  pedestals_=new SiStripPedestals();
  noises_=new SiStripNoises();
  threshold_= new SiStripThreshold();
  quality_=new SiStripQuality();

  // Iterate through active DetIds
  vector<uint32_t>::const_iterator det_id = det_ids.begin();
  for ( ; det_id != det_ids.end(); det_id++ ) {
    
    // Ignore NULL DetIds
    if ( !(*det_id) ) { continue; }
    if ( *det_id == sistrip::invalid32_ ) { continue; }
    
    // Iterate through connections for given DetId and fill peds container
    SiStripPedestals::InputVector inputPedestals;
    SiStripNoises::InputVector inputNoises;
    SiStripThreshold::InputVector inputThreshold;
    SiStripQuality::InputVector inputQuality;


    const vector<FedChannelConnection>& conns = det_cabling.getConnections(*det_id);
    vector<FedChannelConnection>::const_iterator ipair = conns.begin();
    for ( ; ipair != conns.end(); ipair++ ) {
      
      // Check if the ApvPair is connected
      if ( !(ipair->fedId()) ) {
	edm::LogWarning(mlESSources_)
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " DetId " << ipair->detId() 
	  << " is missing APV pair number " << ipair->apvPairNumber() 
	  << " out of " << ipair->nApvPairs() << " APV pairs";
	// Fill object with default values
	for ( uint16_t istrip = ipair->apvPairNumber()*sistrip::STRIPS_PER_FEDCH;istrip < (ipair->apvPairNumber()+1)*sistrip::STRIPS_PER_FEDCH; istrip++ ){
	  pedestals_->setData( 0.,inputPedestals );
	  noises_->setData( 0., inputNoises );
	  threshold_->setData( istrip, 0., 0., inputThreshold );
	  inputQuality.push_back(istrip);
	}
	continue;
      }
      
      // Check if description exists for given FED id 
      SiStripConfigDb::FedDescriptions::const_iterator description = descriptions.begin();
      while ( description != descriptions.end() ) {
	if ( (*description)->getFedId() ==ipair->fedId() ) { break; }
	description++;
      }
      if ( description == descriptions.end() ) { 
	edm::LogWarning(mlESSources_)
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
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
	uint16_t jstrip = ipair->apvPairNumber()*sistrip::STRIPS_PER_FEDCH;
	for ( ; istrip != strip.end(); istrip++ ) {
	  pedestals_->setData( istrip->getPedestal() , inputPedestals);
	  noises_   ->setData( istrip->getNoise()    , inputNoises );
	  threshold_->setData( jstrip, istrip->getLowThresholdFactor(),
			       istrip->getHighThresholdFactor(), inputThreshold );
	  if(istrip->getDisable())
	    inputQuality.push_back(jstrip);
	  jstrip++;
	} // strip loop
      } // apv loop
    } // connection loop
    
    // Insert pedestal values into Pedestals object
    if ( !pedestals_->put( *det_id, inputPedestals ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripPedestals object!"
	<< " DetId already exists!";
    }

    // Insert noise values into Noises object
    if ( !noises_->put( *det_id, inputNoises ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripNoises object!"
	<< " DetId already exists!";
    }

    // Insert threshold values into Threshold object
    if ( !threshold_->put( *det_id, inputThreshold ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripThreshold object!"
	<< " DetId already exists!";
    }

    // Insert quality values into Quality object
    uint32_t detid=*det_id;
    quality_->compact(detid,inputQuality);
    if ( !quality_->put( *det_id, inputQuality ) ) {
      edm::LogWarning(mlESSources_)
	<< "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to insert values into SiStripThreshold object!"
	<< " DetId already exists!";
    }

    
  } // det id loop

}
