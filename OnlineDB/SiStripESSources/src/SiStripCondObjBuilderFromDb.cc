// Last commit: $Id: SiStripCondObjBuilderFromDb.cc,v 1.4 2008/05/26 13:37:26 giordano Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/src/SiStripCondObjBuilderFromDb.cc,v $

#include "OnlineDB/SiStripESSources/interface/SiStripCondObjBuilderFromDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripPedestals.h"
#include "CondFormats/SiStripObjects/interface/SiStripNoises.h"
#include "CondFormats/SiStripObjects/interface/SiStripThreshold.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "Fed9UUtils.hh"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb(const edm::ParameterSet&,
							 const edm::ActivityRegistry&) 
{
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripCondObjBuilderFromDb::SiStripCondObjBuilderFromDb() 
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
void SiStripCondObjBuilderFromDb::checkUpdate() {
  if (!(dbParams_==dbParams())){
    dbParams_=dbParams();
    buildCondObj();
  }  
}

// -----------------------------------------------------------------------------
/** */
void SiStripCondObjBuilderFromDb::buildCondObj() {
  LogTrace(mlESSources_) 
    << "[SiStripCondObjBuilderFromDb::" << __func__ << "]";

  /*
  LogTrace(mlConfigDb_) 
    << "TEST db: " << db_;
  */

  // Check if DB connection is made 
  if ( db_ ) { 

    /*    
    LogTrace(mlConfigDb_) 
      << "TEST dv: " << db_->deviceFactory();
    */

    if ( db_->deviceFactory() ) { 
      
      // Build FEC cabling object
      SiStripFecCabling fec_cabling;
      SiStripFedCablingBuilderFromDb::buildFecCabling( &*db_, 
						       fec_cabling, 
						       sistrip::CABLING_FROM_CONNS );
      
      // Retrieve DET cabling (should be improved)
      fed_cabling_=new SiStripFedCabling;
      SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, *fed_cabling_ );
      SiStripDetCabling det_cabling( *fed_cabling_ );
      
      /* 
      // Populate Pedestals object
      LogTrace(mlConfigDb_) 
	<< "TEST db1: " << db_;
      */

      buildStripRelatedObjects( &*db_, det_cabling );
      
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
  SiStripConfigDb::FedDescriptionsRange descriptions = db->getFedDescriptions();
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
    
    //if(*det_id==369158216)
    //edm::LogWarning(mlESSources_) << "TEST this is my detid " << *det_id << std::endl;
  
    const vector<FedChannelConnection>& conns = det_cabling.getConnections(*det_id);
    if (conns.size()==0){
      edm::LogWarning(mlESSources_)
	<< "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	<< " Unable to build Pedestals object!"
	<< " No FED channel connections found for detid "<< *det_id;
      continue;
    }

    vector<FedChannelConnection>::const_iterator ipair = conns.begin();
    vector< vector<FedChannelConnection>::const_iterator > listConns(ipair->nApvPairs(),conns.end());
    for ( ; ipair != conns.end(); ipair++ ){ 
      // Check if the ApvPair is connected
      if (ipair->fedId() && ipair->apvPairNumber()<3){
	//	if(*det_id==369158216 || 369124437==*det_id)
	//  edm::LogWarning(mlESSources_) << "TEST this is the position of the listConns for detid " << *det_id << "  " << ipair-conns.begin() << " " << ipair->apvPairNumber();
	listConns[ipair-conns.begin()]=ipair;
      } else {
	edm::LogWarning(mlESSources_)
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " DetId " << ipair->detId() 
	  << " is missing \n a) APV pair number " << ipair->apvPairNumber() 
	  << " out of " << ipair->nApvPairs() << " APV pairs\n or \n b) fedId " << ipair->fedId();
      } 
    }

    //if(*det_id==369158216)
    //  edm::LogWarning(mlESSources_) << "TEST this is my  vector<FedChannelConnection> size " << conns.size() << " listConn.size() " << listConns.size()<< std::endl;
    

    // Iterate through connections for given DetId and fill peds container
    SiStripPedestals::InputVector inputPedestals;
    SiStripNoises::InputVector inputNoises;
    SiStripThreshold::InputVector inputThreshold;
    SiStripQuality::InputVector inputQuality;

    uint16_t apvPair;
    vector< vector<FedChannelConnection>::const_iterator >::const_iterator ilistConns=listConns.begin();
    for ( ; ilistConns != listConns.end(); ++ilistConns ) {
      ipair=*ilistConns;
      apvPair=(ilistConns-listConns.begin());

      if ( ipair == conns.end() ) {
	// Fill object with default values
	edm::LogWarning(mlESSources_)
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED connection for detid : " << *det_id << " APV pair number " << apvPair
	  << " Writing default values";
	uint16_t istrip = apvPair*sistrip::STRIPS_PER_FEDCH;  
	inputQuality.push_back(quality_->encode(istrip,sistrip::STRIPS_PER_FEDCH));
	threshold_->setData( istrip, 0., 0., inputThreshold );
	for ( ;istrip < (apvPair+1)*sistrip::STRIPS_PER_FEDCH; ++istrip ){
	  pedestals_->setData( 0.,inputPedestals );
	  noises_->setData( 0., inputNoises );
	  //edm::LogWarning(mlESSources_) << "TEST default values for " << *det_id << " strip " << istrip << std::endl;
	}
	continue;
      }
      
      //if(*det_id==369158216)
      //edm::LogWarning(mlESSources_) << "TEST this is my  vector<FedChannelConnection> entry " 
      //<< ipair-conns.begin() << " ilistConn " << ilistConns - listConns.begin()<< std::endl;
      
      // Check if description exists for given FED id 
      SiStripConfigDb::FedDescriptionsV::const_iterator description = descriptions.begin();
      while ( description != descriptions.end() ) {
	if ( (*description)->getFedId() ==ipair->fedId() ) { break; }
	description++;
      }
      if ( description == descriptions.end() ) { 
	edm::LogWarning(mlESSources_)
	  << "SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to find FED description for FED id: " << ipair->fedId() << " detid : " << *det_id << " APV pair number " << apvPair
	  << " Writing default values";
	uint16_t istrip = apvPair*sistrip::STRIPS_PER_FEDCH;  
	inputQuality.push_back(quality_->encode(istrip,sistrip::STRIPS_PER_FEDCH));
	threshold_->setData( istrip, 0., 0., inputThreshold );
	for ( ;istrip < (apvPair+1)*sistrip::STRIPS_PER_FEDCH; ++istrip ){
	  pedestals_->setData( 0.,inputPedestals );
	  noises_->setData( 0., inputNoises );
	  //edm::LogWarning(mlESSources_) << "TEST default values for " << *det_id << " strip " << istrip << std::endl;
	}
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
	
	//if(*det_id==369158216)
	//edm::LogWarning(mlESSources_) << "TEST this is my apvPairNumber " <<    ipair->apvPairNumber()<< " out of " << ipair->nApvPairs() << std::endl;

	vector<Fed9U::Fed9UStripDescription>::const_iterator istrip = strip.begin();
	uint16_t jstrip = ipair->apvPairNumber()*sistrip::STRIPS_PER_FEDCH;
	for ( ; istrip != strip.end(); istrip++ ) {
	  //if(*det_id==369158216 || 369124437==*det_id)
	  //edm::LogWarning(mlESSources_) << "TEST this is ped " << *det_id << " strip " << jstrip << " value " << istrip->getPedestal() << std::endl;

	  pedestals_->setData( istrip->getPedestal() , inputPedestals);
	  noises_   ->setData( istrip->getNoise()    , inputNoises );
	  threshold_->setData( jstrip, istrip->getLowThresholdFactor(),
			       istrip->getHighThresholdFactor(), inputThreshold );
	  if(istrip->getDisable())
	    inputQuality.push_back(quality_->encode(jstrip,1.));
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
    if (inputQuality.size()){
      quality_->compact(detid,inputQuality);
      if ( !quality_->put( *det_id, inputQuality ) ) {
	edm::LogWarning(mlESSources_)
	  << "[SiStripCondObjBuilderFromDb::" << __func__ << "]"
	  << " Unable to insert values into SiStripThreshold object!"
	  << " DetId already exists!";
      }
    }

    
  } // det id loop

}
