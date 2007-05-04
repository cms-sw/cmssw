// Last commit: $Id:$
// Latest tag:  $Name:$
// Location:    $Source:$

#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderUsingDbService.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/SiStripCommon/interface/SiStripHistoNamingScheme.h"
#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderUsingDbService::SiStripFedCablingBuilderUsingDbService( const edm::ParameterSet& pset ) 
  : SiStripFedCablingBuilderFromDb( pset )
{
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]"
    << " Constructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCablingBuilderUsingDbService::~SiStripFedCablingBuilderUsingDbService() {
  LogTrace(mlCabling_)
    << "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
SiStripFedCabling* SiStripFedCablingBuilderUsingDbService::makeFedCabling() {
  LogTrace(mlCabling_) 
    << "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]"
    << " Constructing FED cabling...";

  // Create FED cabling object 
  SiStripFedCabling* fed_cabling = new SiStripFedCabling();
  
  // Build and retrieve SiStripConfigDb object using service
  //@@ NOT GUARANTEED TO BE THREAD SAFE!!! NEED DB CACHE/CLIENT
  db_ = edm::Service<SiStripConfigDb>().operator->();
  
  // Check if DB connection is made 
  if ( db_ ) { 

    if ( db_->deviceFactory() ) { 

      // Build FEC cabling object
      SiStripFecCabling fec_cabling;
      SiStripConfigDb::DcuDetIdMap dcu_detid_map;
      buildFecCabling( db_, fec_cabling, dcu_detid_map, source_ );

      // Populate FED cabling object
      getFedCabling( fec_cabling, *fed_cabling );
      
      // Call virtual method that writes FED cabling object to conditions DB
      writeFedCablingToCondDb( *fed_cabling );
      
      // Prints FED cabling
      stringstream ss;
      ss << "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]" 
	 << " Printing cabling map..." << endl 
	 << *fed_cabling;
      LogTrace(mlCabling_) << ss.str();
      
    } else {
      edm::LogWarning(mlCabling_)
	<< "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]"
	<< " NULL pointer to DeviceFactory returned by SiStripConfigDb!"
	<< " Cannot build FED cabling object!";
    }
  } else {
    edm::LogWarning(mlCabling_)
      << "[SiStripFedCablingBuilderUsingDbService::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb returned by DB \"service\"!"
      << " Cannot build FED cabling object!";
  }
  
  return fed_cabling;
  
}
