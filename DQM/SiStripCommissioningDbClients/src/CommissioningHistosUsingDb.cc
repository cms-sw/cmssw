// Last commit: $Id: CommissioningHistosUsingDb.cc,v 1.3 2007/04/04 07:21:08 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( const DbParams& params )
  : db_(0),
    cabling_(0),
    test_(false)
{
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Constructing object..." << endl;
  
  if ( params.usingDb_ ) {

    // Extract db connections params from CONFDB
    std::string login = "";
    std::string passwd = "";
    std::string path = "";
    uint32_t ipass = params.confdb_.find("/");
    uint32_t ipath = params.confdb_.find("@");
    if ( ( ipass != std::string::npos ) && 
	 ( ipath != std::string::npos ) ) {
      login = params.confdb_.substr( 0, ipass );
      passwd = params.confdb_.substr( ipass+1, ipath-ipass-1 );
      path = params.confdb_.substr( ipath+1, params.confdb_.size() );
    }
  
    // Create database interface
    if ( login != "" && passwd != "" && path != "" && params.partition_ != "" ) {
      db_ = new SiStripConfigDb( login, 
				 passwd, 
				 path, 
				 params.partition_, 
				 params.major_, 
				 params.minor_ );
      db_->openDbConnection();
    } else {
      edm::LogWarning(mlDqmClient_) 
	<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	<< " Unexpected value for database connection parameters!"
	<< " confdb=" << params.confdb_
	<< " login/passwd@path=" << login << "/" << passwd << "@" << path
	<< " partition=" << params.partition_;
    }
    
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Using a database account!"
      << " SiStripConfigDB ptr: " << db_
      << " confdb: " << params.confdb_
      << " login: " << login
      << " passwd: " << passwd
      << " path: " << path
      << " partition: " << params.partition_
      << " major: " << params.major_
      << " minor: " << params.minor_;
    
  } else {
    
    db_ = new SiStripConfigDb( "", "", "", "" );
    
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Using XML files!"
      << " SiStripConfigDB ptr: " << db_;
    
  }
  
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( SiStripConfigDb* const db )
  : db_(db),
    cabling_(0),
    test_(false)
{
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
  
  // Retrieve DCU-DetId map from DB
  SiStripConfigDb::DcuDetIdMap dcuid_detid_map = db->getDcuDetIdMap();
  
  // Build FEC cabling object from connections found in DB
  SiStripFecCabling fec_cabling;
  SiStripFedCablingBuilderFromDb::buildFecCabling( db_,
						   fec_cabling,
						   dcuid_detid_map );
  
  // Build FED cabling from FEC cabling
  cabling_ = new SiStripFedCabling();
  SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, 
						 *cabling_ );
  
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::~CommissioningHistosUsingDb() {
  if ( db_ ) { delete db_; }
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Destructing object...";
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::DbParams::DbParams() :
  usingDb_(true),
  confdb_(""),
  partition_(""),
  major_(0),
  minor_(0) 
{;}
