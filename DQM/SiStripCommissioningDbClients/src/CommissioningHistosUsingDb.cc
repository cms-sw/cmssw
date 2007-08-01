// Last commit: $Id: CommissioningHistosUsingDb.cc,v 1.2 2007/03/21 16:55:07 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( const DbParams& params )
  : db_(0),
    test_(false)
{
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Constructing object..." << endl;
  
  if ( params.usingDb_ ) {

    // Extract db connections params from CONFDB
    string login = "";
    string passwd = "";
    string path = "";
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
    test_(false)
{
  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Constructing object...";
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
