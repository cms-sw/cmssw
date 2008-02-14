// Last commit: $Id: CommissioningHistosUsingDb.cc,v 1.7 2008/02/07 17:02:57 bainbrid Exp $

#include "DQM/SiStripCommissioningDbClients/interface/CommissioningHistosUsingDb.h"
#include "CalibFormats/SiStripObjects/interface/NumberOfDevices.h"
#include "CalibFormats/SiStripObjects/interface/SiStripFecCabling.h"
#include "CondFormats/SiStripObjects/interface/CommissioningAnalysis.h"
#include "CondFormats/SiStripObjects/interface/FedChannelConnection.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "DQMServices/Core/interface/MonitorUserInterface.h"
#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "OnlineDB/SiStripESSources/interface/SiStripFedCablingBuilderFromDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

using namespace sistrip;

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( const DbParams& params )
  : runType_(sistrip::UNDEFINED_RUN_TYPE),
    db_(0),
    cabling_(0),
    uploadAnal_(true),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
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
      edm::LogError(mlDqmClient_) 
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
CommissioningHistosUsingDb::CommissioningHistosUsingDb( SiStripConfigDb* const db,
							sistrip::RunType type )
  : CommissioningHistograms(),
    runType_(type),
    db_(db),
    cabling_(0),
    uploadAnal_(true),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";
  
  // Retrieve DCU-DetId map from DB
  SiStripConfigDb::DcuDetIdMap dcuid_detid_map = db->getDcuDetIdMap();
  
  // Build FEC cabling object from connections found in DB
  SiStripFecCabling fec_cabling;
  if ( runType_ == sistrip::FAST_CABLING ) {
    SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( db_,
								fec_cabling,
								dcuid_detid_map );
  } else {
    SiStripFedCablingBuilderFromDb::buildFecCabling( db_,
						     fec_cabling,
						     dcuid_detid_map );
  }
  
  // Build FED cabling from FEC cabling
  cabling_ = new SiStripFedCabling();
  SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, 
						 *cabling_ );
  std::stringstream ss;
  ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
     << " Terse print out of FED cabling:" << std::endl;
  cabling_->terse(ss);
  LogTrace(mlDqmClient_) << ss.str();
  
  std::stringstream sss;
  sss << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Summary of FED cabling:" << std::endl;
  cabling_->summary(sss);
  edm::LogVerbatim(mlDqmClient_) << sss.str();
  
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb( SiStripConfigDb* const db,
							MonitorUserInterface* const mui,
							sistrip::RunType type )
  : CommissioningHistograms( mui, type ),
    runType_(type),
    db_(db),
    cabling_(0),
    uploadAnal_(true),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object...";
  
  // Retrieve DCU-DetId map from DB
  SiStripConfigDb::DcuDetIdMap dcuid_detid_map = db->getDcuDetIdMap();
  
  // Build FEC cabling object from connections found in DB
  SiStripFecCabling fec_cabling;
  if ( runType_ == sistrip::FAST_CABLING ) {
    SiStripFedCablingBuilderFromDb::buildFecCablingFromDevices( db_,
								fec_cabling,
								dcuid_detid_map );
  } else {
    SiStripFedCablingBuilderFromDb::buildFecCabling( db_,
						     fec_cabling,
						     dcuid_detid_map );
  }
  
  // Build FED cabling from FEC cabling
  cabling_ = new SiStripFedCabling();
  SiStripFedCablingBuilderFromDb::getFedCabling( fec_cabling, 
						 *cabling_ );
  std::stringstream ss;
  ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
     << " Terse print out of FED cabling:" << std::endl;
  cabling_->terse(ss);
  LogTrace(mlDqmClient_) << ss.str();
  
  std::stringstream sss;
  sss << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Summary of FED cabling:" << std::endl;
  cabling_->summary(sss);
  edm::LogVerbatim(mlDqmClient_) << sss.str();
  
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::CommissioningHistosUsingDb()
  : CommissioningHistograms( reinterpret_cast<MonitorUserInterface*>(0), sistrip::UNDEFINED_RUN_TYPE ),
    runType_(sistrip::UNDEFINED_RUN_TYPE),
    db_(0),
    cabling_(0),
    uploadAnal_(false),
    uploadConf_(false)
{
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
    << " Constructing object..." << endl;
}

// -----------------------------------------------------------------------------
/** */
CommissioningHistosUsingDb::~CommissioningHistosUsingDb() {
  if ( db_ ) { delete db_; }
  LogTrace(mlDqmClient_) 
    << "[" << __PRETTY_FUNCTION__ << "]"
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

// -----------------------------------------------------------------------------
/** */
void CommissioningHistosUsingDb::uploadAnalyses() {
  
  if ( !db_ ) {
    edm::LogError(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }
  
  // Upload commissioning analysis results 
  SiStripConfigDb::AnalysisDescriptions anals;
  create( anals );
  db_->createAnalysisDescriptions( anals );

  edm::LogVerbatim(mlDqmClient_) 
    << "[ApvTimingHistosUsingDb::" << __func__ << "]"
    << " Created analysis descriptions for " 
    << anals.size() << " devices";
  
  // Update analysis descriptions with new commissioning results
  if ( uploadAnal_ ) {
    if ( uploadConf_ ) { 
      edm::LogVerbatim(mlDqmClient_)
	<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	<< " Uploading major version of analysis descriptions to DB"
	<< " (will be used for physics)...";
    } else {
      edm::LogVerbatim(mlDqmClient_)
	<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	<< " Uploading minor version of analysis descriptions to DB"
	<< " (will not be used for physics)...";
    }
    db_->uploadAnalysisDescriptions( uploadConf_ ); 
    edm::LogVerbatim(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " Upload of analysis descriptions to DB finished!";
  } else {
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " TEST only! No analysis descriptions will be uploaded to DB...";
  }

}

// -----------------------------------------------------------------------------
/** */
void CommissioningHistosUsingDb::addDcuDetIds() {

  if ( !cabling_ ) {
    edm::LogWarning(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripFedCabling object!";
    return;
  }
  
  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    CommissioningAnalysis* anal = ianal->second;
  
    if ( !anal ) {
      edm::LogWarning(mlDqmClient_) 
	<< "[CommissioningHistosUsingDb::" << __func__ << "]"
	<< " NULL pointer to CommissioningAnalysis object!";
      return;
    }
    
    SiStripFedKey fed_key = anal->fedKey();
    SiStripFecKey fec_key = anal->fecKey();
    
    FedChannelConnection conn = cabling_->connection( fed_key.fedId(),
						      fed_key.fedChannel() );
  
    SiStripFedKey fed( conn.fedId(),
		       SiStripFedKey::feUnit( conn.fedCh() ),
		       SiStripFedKey::feChan( conn.fedCh() ) );
  
    SiStripFecKey fec( conn.fecCrate(),
		       conn.fecSlot(),
		       conn.fecRing(),
		       conn.ccuAddr(),
		       conn.ccuChan(),
		       conn.lldChannel() );
  
    if ( fed_key.path() != fed.path() ) {

      std::stringstream ss;
      ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
	 << " Cannot set DCU and DetId values in commissioning analysis object!" << std::endl
	 << " Incompatible FED key retrieved from cabling!" << std::endl
	 << " FED key from analysis object  : " << fed_key.path() << std::endl
	 << " FED key from cabling object   : " << fed.path() << std::endl
	 << " FED id/ch from analysis object: " << fed_key.fedId() << "/" << fed_key.fedChannel() << std::endl
	 << " FED id/ch from cabling object : " << conn.fedId() << "/" << conn.fedCh();
      edm::LogWarning(mlDqmClient_) << ss.str();

    } else if ( fec_key.path() != fec.path() ) {

      std::stringstream ss;
      ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
	 << " Cannot set DCU and DetId values in commissioning analysis object!" << std::endl
	 << " Incompatible FEC key retrieved from cabling!" << std::endl
	 << " FEC key from analysis object : " << fec_key.path() << std::endl
	 << " FEC key from cabling object  : " << fec.path();
      edm::LogWarning(mlDqmClient_) << ss.str();

    } else {

      anal->dcuId( conn.dcuId() );
      anal->detId( conn.detId() );

    }

  }

}

// -----------------------------------------------------------------------------
//
void CommissioningHistosUsingDb::create( SiStripConfigDb::AnalysisDescriptions& desc ) {

  LogTrace(mlDqmClient_) 
    << "[CommissioningHistosUsingDb::" << __func__ << "]"
    << " Creating AnalysisDescriptions...";

  desc.clear();
  
//   uint16_t size = 0;
//   std::stringstream ss;
//   ss << "[CommissioningHistosUsingDb::" << __func__ << "]"
//      << " Analysis descriptions:" << std::endl;

  Analyses::iterator ianal = data().begin();
  Analyses::iterator janal = data().end();
  for ( ; ianal != janal; ++ianal ) { 

    // create analysis description
    create( desc, ianal ); 
    
//     // debug
//     if ( ianal->second ) {
//       if ( desc.size()/2 > size ) { // print every 2nd description
// 	size = desc.size()/2;
// 	ianal->second->print(ss); 
// 	ss << (*(desc.end()-2))->toString();
// 	ss << (*(desc.end()-1))->toString();
// 	ss << std::endl;
//       }
//     }

  }

//   LogTrace(mlDqmClient_) << ss.str(); 
  
}

// -----------------------------------------------------------------------------
//
void CommissioningHistosUsingDb::detInfo( DetInfoMap& det_info ) {

  if ( !db() ) {
    edm::LogError(mlDqmClient_) 
      << "[CommissioningHistosUsingDb::" << __func__ << "]"
      << " NULL pointer to SiStripConfigDb interface!"
      << " Aborting upload...";
    return;
  }

  // Retrieve DCUs and DetIds
  SiStripConfigDb::DeviceDescriptions dcus = db()->getDeviceDescriptions( DCU ); 
  SiStripConfigDb::DcuDetIdMap detids = db()->getDcuDetIdMap(); 
  
  // Iterate through DCUs
  bool found = false;
  SiStripConfigDb::DeviceDescriptions::const_iterator idcu = dcus.begin();
  SiStripConfigDb::DeviceDescriptions::const_iterator jdcu = dcus.end();
  for ( ; idcu != jdcu; ++idcu ) {

    // Extract descriptions for FEH DCUs
    dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idcu );
    if ( !dcu ) { continue; }
    if ( dcu->getDcuType() != "FEH" ) { continue; }

    // Set DCU id
    DetInfo info;
    info.dcuId_ = dcu->getDcuHardId();

    // Set DetId adn number of APV pairs
    SiStripConfigDb::DcuDetIdMap::const_iterator idet = detids.find( dcu->getDcuHardId() );
    if ( idet != detids.end() ) { 
      info.detId_ = idet->second->getDetId();
      info.pairs_ = idet->second->getApvNumber()/2; 
    }

    // Build FEC key
    const SiStripConfigDb::DeviceAddress& addr = db()->deviceAddress( *dcu );
    SiStripFecKey fec_key( addr.fecCrate_,
			   addr.fecSlot_,
			   addr.fecRing_,
			   addr.ccuAddr_,
			   addr.ccuChan_ );
    
    // Add to map
    if ( fec_key.isValid() ) { det_info[ fec_key.key() ] = info; }

  }
  
}
