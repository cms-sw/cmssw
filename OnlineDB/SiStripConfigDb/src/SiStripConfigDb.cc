// Last commit: $Id: SiStripConfigDb.cc,v 1.62 2008/04/24 16:10:05 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <fstream>

using namespace sistrip;

// -----------------------------------------------------------------------------
// 
uint32_t SiStripConfigDb::cntr_ = 0;

// -----------------------------------------------------------------------------
// 
bool SiStripConfigDb::allowCalibUpload_ = false;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( const edm::ParameterSet& pset,
				  const edm::ActivityRegistry& activity ) :
  factory_(0), 
  dbCache_(0), 
  dbParams_(),
  // Local cache
  connections_(), 
  devices_(), 
  feds_(), 
  dcuDetIdMap_(), 
#ifdef USING_NEW_DATABASE_MODEL
  analyses_(),
#endif
  typedDevices_(), 
  fedIds_(),
  // Misc
  usingStrips_(true),
  openConnection_(false)
{
  cntr_++;
  edm::LogVerbatim(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Constructing database service..."
    << " (Class instance: " << cntr_ << ")";
  
  // Set DB connection parameters
  dbParams_.reset();
  dbParams_.pset( pset );
  //edm::LogVerbatim(mlConfigDb_) << dbParams_; 

  // Open connection
  openDbConnection();
  
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  closeDbConnection(); 
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Destructing object...";
  if ( cntr_ ) { cntr_--; }
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress::DeviceAddress() : 
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_), 
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_), 
  ccuChan_(sistrip::invalid_), 
  lldChan_(sistrip::invalid_), 
  i2cAddr_(sistrip::invalid_),
  fedId_(sistrip::invalid_),
  feUnit_(sistrip::invalid_),
  feChan_(sistrip::invalid_)
{ reset(); }

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DeviceAddress::reset() { 
  fecCrate_ = sistrip::invalid_; 
  fecSlot_  = sistrip::invalid_; 
  fecRing_  = sistrip::invalid_; 
  ccuAddr_  = sistrip::invalid_; 
  ccuChan_  = sistrip::invalid_; 
  lldChan_  = sistrip::invalid_;
  i2cAddr_  = sistrip::invalid_;
  fedId_    = sistrip::invalid_;
  feUnit_   = sistrip::invalid_;
  feChan_   = sistrip::invalid_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::openDbConnection() {

  LogTrace(mlConfigDb_) 
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Opening connection to database...";
  
  // Check if connection already exists
  if ( openConnection_ ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Connection already open!";
    return;
  }
  openConnection_ = true;

  // Establish database connection
  if ( dbParams_.usingDb_ ) { 
    if ( dbParams_.usingDbCache_ ) { usingDatabaseCache(); }
    else { usingDatabase(); }
  } else { usingXmlFiles(); }

  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters: "
     << std::endl << dbParams_;
  edm::LogVerbatim(mlConfigDb_) << ss.str();

  // Clear local caches
  clearLocalCache();

  LogTrace(mlConfigDb_) 
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Opened connection to database!";
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::closeDbConnection() {

  LogTrace(mlConfigDb_) 
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Closing connection to database...";

  // Check if connection exists
  if ( !openConnection_ ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No connection open!";
    return;
  }
  openConnection_ = false;
  
  // Clear local caches
  clearLocalCache();
  
  try { 
    if ( factory_ ) { delete factory_; }
  } catch (...) { handleException( __func__, "Attempting to delete DeviceFactory object..." ); }
  factory_ = 0; 

#ifdef USING_NEW_DATABASE_MODEL
  try { 
    if ( dbCache_ ) { delete dbCache_; }
  } catch (...) { handleException( __func__, "Attempting to delete DbClient object..." ); }
  dbCache_ = 0; 
#endif
  
  LogTrace(mlConfigDb_) 
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Closed connection to database...";
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::clearLocalCache() {

  LogTrace(mlConfigDb_) 
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Clearing local caches...";

  clearFedConnections();
  clearDeviceDescriptions();
  clearFedDescriptions();
  clearDcuDetIdMap();
#ifdef USING_NEW_DATABASE_MODEL
  analyses_.clear();
#endif
  typedDevices_.clear();
  fedIds_.clear();

}

// -----------------------------------------------------------------------------
//
DeviceFactory* const SiStripConfigDb::deviceFactory( std::string method_name ) const { 
  if ( factory_ ) { return factory_; }
  else { 
    if ( method_name != "" ) { 
      stringstream ss;
      ss << "[SiStripConfigDb::" << __func__ << "]"
	 << " NULL pointer to DeviceFactory requested by" 
	 << " method SiStripConfigDb::" << method_name << "()!";
      edm::LogWarning(mlConfigDb_) << ss.str();
    }
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
DbClient* const SiStripConfigDb::databaseCache( std::string method_name ) const { 
  if ( dbCache_ ) { return dbCache_; }
  else { 
    if ( method_name != "" ) { 
      stringstream ss;
      ss << "[SiStripConfigDb::" << __func__ << "]"
	 << " NULL pointer to DbClient requested by" 
	 << " method SiStripConfigDb::" << method_name << "()!";
      edm::LogWarning(mlConfigDb_) << ss.str();
    }
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {
  
  // Retrieve connection params from CONFDB env. var. and override .cfg values 
  std::string user = "";
  std::string passwd = "";
  std::string path = "";
  DbAccess::getDbConfiguration( user, passwd, path );
  if ( user != "" && passwd != "" && path != "" ) {

    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"user/passwd@path\" to \""
       << user << "/" << passwd << "@" << path
       << "\" using 'CONFDB' environmental variable";
    if ( dbParams_.user_ != null_ || 
	 dbParams_.passwd_ != null_ || 
	 dbParams_.path_ != null_ ) { 
      ss << " (Overwriting existing value of \""
	 << dbParams_.user_ << "/" 
	 << dbParams_.passwd_ << "@" 
	 << dbParams_.path_ 
	 << "\" read from .cfg file)";
    }
    LogTrace(mlConfigDb_) << ss.str() << std::endl;
    dbParams_.confdb( user, passwd, path );

  } else if ( dbParams_.user_ != null_ && 
	      dbParams_.passwd_ != null_ && 
	      dbParams_.path_ != null_ ) { 

    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"user/passwd@path\" to \""
       << dbParams_.user_ << "/" 
       << dbParams_.passwd_ << "@" 
       << dbParams_.path_ 
       << "\" using 'ConfDb' configurable read from .cfg file";
    LogTrace(mlConfigDb_) << ss.str();

  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unable to retrieve 'user/passwd@path' parameters"
      << " from 'CONFDB' environmental variable or .cfg file"
      << " (present value is \"" 
      << user << "/" 
      << passwd << "@" 
      << path 
      << "\"). Aborting connection to database...";
    return;
  }
  
  // Retrieve partition name from ENV_CMS_TK_PARTITION env. var. and override .cfg value
  std::string partition = "ENV_CMS_TK_PARTITION";
  if ( getenv(partition.c_str()) != NULL ) { 

    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"partitions\" to \""
       << getenv( partition.c_str() )
       << "\" using 'ENV_CMS_TK_PARTITION' environmental variable";
    if ( !dbParams_.partitions().empty() ) {
      ss << " (Overwriting existing value of \""
	 << dbParams_.partitions( dbParams_.partitions() )
	 << "\" read from .cfg file)";
    }
    LogTrace(mlConfigDb_) << ss.str() << std::endl;

    // Build partitions from env. var.
    std::vector<std::string> partitions = dbParams_.partitions( getenv( partition.c_str() ) );
    if ( !partitions.empty() ) {
      dbParams_.partitions_.clear();
      std::vector<std::string>::iterator ii = partitions.begin();
      std::vector<std::string>::iterator jj = partitions.end();
      for ( ; ii != jj; ++ii ) {
	dbParams_.partitions_[*ii] = SiStripPartition();
	dbParams_.partitions_[*ii].partitionName() = *ii;
      }
    }

  } else if ( !dbParams_.partitions().empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"partitions\" to \""
       << dbParams_.partitions( dbParams_.partitions() )
       << "\" using 'PartitionName' configurables read from .cfg file";
    LogTrace(mlConfigDb_) << ss.str();

  } else { 
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unable to retrieve 'partition' parameter"
      << " from 'CONFDB' environmental variable or .cfg file!"
      << " Aborting connection to database...";
    return;
  } 

  // Check TNS_ADMIN environmental variable
  std::string pattern = "TNS_ADMIN";
  std::string tns_admin = "/afs/cern.ch/project/oracle/admin";
  if ( getenv( pattern.c_str() ) != NULL ) { 
    tns_admin = getenv( pattern.c_str() ); 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " TNS_ADMIN is set to: \"" 
      << tns_admin << "\"";
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " TNS_ADMIN is not set!"
      << " Trying to use /afs and setting to: \"" 
      << tns_admin << "\"";
  }

  // Retrieve TNS_ADMIN from .cfg file and override
  if ( !dbParams_.tnsAdmin_.empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Overriding TNS_ADMIN value using cfg file!" << std::endl
       << "  Original value : \"" << tns_admin << "\"!" << std::endl
       << "  New value      : \"" << dbParams_.tnsAdmin_ << "\"!";
    tns_admin = dbParams_.tnsAdmin_;
    edm::LogVerbatim(mlConfigDb_) << ss.str();
  }
  
  // Remove trailing slash and set TNS_ADMIN
  if ( tns_admin.empty() ) { tns_admin = "."; }
  std::string slash = tns_admin.substr( tns_admin.size()-1, 1 ); 
  if ( slash == sistrip::dir_ ) { tns_admin = tns_admin.substr( 0, tns_admin.size()-1 ); }
  setenv( pattern.c_str(), tns_admin.c_str(), 1 ); 
  
  // Check if database is found in tnsnames.ora file
  std::string filename( tns_admin + "/tnsnames.ora" ); 
  std::ifstream tnsnames_ora( filename.c_str() );
  bool ok = false;
  if ( tnsnames_ora.is_open() ) {
    std::string line;
    while ( !tnsnames_ora.eof() ) {
      getline( tnsnames_ora, line );
      if ( !dbParams_.path_.empty() && 
	   line.find( dbParams_.path_ ) != std::string::npos ) { ok = true; }
    }
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Cannot open file \""
      << filename << "\"";
  }

  if ( ok ) {
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Found database account \"" 
      << dbParams_.path_ << "\" in file \""
      << filename << "\"!";
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Cannot find database account \"" 
      << dbParams_.path_ << "\" in file \""
      << filename << "\""
      << " Aborting connection to database...";
    return; 
  }
  
  // Create device factory object
  try { 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Creating DeviceFactory object...";
    factory_ = new DeviceFactory( dbParams_.user_, 
				  dbParams_.passwd_, 
				  dbParams_.path_ ); 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Created DeviceFactory object!";
  } catch (...) { 
    std::stringstream ss; 
    ss << "Failed to connect to database using parameters '" 
       << dbParams_.user_ << "/" 
       << dbParams_.passwd_ << "@" 
       << dbParams_.path_ 
       << "' and partitions '" 
       << dbParams_.partitions( dbParams_.partitions() ) << "'";
    handleException( __func__, ss.str() );
    return;
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__func__) ) { 
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DeviceFactory created at address 0x" 
       << std::hex << std::setw(8) << std::setfill('0') << factory_ << std::dec
       << ", using database account with parameters '" 
       << dbParams_.user_ << "/" 
       << dbParams_.passwd_ << "@" 
       << dbParams_.path_
       << "' and partitions '" 
       << dbParams_.partitions( dbParams_.partitions() ) << "'";
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database using connection parameters '" 
      << dbParams_.user_ << "/" 
      << dbParams_.passwd_ << "@" 
      << dbParams_.path_
      << "' and partitions '" 
      << dbParams_.partitions( dbParams_.partitions() ) << "'";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb_ ); 
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  // Check if should use current state, run number or versions
  SiStripDbParams::SiStripPartitions::iterator ip = dbParams_.partitions_.begin();
  SiStripDbParams::SiStripPartitions::iterator jp = dbParams_.partitions_.end();
  for ( ; ip != jp; ++ip ) { ip->second.update( this ); }
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabaseCache() {
  
  // Reset all DbParams except for those concerning database cache
  SiStripDbParams temp;
  temp = dbParams_;
  dbParams_.reset();
  dbParams_.usingDb_ = temp.usingDb_;
  dbParams_.usingDbCache_ = temp.usingDbCache_;
  dbParams_.sharedMemory_ = temp.sharedMemory_;
  
  // Check shared memory name from .cfg file
  if ( dbParams_.sharedMemory_.empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Empty string for shared memory name!" 
       << " Cannot accept shared memory!";
    edm::LogError(mlConfigDb_) << ss.str();
    return;
  }
  
  // Create database cache object
#ifdef USING_NEW_DATABASE_MODEL
  try { 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Creating DbClient object...";
    dbCache_ = new DbClient( dbParams_.sharedMemory_ );
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Created DbClient object...";
  } catch (...) { 
    std::stringstream ss; 
    ss << "Failed to connect to database cache using shared memory name: '" 
       << dbParams_.sharedMemory_ << "'!";
    handleException( __func__, ss.str() );
    return;
  }
#endif
  
  // Check for valid pointer to DbClient object
  if ( databaseCache(__func__) ) { 
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DbClient object created at address 0x" 
       << std::hex << std::setw(8) << std::setfill('0') << dbCache_ << std::dec
       << " using shared memory name '" 
       << dbParams_.sharedMemory_ << "'"; 
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DbClient object!"
      << " Unable to connect to database cache using shared memory name '" 
      << dbParams_.sharedMemory_ << "'"; 
    return; 
  }
  
  // Try retrieve descriptions from Database Client
#ifdef USING_NEW_DATABASE_MODEL
  try { 
    databaseCache(__func__)->parse(); 
  } catch (...) { 
    handleException( __func__, "Attempted to called DbClient::parse() method" );
  }
#endif
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingXmlFiles() {
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Using XML description files...";

  // Create device factory object
  try { 
    factory_ = new DeviceFactory(); 
  } catch (...) { 
    handleException( __func__, "Attempting to create DeviceFactory for use with xml files" );
  }
  
 // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__func__) ) { 
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DeviceFactory created at address 0x" 
       << std::hex << std::setw(8) << std::setfill('0') << factory_ << std::dec
       << ", using XML description files";
    LogTrace(mlConfigDb_) << ss.str();
  } else {    
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database!";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb_ );
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }

  try { 
#ifndef USING_NEW_DATABASE_MODEL
    deviceFactory(__func__)->createInputFileAccess();
#endif
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputFileAccess'" ); 
  }

  // Iterate through partitions
  SiStripDbParams::SiStripPartitions::iterator ip = dbParams_.partitions_.begin();
  SiStripDbParams::SiStripPartitions::iterator jp = dbParams_.partitions_.end();
  for ( ; ip != jp; ++ip ) {
    
    // Input module.xml file
    if ( ip->second.inputModuleXml() == "" ) {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL path to input 'module.xml' file!";
    } else {
      if ( checkFileExists( ip->second.inputModuleXml() ) ) { 
	try { 
#ifdef USING_NEW_DATABASE_MODEL
	  deviceFactory(__func__)->addConnectionFileName( ip->second.inputModuleXml() ); 
#else
	  deviceFactory(__func__)->addFedFecConnectionFileName( ip->second.inputModuleXml() ); 
#endif
	} catch (...) { 
	  handleException( __func__ ); 
	}
	LogTrace(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " Added input 'module.xml' file: " << ip->second.inputModuleXml();
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " No 'module.xml' file found at " << ip->second.inputModuleXml();
	ip->second.inputModuleXml() = ""; 
      }
    }
  
    // Input dcuinfo.xml file
    if ( ip->second.inputDcuInfoXml() == "" ) {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL path to input 'dcuinfo.xml' file!";
    } else { 
      if ( checkFileExists( ip->second.inputDcuInfoXml() ) ) { 
	try { 
	  deviceFactory(__func__)->addTkDcuInfoFileName( ip->second.inputDcuInfoXml() ); 
	} catch (...) { 
	  handleException( __func__ ); 
	}
	LogTrace(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " Added 'dcuinfo.xml' file: " << ip->second.inputDcuInfoXml();
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " No 'dcuinfo.xml' file found at " << ip->second.inputDcuInfoXml();
	ip->second.inputDcuInfoXml() = ""; 
      } 
    }

    // Input FEC xml files
    if ( ip->second.inputFecXml().empty() ) {
      edm::LogWarning(mlConfigDb_) 
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL paths to input 'fec.xml' files!";
    } else {
      std::vector<std::string>::iterator iter = ip->second.inputFecXml().begin();
      for ( ; iter != ip->second.inputFecXml().end(); iter++ ) {
	if ( *iter == "" ) {
	  edm::LogWarning(mlConfigDb_)
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " NULL path to input 'fec.xml' file!";
	} else {
	  if ( checkFileExists( *iter ) ) { 
	    try { 
	      deviceFactory(__func__)->addFecFileName( *iter ); 
	    } catch (...) { handleException( __func__ ); }
	    LogTrace(mlConfigDb_) 
	      << "[SiStripConfigDb::" << __func__ << "]"
	      << " Added 'fec.xml' file: " << *iter;
	  } else {
	    edm::LogWarning(mlConfigDb_) 
	      << "[SiStripConfigDb::" << __func__ << "]"
	      << " No 'fec.xml' file found at " << *iter;
	    *iter = ""; 
	  } 
	}
      }
    }
    
    // Input FED xml files
    if ( ip->second.inputFedXml().empty() ) {
      edm::LogWarning(mlConfigDb_) 
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL paths to input 'fed.xml' files!";
    } else {
      std::vector<std::string>::iterator iter = ip->second.inputFedXml().begin();
      for ( ; iter != ip->second.inputFedXml().end(); iter++ ) {
	if ( *iter == "" ) {
	  edm::LogWarning(mlConfigDb_) 
	    << "[SiStripConfigDb::" << __func__ << "]"
	    << " NULL path to input 'fed.xml' file!";
	} else {
	  if ( checkFileExists( *iter ) ) { 
	    try { 
		deviceFactory(__func__)->addFedFileName( *iter ); 
	    } catch (...) { 
	      handleException( __func__ ); 
	    }
	    LogTrace(mlConfigDb_) 
	      << "[SiStripConfigDb::" << __func__ << "]"
	      << " Added 'fed.xml' file: " << *iter;
	  } else {
	    edm::LogWarning(mlConfigDb_) 
	      << "[SiStripConfigDb::" << __func__ << "]"
	      << " No 'fed.xml' file found at " << *iter;
	    *iter = ""; 
	  } 
	}
      }
    }

  }

  // Output module.xml file
  if ( dbParams_.outputModuleXml_ == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'module.xml' file!"
      << " Setting to '/tmp/module.xml'...";
    dbParams_.outputModuleXml_ = "/tmp/module.xml"; 
  } else {
    try { 
#ifdef USING_NEW_DATABASE_MODEL
      ConnectionFactory* factory = deviceFactory(__func__);
#else
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
#endif
      factory->setOutputFileName( dbParams_.outputModuleXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'module.xml' file!" ); 
    }
  }

  // Output dcuinfo.xml file
  if ( dbParams_.outputDcuInfoXml_ == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'dcuinfo.xml' file!"
      << " Setting to '/tmp/dcuinfo.xml'...";
    dbParams_.outputModuleXml_ = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputDcuInfoXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'dcuinfo.xml' file!" ); 
    }
  }

  // Output fec.xml file
  if ( dbParams_.outputFecXml_ == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fec.xml' file!"
      << " Setting to '/tmp/fec.xml'...";
    dbParams_.outputFecXml_ = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputFecXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'fec.xml' file!" ); 
    }
  }

  // Output fed.xml file
  if ( dbParams_.outputFedXml_ == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fed.xml' file!"
      << " Setting to '/tmp/fed.xml'...";
    dbParams_.outputFedXml_ = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputFedXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'fed.xml' file!" ); 
    }
  }

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const std::string& method_name,
				       const std::string& extra_info ) { //throw (cms::Exception) {

  std::stringstream ss;
  try {
    throw; // rethrow caught exception to be dealt with below
  } 

  catch ( const cms::Exception& e ) { 
    ss << " Caught cms::Exception in method "
       << method_name << " with message: " << std::endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw e; // rethrow cms::Exception
  }
  
  catch ( const oracle::occi::SQLException& e ) { 
    ss << " Caught oracle::occi::SQLException in method "
       << method_name << " with message: " << std::endl 
       << e.getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }

  catch ( const FecExceptionHandler& e ) {
    ss << " Caught FecExceptionHandler exception in method "
       << method_name << " with message: " << std::endl 
       << const_cast<FecExceptionHandler&>(e).getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }

  catch ( const ICUtils::ICException& e ) {
    ss << " Caught ICUtils::ICException in method "
       << method_name << " with message: " << std::endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }

  catch ( const exception& e ) {
    ss << " Caught std::exception in method "
       << method_name << " with message: " << std::endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }

  catch (...) {
    ss << " Caught unknown exception in method "
       << method_name << " (No message) " << std::endl;
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }
  
  // Message
  edm::LogError(mlConfigDb_) << ss.str();
  
}

// -----------------------------------------------------------------------------
//
bool SiStripConfigDb::checkFileExists( const std::string& path ) {
  fstream fs; 
  fs.open( path.c_str(), ios::in );
  if( !fs.is_open() ) { return false; }
  fs.close();
  return true;
}
