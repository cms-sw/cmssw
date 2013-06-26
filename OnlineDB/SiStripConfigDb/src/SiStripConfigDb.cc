// Last commit: $Id: SiStripConfigDb.cc,v 1.76 2009/04/06 16:57:28 lowette Exp $

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
  dcuDetIds_(), 
  analyses_(),
  apvDevices_(),
  muxDevices_(),
  dcuDevices_(),
  lldDevices_(),
  pllDevices_(),
  dohDevices_(),
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
  if ( dbParams_.usingDb() ) { 
    if ( dbParams_.usingDbCache() ) { usingDatabaseCache(); }
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

  try { 
    if ( dbCache_ ) { delete dbCache_; }
  } catch (...) { handleException( __func__, "Attempting to delete DbClient object..." ); }
  dbCache_ = 0; 
  
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
  clearDcuDetIds();
  clearAnalysisDescriptions();

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
    if ( dbParams_.user() != null_ || 
	 dbParams_.passwd() != null_ || 
	 dbParams_.path() != null_ ) { 
      ss << " (Overwriting existing value of \""
	 << dbParams_.user() << "/" 
	 << dbParams_.passwd() << "@" 
	 << dbParams_.path() 
	 << "\" read from .cfg file)";
    }
    edm::LogVerbatim(mlConfigDb_) << ss.str() << std::endl;
    dbParams_.confdb( user, passwd, path );

  } else if ( dbParams_.user() != null_ && 
	      dbParams_.passwd() != null_ && 
	      dbParams_.path() != null_ ) { 

    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"user/passwd@path\" to \""
       << dbParams_.user() << "/" 
       << dbParams_.passwd() << "@" 
       << dbParams_.path() 
       << "\" using 'ConfDb' configurable read from .cfg file";
    edm::LogVerbatim(mlConfigDb_) << ss.str();

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
  
  // Check TNS_ADMIN environmental variable
  std::string pattern = "TNS_ADMIN";
  std::string tns_admin = "/afs/cern.ch/project/oracle/admin";
  if ( getenv( pattern.c_str() ) != NULL ) { 
    tns_admin = getenv( pattern.c_str() ); 
    edm::LogVerbatim(mlConfigDb_)
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
  if ( !dbParams_.tnsAdmin().empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Overriding TNS_ADMIN value using cfg file!" << std::endl
       << "  Original value : \"" << tns_admin << "\"!" << std::endl
       << "  New value      : \"" << dbParams_.tnsAdmin() << "\"!";
    tns_admin = dbParams_.tnsAdmin();
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
      if ( !dbParams_.path().empty() && 
	   line.find( dbParams_.path() ) != std::string::npos ) { ok = true; }
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
      << dbParams_.path() << "\" in file \""
      << filename << "\"!";
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Cannot find database account \"" 
      << dbParams_.path() << "\" in file \""
      << filename << "\""
      << " Aborting connection to database...";
    return; 
  }
  
  // Create device factory object
  try { 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Creating DeviceFactory object...";
    factory_ = new DeviceFactory( dbParams_.user(), 
				  dbParams_.passwd(), 
				  dbParams_.path() ); 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Created DeviceFactory object!";
  } catch (...) { 
    std::stringstream ss; 
    ss << "Failed to connect to database using parameters '" 
       << dbParams_.user() << "/" 
       << dbParams_.passwd() << "@" 
       << dbParams_.path() 
       << "' and partitions '" 
       << dbParams_.partitionNames( dbParams_.partitionNames() ) << "'";
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
       << dbParams_.user() << "/" 
       << dbParams_.passwd() << "@" 
       << dbParams_.path();
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database using connection parameters '" 
      << dbParams_.user() << "/" 
      << dbParams_.passwd() << "@" 
      << dbParams_.path()
      << "' and partitions '" 
      << dbParams_.partitionNames( dbParams_.partitionNames() ) << "'";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb() ); 
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  // Retrieve partition name from ENV_CMS_TK_PARTITION env. var. and override .cfg value
  std::string partition = "ENV_CMS_TK_PARTITION";
  if ( getenv(partition.c_str()) != NULL ) { 
    
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"partitions\" to \""
       << getenv( partition.c_str() )
       << "\" using 'ENV_CMS_TK_PARTITION' environmental variable";
    if ( !dbParams_.partitionNames().empty() ) {
      ss << " (Overwriting existing value of \""
	 << dbParams_.partitionNames( dbParams_.partitionNames() )
	 << "\" read from .cfg file)";
    }
    edm::LogVerbatim(mlConfigDb_) << ss.str() << std::endl;
    
    // Build partitions from env. var.
    std::vector<std::string> partitions = dbParams_.partitionNames( getenv( partition.c_str() ) );
    if ( !partitions.empty() ) {
      dbParams_.clearPartitions();
      std::vector<std::string>::iterator ii = partitions.begin();
      std::vector<std::string>::iterator jj = partitions.end();
      for ( ; ii != jj; ++ii ) {
	SiStripPartition partition( *ii );
	dbParams_.addPartition( partition );
      }
    }

  } else if ( !dbParams_.partitionNames().empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"partitions\" to \""
       << dbParams_.partitionNames( dbParams_.partitionNames() )
       << "\" using 'PartitionName' configurables read from .cfg file";
    edm::LogVerbatim(mlConfigDb_) << ss.str();
  } else { 
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unable to retrieve 'partition' parameter"
      << " from 'CONFDB' environmental variable or .cfg file!"
      << " Aborting connection to database...";
    return;
  } 

  // Check if should use current state, run number or versions
  SiStripDbParams::SiStripPartitions::iterator ip = dbParams_.partitions().begin();
  SiStripDbParams::SiStripPartitions::iterator jp = dbParams_.partitions().end();
  for ( ; ip != jp; ++ip ) { ip->second.update( this ); }
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabaseCache() {
  
  // Reset all DbParams except for those concerning database cache
  SiStripDbParams temp;
  temp = dbParams_;
  dbParams_.reset();
  dbParams_.usingDb( temp.usingDb() );
  dbParams_.usingDbCache( temp.usingDbCache() );
  dbParams_.sharedMemory( temp.sharedMemory() );

  // Add default partition 
  dbParams_.addPartition( SiStripPartition( SiStripPartition::defaultPartitionName_ ) );
  
  // Check shared memory name from .cfg file
  if ( dbParams_.sharedMemory().empty() ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Empty string for shared memory name!" 
       << " Cannot accept shared memory!";
    edm::LogError(mlConfigDb_) << ss.str();
    return;
  }
  
  // Create database cache object
  try { 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Creating DbClient object...";
    dbCache_ = new DbClient( dbParams_.sharedMemory() );
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Created DbClient object...";
  } catch (...) { 
    std::stringstream ss; 
    ss << "Failed to connect to database cache using shared memory name: '" 
       << dbParams_.sharedMemory() << "'!";
    handleException( __func__, ss.str() );
    return;
  }
  
  // Check for valid pointer to DbClient object
  if ( databaseCache(__func__) ) { 
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DbClient object created at address 0x" 
       << std::hex << std::setw(8) << std::setfill('0') << dbCache_ << std::dec
       << " using shared memory name '" 
       << dbParams_.sharedMemory() << "'"; 
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DbClient object!"
      << " Unable to connect to database cache using shared memory name '" 
      << dbParams_.sharedMemory() << "'"; 
    return; 
  }
  
  // Try retrieve descriptions from Database Client
  try { 
    databaseCache(__func__)->parse(); 
  } catch (...) { 
    handleException( __func__, "Attempted to called DbClient::parse() method" );
  }
  
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
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb() );
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }

  // Iterate through partitions
  SiStripDbParams::SiStripPartitions::const_iterator ip = dbParams_.partitions().begin();
  SiStripDbParams::SiStripPartitions::const_iterator jp = dbParams_.partitions().end();
  for ( ; ip != jp; ++ip ) {
    
    // Input module.xml file
    if ( ip->second.inputModuleXml() == "" ) {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL path to input 'module.xml' file!";
    } else {
      if ( checkFileExists( ip->second.inputModuleXml() ) ) { 
	try { 
	  deviceFactory(__func__)->addConnectionFileName( ip->second.inputModuleXml() ); 
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
  if ( dbParams_.outputModuleXml() == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'module.xml' file!"
      << " Setting to '/tmp/module.xml'...";
    dbParams_.outputModuleXml() = "/tmp/module.xml"; 
  } else {
    try { 
      ConnectionFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputModuleXml() ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'module.xml' file!" ); 
    }
  }

  // Output dcuinfo.xml file
  if ( dbParams_.outputDcuInfoXml() == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'dcuinfo.xml' file!"
      << " Setting to '/tmp/dcuinfo.xml'...";
    dbParams_.outputModuleXml() = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputDcuInfoXml() ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'dcuinfo.xml' file!" ); 
    }
  }

  // Output fec.xml file
  if ( dbParams_.outputFecXml() == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fec.xml' file!"
      << " Setting to '/tmp/fec.xml'...";
    dbParams_.outputFecXml() = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputFecXml() ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'fec.xml' file!" ); 
    }
  }

  // Output fed.xml file
  if ( dbParams_.outputFedXml() == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fed.xml' file!"
      << " Setting to '/tmp/fed.xml'...";
    dbParams_.outputFedXml() = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( dbParams_.outputFedXml() ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'fed.xml' file!" ); 
    }
  }

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const std::string& method_name,
				       const std::string& extra_info ) const { 

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
       << const_cast<FecExceptionHandler&>(e).what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
  }

//   catch ( const Fed9UDeviceFactoryException& e ) {
//     ss << " Caught Fed9UDeviceFactoryException exception in method "
//        << method_name << " with message: " << std::endl 
//        << e.what();
//     if ( extra_info != "" ) { ss << "Additional info: " << extra_info << std::endl; }
//     //throw cms::Exception(mlConfigDb_) << ss.str() << std::endl;
//   }

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

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::runs( SiStripConfigDb::Runs& runs ) const {
  
  runs.clear();
  
  // Check DF pointer
  DeviceFactory* const df = deviceFactory(__func__);
  if ( !df ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripPartition::" << __func__ << "]"
      << " NULL pointer to DeviceFactory object!";
    return;
  }
  
  // Retrieve runs
  tkRunVector all;
  all = df->getAllRuns();

  // Iterate through tkRunVector
  tkRunVector::const_iterator ii = all.begin();
  tkRunVector::const_iterator jj = all.end();
  for ( ; ii != jj; ++ii ) {

    // Check TkRun pointer
    if ( *ii ) {

      // Retrieve run type
      uint16_t type = (*ii)->getModeId( (*ii)->getMode() );
      sistrip::RunType temp = sistrip::UNKNOWN_RUN_TYPE;
      if      ( type ==  1 ) { temp = sistrip::PHYSICS; }
      else if ( type ==  2 ) { temp = sistrip::PEDESTALS; }
      else if ( type ==  3 ) { temp = sistrip::CALIBRATION; }
      else if ( type == 33 ) { temp = sistrip::CALIBRATION_DECO; }
      else if ( type ==  4 ) { temp = sistrip::OPTO_SCAN; }
      else if ( type ==  5 ) { temp = sistrip::APV_TIMING; }
      else if ( type ==  6 ) { temp = sistrip::APV_LATENCY; }
      else if ( type ==  7 ) { temp = sistrip::FINE_DELAY_PLL; }
      else if ( type == 10 ) { temp = sistrip::MULTI_MODE; }
      else if ( type ==  8 ) { temp = sistrip::FINE_DELAY_TTC; }
      else if ( type == 12 ) { temp = sistrip::FED_TIMING; }
      else if ( type == 13 ) { temp = sistrip::FED_CABLING; }
      else if ( type == 14 ) { temp = sistrip::VPSP_SCAN; }
      else if ( type == 15 ) { temp = sistrip::DAQ_SCOPE_MODE; }
      else if ( type == 16 ) { temp = sistrip::QUITE_FAST_CABLING; }
      else if ( type == 17 ) { temp = sistrip::FINE_DELAY; }
      else if ( type == 18 ) { temp = sistrip::PHYSICS_ZS; }
      else if ( type == 19 ) { temp = sistrip::CALIBRATION_SCAN; }
      else if ( type == 20 ) { temp = sistrip::CALIBRATION_SCAN_DECO; }
      else if ( type == 21 ) { temp = sistrip::FAST_CABLING; }
      else if ( type ==  0 ) { temp = sistrip::UNDEFINED_RUN_TYPE; }
      else                   { temp = sistrip::UNKNOWN_RUN_TYPE; }
      
      // Store run details
      Run r;
      r.type_      = temp;
      r.partition_ = (*ii)->getPartitionName();
      r.number_    = (*ii)->getRunNumber();
      runs.push_back(r);

    } else {
      edm::LogWarning(mlConfigDb_)
 	<< "[SiStripPartition::" << __func__ << "]"
 	<< " NULL pointer to TkRun object!";
    }

  }

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::runs( const SiStripConfigDb::Runs& in,
			    SiStripConfigDb::RunsByType& out,
			    std::string optional_partition ) const {
  
  out.clear();
  
  // Check partition name (if not empty string)
  if ( !optional_partition.empty() ) {
    SiStripDbParams::SiStripPartitions::const_iterator iter = dbParams_.partition( optional_partition );
    if ( iter == dbParams_.partitions().end() ) { 
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripPartition::" << __func__ << "]"
	<< " Partition name not found!";
      return; 
    }
  }

  // Iterate through runs
  Runs::const_iterator ii = in.begin();
  Runs::const_iterator jj = in.end();
  for ( ; ii != jj; ++ii ) {
    // Check partition name
    if ( ii->partition_ == optional_partition || optional_partition == "" ) { 
      // Check run type
      if ( ii->type_ != sistrip::UNKNOWN_RUN_TYPE &&
	   ii->type_ != sistrip::UNDEFINED_RUN_TYPE ) { 
	// Check run number
	if ( ii->number_ ) { 
	  bool found = false;
	  if ( out.find( ii->type_ ) != out.end() ) {
	    Runs::const_iterator irun = out[ ii->type_ ].begin();
	    Runs::const_iterator jrun = out[ ii->type_ ].end();
	    while ( !found && irun != jrun ) {
	      if ( irun->number_ == ii->number_ ) { found = true; }
	      ++irun;
	    }
	  }
	  // Check if run number already found
	  if ( !found ) { 
	    out[ ii->type_ ].push_back( *ii ); 
	  } else {
	    // 	      edm::LogWarning(mlConfigDb_)
	    // 		<< "[SiStripPartition::" << __func__ << "]"
	    // 		<< " Run number already found!";
	  }
	} else {
	  // 	    edm::LogWarning(mlConfigDb_)
	  // 	      << "[SiStripPartition::" << __func__ << "]"
	  // 	      << " NULL run number!";
	}
      } else {
	// 	  edm::LogWarning(mlConfigDb_)
	// 	    << "[SiStripPartition::" << __func__ << "]"
	// 	    << " Unexpected run type!";
      }
    } else {
      // 	edm::LogWarning(mlConfigDb_)
      // 	  << "[SiStripPartition::" << __func__ << "]"
      // 	  << " Partition name does not match!";
    }

  }

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::runs( const SiStripConfigDb::Runs& in,
			    SiStripConfigDb::RunsByPartition& out,
			    sistrip::RunType optional_type ) const {
  
  out.clear();

  // Iterate through runs
  Runs::const_iterator ii = in.begin();
  Runs::const_iterator jj = in.end();
  for ( ; ii != jj; ++ii ) {
    // Check partition name
    if ( ii->partition_ != "" ) {
      // Check run type
      if ( ii->type_ == optional_type || optional_type == sistrip::UNDEFINED_RUN_TYPE ) { 
	// Check run number
	if ( ii->number_ ) { 
	  bool found = false;
	  if ( out.find( ii->partition_ ) != out.end() ) {
	    Runs::const_iterator irun = out[ ii->partition_ ].begin();
	    Runs::const_iterator jrun = out[ ii->partition_ ].end();
	    while ( !found && irun != jrun ) {
	      if ( irun->number_ == ii->number_ ) { found = true; }
	      ++irun;
	    }
	  }
	  // Check if run number already found
	  if ( !found ) { 
	    out[ ii->partition_ ].push_back( *ii ); 
	  } else {
	    // 	      edm::LogWarning(mlConfigDb_)
	    // 		<< "[SiStripPartition::" << __func__ << "]"
	    // 		<< " Run number already found!";
	  }
	} else {
	  // 	    edm::LogWarning(mlConfigDb_)
	  // 	      << "[SiStripPartition::" << __func__ << "]"
	  // 	      << " NULL run number!";
	}
      } else {
	// 	  edm::LogWarning(mlConfigDb_)
	// 	    << "[SiStripPartition::" << __func__ << "]"
	// 	    << " Run type does not match!";
      }
    } else {
      // 	edm::LogWarning(mlConfigDb_)
      // 	  << "[SiStripPartition::" << __func__ << "]"
      // 	  << " NULL value for partition!";
    }

  }

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::partitions( std::list<std::string>& partitions ) const {

  partitions.clear();

  // Check DF pointer
  DeviceFactory* const df = deviceFactory(__func__);
  if ( !df ) {
    edm::LogError(mlConfigDb_)
      << "[SiStripPartition::" << __func__ << "]"
      << " NULL pointer to DeviceFactory object!";
    return;
  }

  partitions = df->getAllPartitionNames();
  
}

  
