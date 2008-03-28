// Last commit: $Id: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
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
  devices_(), 
  feds_(), 
  connections_(), 
  dcuDetIdMap_(), 
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
  dbParams_.setParams( pset );
  
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
SiStripConfigDb::DbParams::DbParams() :
  usingDb_(false),
  user_(null_),
  passwd_(null_),
  path_(null_),
  partitions_(), 
  usingDbCache_(false),
  sharedMemory_(""),
  runNumber_(0),
  cabMajor_(0),
  cabMinor_(0),
  fedMajor_(0),
  fedMinor_(0),
  fecMajor_(0),
  fecMinor_(0),
  calMajor_(0),
  calMinor_(0),
  dcuMajor_(0),
  dcuMinor_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE),
  force_(true),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_(),
  inputDcuConvXml_(""),
  outputModuleXml_("/tmp/module.xml"),
  outputDcuInfoXml_("/tmp/dcuinfo.xml"),
  outputFecXml_("/tmp/fec.xml"),
  outputFedXml_("/tmp/fed.xml"),
  tnsAdmin_("")
{
  reset();
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DbParams::~DbParams() {
  inputFecXml_.clear();
  inputFedXml_.clear();
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::reset() {
  usingDb_ = false;
  confdb_ = null_;
  confdb( confdb_ );
  partitions_ = std::vector<std::string>();
  usingDbCache_ = false;
  sharedMemory_ = "";
  runNumber_ = 0;
  cabMajor_ = 0;
  cabMinor_ = 0;
  fedMajor_ = 0;
  fedMinor_ = 0;
  fecMajor_ = 0;
  fecMinor_ = 0;
  calMajor_ = 0;
  calMinor_ = 0;
  dcuMajor_ = 0;
  dcuMinor_ = 0;
  runType_  = sistrip::UNDEFINED_RUN_TYPE;
  force_    = true;
  inputModuleXml_   = "";
  inputDcuInfoXml_  = "";
  inputFecXml_      = std::vector<std::string>(1,"");
  inputFedXml_      = std::vector<std::string>(1,"");
  inputDcuConvXml_  = "";
  outputModuleXml_  = "";
  outputDcuInfoXml_ = "";
  outputFecXml_     = "";
  outputFedXml_     = "";
  tnsAdmin_         = "";
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::setParams( const edm::ParameterSet& pset ) {
  reset();
  usingDb_ = pset.getUntrackedParameter<bool>( "UsingDb", false ); 
  confdb( pset.getUntrackedParameter<std::string>( "ConfDb", "") );
  partitions_ = pset.getUntrackedParameter< std::vector<std::string> >( "Partitions", std::vector<std::string>() );
  runNumber_ = pset.getUntrackedParameter<unsigned int>( "RunNumber", 0 );
  usingDbCache_ = pset.getUntrackedParameter<bool>( "UsingDbCache", false ); 
  sharedMemory_ = pset.getUntrackedParameter<std::string>( "SharedMemory", "" ); 
  cabMajor_ = pset.getUntrackedParameter<unsigned int>( "MajorVersion", 0 );
  cabMinor_ = pset.getUntrackedParameter<unsigned int>( "MinorVersion", 0 );
  fedMajor_ = pset.getUntrackedParameter<unsigned int>( "FedMajorVersion", 0 );
  fedMinor_ = pset.getUntrackedParameter<unsigned int>( "FedMinorVersion", 0 );
  fecMajor_ = pset.getUntrackedParameter<unsigned int>( "FecMajorVersion", 0 );
  fecMinor_ = pset.getUntrackedParameter<unsigned int>( "FecMinorVersion", 0 );
  dcuMajor_ = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMajorVersion", 0 );
  dcuMinor_ = pset.getUntrackedParameter<unsigned int>( "DcuDetIdMinorVersion", 0 );
  calMajor_ = pset.getUntrackedParameter<unsigned int>( "CalibMajorVersion", 0 );
  calMinor_ = pset.getUntrackedParameter<unsigned int>( "CalibMinorVersion", 0 );
  force_ = pset.getUntrackedParameter<bool>( "ForceDcuDetIdVersions", true );
  inputModuleXml_ = pset.getUntrackedParameter<std::string>( "InputModuleXml", "" );
  inputDcuInfoXml_ = pset.getUntrackedParameter<std::string>( "InputDcuInfoXml", "" ); 
  inputFecXml_ = pset.getUntrackedParameter< std::vector<std::string> >( "InputFecXml", std::vector<std::string>(1,"") ); 
  inputFedXml_ = pset.getUntrackedParameter< std::vector<std::string> >( "InputFedXml", std::vector<std::string>(1,"") ); 
  inputDcuConvXml_ = pset.getUntrackedParameter<std::string>( "InputDcuConvXml", "" );
  outputModuleXml_ = pset.getUntrackedParameter<std::string>( "OutputModuleXml", "/tmp/module.xml" );
  outputDcuInfoXml_ = pset.getUntrackedParameter<std::string>( "OutputDcuInfoXml", "/tmp/dcuinfo.xml" );
  outputFecXml_ = pset.getUntrackedParameter<std::string>( "OutputFecXml", "/tmp/fec.xml" );
  outputFedXml_ = pset.getUntrackedParameter<std::string>( "OutputFedXml", "/tmp/fed.xml" );
  tnsAdmin_ = pset.getUntrackedParameter<std::string>( "TNS_ADMIN", "" );
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::confdb( const std::string& confdb ) {
  confdb_ = confdb;
  uint32_t ipass = confdb.find("/");
  uint32_t ipath = confdb.find("@");
  if ( ipass != std::string::npos && 
       ipath != std::string::npos ) {
    user_   = confdb.substr(0,ipass); 
    passwd_ = confdb.substr(ipass+1,ipath-ipass-1); 
    path_   = confdb.substr(ipath+1,confdb.size());
  } else {
    user_   = null_;
    passwd_ = null_;
    path_   = null_;
  }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::confdb( const std::string& user,
					const std::string& passwd,
					const std::string& path ) {
  if ( user != "" && passwd != "" && path != "" &&
       user != null_ && passwd != null_ && path != null_ ) {
    user_   = user;
    passwd_ = passwd;
    path_   = path;
  } else {
    user_   = null_;
    passwd_ = null_;
    path_   = null_;
  }
  confdb_ = user_ + "/" + passwd_ + "@" + path_;
}

// -----------------------------------------------------------------------------
// 
std::string SiStripConfigDb::DbParams::partitions() const {
  std::stringstream ss;
  std::vector<std::string>::const_iterator ii = partitions_.begin();
  std::vector<std::string>::const_iterator jj = partitions_.end();
  for ( ; ii != jj; ++ii ) { ii == partitions_.begin() ? ss << *ii : ss << ", " << *ii; }
  return ss.str();
}

// -----------------------------------------------------------------------------
// 
std::vector<std::string> SiStripConfigDb::DbParams::partitions( std::string input ) const {
  std::istringstream ss(input);
  std::vector<std::string> partitions;
  std::string delimiter = ":";
  std::string token;
  while ( getline( ss, token, ':' ) ) { partitions.push_back(token); }
  return partitions;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::print( std::stringstream& ss ) const {

  ss << " Using database account    : " << std::boolalpha << usingDb_ << std::noboolalpha << std::endl;
  ss << " Using database cache      : " << std::boolalpha << usingDbCache_ << std::noboolalpha << std::endl;
  ss << " Shared memory name        : " << std::boolalpha << sharedMemory_ << std::noboolalpha << std::endl;

  if ( usingDb_ ) {

    ss << " ConfDb                    : " << confdb_ << std::endl;
      //<< " User, Passwd, Path        : " << user_ << ", " << passwd_ << ", " << path_ << std::endl;

  } else {

    // Input
    ss << " Input \"module.xml\" file   : " << inputModuleXml_ << std::endl
       << " Input \"dcuinfo.xml\" file  : " << inputDcuInfoXml_ << std::endl
       << " Input \"fec.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifec = inputFecXml_.begin();
    for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << *ifec << ", "; }
    ss << std::endl;
    ss << " Input \"fed.xml\" file(s)   : ";
    std::vector<std::string>::const_iterator ifed = inputFedXml_.begin();
    for ( ; ifed != inputFedXml_.end(); ifed++ ) { ss << *ifed << ", "; }
    ss << std::endl;

    // Output 
    ss << " Output \"module.xml\" file  : " << outputModuleXml_ << std::endl
       << " Output \"dcuinfo.xml\" file : " << outputDcuInfoXml_ << std::endl
       << " Output \"fec.xml\" file(s)  : " << outputFecXml_ << std::endl
       << " Output \"fed.xml\" file(s)  : " << outputFedXml_ << std::endl;

  }
  
  ss << " Partitions                : " << partitions() << std::endl;
  ss << " Run number                : " << runNumber_ << std::endl
     << " Run type                  : " << SiStripEnumsAndStrings::runType( runType_ ) << std::endl
     << " Cabling major/minor vers  : " << cabMajor_ << "." << cabMinor_ << std::endl
     << " FED major/minor vers      : " << fedMajor_ << "." << fedMinor_ << std::endl
     << " FEC major/minor vers      : " << fecMajor_ << "." << fecMinor_ << std::endl
     << " Calibration maj/min vers  : " << calMajor_ << "." << calMinor_ << std::endl
     << " DCU-DetId maj/min vers    : " << dcuMajor_ << "." << dcuMinor_;
  if ( force_ ) { ss << " (version not overridden by run number)"; }
  ss << std::endl;
  
}

// -----------------------------------------------------------------------------
// 
ostream& operator<< ( ostream& os, const SiStripConfigDb::DbParams& params ) {
  std::stringstream ss;
  params.print(ss);
  os << ss.str();
  return os;
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
  
  devices_.clear();
  feds_.clear();
  connections_.clear();
  dcuDetIdMap_.clear();
  fedIds_.clear();

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
  
  try { 
    if ( factory_ ) { delete factory_; }
  } catch (...) { handleException( __func__, "Attempting to delete DeviceFactory object..." ); }
  factory_ = 0; 

#ifdef USING_DATABASE_CACHE
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
    if ( !dbParams_.partitions_.empty() && 
	 dbParams_.partitions_.front() != "" ) {
      ss << " (Overwriting existing value of \""
	 << dbParams_.partitions()
	 << "\" read from .cfg file)";
    }
    LogTrace(mlConfigDb_) << ss.str() << std::endl;
    dbParams_.partitions_ = dbParams_.partitions( getenv( partition.c_str() ) );
  } else if ( !dbParams_.partitions_.empty() && 
	      dbParams_.partitions_.front() != "" ) {
    std::stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"partitions\" to \""
       << dbParams_.partitions()
       << "\" using 'Partitions' configurable read from .cfg file";
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
       << dbParams_.partitions() << "'";
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
       << dbParams_.partitions() << "'";
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
      << dbParams_.partitions() << "'";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb_ ); 
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  // Retrieve versioning from DB for a non-zero run numbers
  if ( dbParams_.runNumber_ ) {
    TkRun* run = deviceFactory(__func__)->getRun( dbParams_.partitions_.front(), //@@ only using first here! 
						  dbParams_.runNumber_ );
    if ( run ) {

      if ( run->getRunNumber() ) {

#ifdef USING_NEW_DATABASE_MODEL
	dbParams_.cabMajor_ = run->getConnectionVersionMajorId();
	dbParams_.cabMinor_ = run->getConnectionVersionMinorId();
#endif

	dbParams_.fecMajor_ = run->getFecVersionMajorId();
	dbParams_.fecMinor_ = run->getFecVersionMinorId();
	dbParams_.fedMajor_ = run->getFedVersionMajorId();
	dbParams_.fedMinor_ = run->getFedVersionMinorId();

#ifdef USING_NEW_DATABASE_MODEL
	if ( !dbParams_.force_ ) { //@@ check if forcing versions specified in .cfi
	  dbParams_.dcuMajor_ = run->getDcuInfoVersionMajorId();
	  dbParams_.dcuMinor_ = run->getDcuInfoVersionMinorId();
	}
#endif

#ifdef USING_NEW_DATABASE_MODEL
	//@@ dbParams_.psuMajor_ = run->getDcuPsuMapVersionMajorId();
	//@@ dbParams_.psuMinor_ = run->getDcuPsuMapVersionMinorId();
#endif

#ifdef USING_NEW_DATABASE_MODEL
	//@@ dbParams_.calMajor_ = run->getAnalysisVersionMajorId();
	//@@ dbParams_.calMinor_ = run->getAnalysisVersionMinorId();
#endif

	std::stringstream ss;
	LogTrace(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " Description versions overridden"
	  << " using values retrieved for run number "
	  << run->getRunNumber();
	
      } else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL run number returned!";
      }
      
      uint16_t type = run->getModeId( run->getMode() );
      if      ( type ==  1 ) { dbParams_.runType_ = sistrip::PHYSICS; }
      else if ( type ==  2 ) { dbParams_.runType_ = sistrip::PEDESTALS; }
      else if ( type ==  3 ) { dbParams_.runType_ = sistrip::CALIBRATION; }
      else if ( type == 33 ) { dbParams_.runType_ = sistrip::CALIBRATION_DECO; }
      else if ( type ==  4 ) { dbParams_.runType_ = sistrip::OPTO_SCAN; }
      else if ( type ==  5 ) { dbParams_.runType_ = sistrip::APV_TIMING; }
      else if ( type ==  6 ) { dbParams_.runType_ = sistrip::APV_LATENCY; }
      else if ( type ==  7 ) { dbParams_.runType_ = sistrip::FINE_DELAY_PLL; }
      else if ( type ==  8 ) { dbParams_.runType_ = sistrip::FINE_DELAY_TTC; }
      else if ( type == 10 ) { dbParams_.runType_ = sistrip::MULTI_MODE; }
      else if ( type == 12 ) { dbParams_.runType_ = sistrip::FED_TIMING; }
      else if ( type == 13 ) { dbParams_.runType_ = sistrip::FED_CABLING; }
      else if ( type == 14 ) { dbParams_.runType_ = sistrip::VPSP_SCAN; }
      else if ( type == 15 ) { dbParams_.runType_ = sistrip::DAQ_SCOPE_MODE; }
      else if ( type == 16 ) { dbParams_.runType_ = sistrip::QUITE_FAST_CABLING; }
      else if ( type == 21 ) { dbParams_.runType_ = sistrip::FAST_CABLING; }
      else if ( type ==  0 ) { 
	dbParams_.runType_ = sistrip::UNDEFINED_RUN_TYPE;
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL run type returned!";
      } else { dbParams_.runType_ = sistrip::UNKNOWN_RUN_TYPE; }
      
    } else {
      edm::LogError(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL pointer to TkRun object!"
	<< " Unable to retrieve versions for run number "
	<< dbParams_.runNumber_
	<< ". Run number may not be consistent with partition \"" 
	<< dbParams_.partitions_.front() << "\"!"; //@@ only using first here!!!
    }
  }
  
  std::stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters: "
     << std::endl << dbParams_;
  edm::LogVerbatim(mlConfigDb_) << ss.str();
  
  // DCU-DetId 
  try { 
#ifdef USING_NEW_DATABASE_MODEL
    deviceFactory(__func__)->addDetIdPartition( dbParams_.partitions_.front(),
						dbParams_.dcuMajor_, 
						dbParams_.dcuMinor_ );
#else
    deviceFactory(__func__)->addDetIdPartition( dbParams_.partitions_.front() );
#endif
  } catch (...) { 
    std::stringstream ss;
    ss << "Attempted to 'addDetIdPartition' for partition: " << dbParams_.partitions_.front();
    handleException( __func__, ss.str() );
  }

#ifndef USING_NEW_DATABASE_MODEL
  // FED-FEC connections
  try { 
    deviceFactory(__func__)->createInputDBAccess();
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
  }
  
  try {
    deviceFactory(__func__)->setInputDBVersion( dbParams_.partitions_.front(),
						dbParams_.cabMajor_,
						dbParams_.cabMinor_ );
  } catch (...) { 
    std::stringstream ss;
    ss << "Attempted to 'setInputDBVersion' for partition: " << dbParams_.partitions_.front();
    handleException( __func__, ss.str() ); 
  }
#endif
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabaseCache() {
  
  // Reset all DbParams except for those concerning databae cache
  DbParams temp;
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
#ifdef USING_DATABASE_CACHE
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
#ifdef USING_DATABASE_CACHE
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
  
  // Input module.xml file
  if ( dbParams_.inputModuleXml_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to input 'module.xml' file!";
  } else {
    if ( checkFileExists( dbParams_.inputModuleXml_ ) ) { 
      try { 
#ifdef USING_NEW_DATABASE_MODEL
	deviceFactory(__func__)->setConnectionInputFileName( dbParams_.inputModuleXml_ ); 
#else
	deviceFactory(__func__)->setFedFecConnectionInputFileName( dbParams_.inputModuleXml_ ); 
#endif
      } catch (...) { 
	handleException( __func__ ); 
      }
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Added input 'module.xml' file: " << dbParams_.inputModuleXml_;
    } else {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " No 'module.xml' file found at " << dbParams_.inputModuleXml_;
      dbParams_.inputModuleXml_ = ""; 
    }
  }
  
  // Input dcuinfo.xml file
  if ( dbParams_.inputDcuInfoXml_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( dbParams_.inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(__func__)->setTkDcuInfoInputFileName( dbParams_.inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( __func__ ); 
      }
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Added 'dcuinfo.xml' file: " << dbParams_.inputDcuInfoXml_;
    } else {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " No 'dcuinfo.xml' file found at " << dbParams_.inputDcuInfoXml_;
      dbParams_.inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( dbParams_.inputFecXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL paths to input 'fec.xml' files!";
  } else {
    std::vector<std::string>::iterator iter = dbParams_.inputFecXml_.begin();
    for ( ; iter != dbParams_.inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL path to input 'fec.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( dbParams_.inputFecXml_.size() == 1 ) {
	      deviceFactory(__func__)->setFecInputFileName( *iter ); 
	    } else {
	      deviceFactory(__func__)->addFecFileName( *iter ); 
	    }
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
  if ( dbParams_.inputFedXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL paths to input 'fed.xml' files!";
  } else {
    std::vector<std::string>::iterator iter = dbParams_.inputFedXml_.begin();
    for ( ; iter != dbParams_.inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_) 
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL path to input 'fed.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( dbParams_.inputFedXml_.size() == 1 ) {
	      deviceFactory(__func__)->setFedInputFileName( *iter ); 
	    } else {
	      deviceFactory(__func__)->addFedFileName( *iter ); 
	    }
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
void SiStripConfigDb::createPartition( const std::string& partition_name,
				       const SiStripFecCabling& fec_cabling,
				       const DcuDetIdMap& dcu_detid_map ) {
  
  // Set partition name and version
  dbParams_.partitions_.push_back( partition_name );
  dbParams_.fecMajor_ = 0;
  dbParams_.fecMinor_ = 0;
  
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Creating partition " << dbParams_.partitions_.front();

  // Create new partition based on device descriptions
  createDeviceDescriptions( fec_cabling );
  if ( !devices_.empty() ) {
    try {
      std::stringstream ss; 
      ss << "/tmp/fec_" << dbParams_.partitions_.front() << ".xml";
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
#ifdef USING_NEW_DATABASE_MODEL
      deviceFactory(__func__)->createPartition( devices_,
						&dbParams_.fecMajor_, 
						&dbParams_.fecMinor_, 
						dbParams_.partitions_.front() );
#else
      deviceFactory(__func__)->createPartition( devices_,
						&dbParams_.fecMajor_, 
						&dbParams_.fecMinor_, 
						dbParams_.partitions_.front(),
						dbParams_.partitions_.front() ); 
#endif
    } catch (...) { 
      std::stringstream ss; 
      ss << "Failed to create new partition with name "
	 << dbParams_.partitions_.front() << " and version " 
	 << dbParams_.fecMajor_ << "." << dbParams_.fecMinor_;
      handleException( __func__, ss.str() );
    } 
  }
  
  // Create and upload FED descriptions
  createFedDescriptions( fec_cabling );
  if ( !feds_.empty() ) {
    try {
      std::stringstream ss; 
      ss << "/tmp/fed_" << dbParams_.partitions_.front() << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      uploadFedDescriptions( true );
    } catch(...) {
      std::stringstream ss; 
      ss << "Failed to create and upload FED descriptions"
	 << " to partition with name "
	 << dbParams_.partitions_.front() << " and version " 
	 << dbParams_.fedMajor_ << "." << dbParams_.fedMinor_;
      handleException( __func__, ss.str() );
    }
  }

  // Create and upload FED connections
  createFedConnections( fec_cabling );
  if ( !connections_.empty() ) {
    try {
      uploadFedConnections( true );
    } catch(...) {
      std::stringstream ss; 
      ss << "Failed to add FedChannelConnectionDescription!";
      handleException( __func__, ss.str() );
    }
    try {
      std::stringstream ss; 
      ss << "/tmp/module_" << dbParams_.partitions_.front() << ".xml";
#ifdef USING_NEW_DATABASE_MODEL
      ConnectionFactory* factory = deviceFactory(__func__);
#else
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
#endif
      factory->setOutputFileName( ss.str() );
#ifndef USING_NEW_DATABASE_MODEL
      deviceFactory(__func__)->write(); //@@ corresponding method in new model???
#endif
    } catch(...) {
      std::stringstream ss; 
      ss << "Failed to create and upload ConnectionDescriptions"
	 << " to partition with name "
	 << dbParams_.partitions_.front() << " and version " 
	 << dbParams_.cabMajor_ << "." << dbParams_.cabMinor_;
      handleException( __func__, ss.str() );
    }
  }

  // Create and upload DCU-DetId map
  dcuDetIdMap_.clear();
  dcuDetIdMap_ = dcu_detid_map;
  if ( !dcuDetIdMap_.empty() ) {
    uploadDcuDetIdMap(); 
  }
  
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Finished creating partition " << dbParams_.partitions_.front();
  
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






  
