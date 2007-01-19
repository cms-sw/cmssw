// Last commit: $Id: $
// Latest tag:  $Name: $
// Location:    $Source: $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
uint32_t SiStripConfigDb::cntr_ = 0;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( const edm::ParameterSet& pset,
				  const edm::ActivityRegistry& activity ) :
  factory_(0), 
  dbParams_(),
  // Local cache
  devices_(), 
  piaResets_(), 
  feds_(), 
  connections_(), 
  dcuDetIdMap_(), 
  dcuConversionFactors_(),
  // Reset flags
  resetDevices_(true), 
  resetPiaResets_(true), 
  resetFeds_(true), 
  resetConnections_(true), 
  resetDcuDetIdMap_(true), 
  resetDcuConvs_(true),
  // Misc
  usingStrips_(true)
{
  cntr_++;
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Constructing object using Service..."
    << " (Class instance: " << cntr_ << ")";

  // Set DB connection parameters
  dbParams_.reset();
  dbParams_.setParams( pset );
  
  stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters:" << endl
     << dbParams_ << endl;
  edm::LogVerbatim(mlConfigDb_) << ss.str();
  
  // Open connection
  openDbConnection();

}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( string confdb, 
				  string partition,
				  uint32_t major,
				  uint32_t minor ) :
  factory_(0), 
  dbParams_(),
  // Local cache
  devices_(), 
  piaResets_(), 
  feds_(), 
  connections_(), 
  dcuDetIdMap_(), 
  dcuConversionFactors_(),
  // Reset flags
  resetDevices_(true), 
  resetPiaResets_(true), 
  resetFeds_(true), 
  resetConnections_(true), 
  resetDcuDetIdMap_(true), 
  resetDcuConvs_(true),
  // Misc
  usingStrips_(true)
{
  cntr_++;
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Constructing object..."
    << " (Class instance: " << cntr_ << ")";

  dbParams_.reset();
  dbParams_.usingDb_ = true; 
  dbParams_.confdb( confdb );
  dbParams_.partition_ = partition; 
  dbParams_.major_ = major;
  dbParams_.minor_ = minor;

  stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters:" << endl
     << dbParams_ << endl;
  edm::LogVerbatim(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( string user, 
				  string passwd, 
				  string path,
				  string partition,
				  uint32_t major,
				  uint32_t minor ) :
  factory_(0), 
  dbParams_(),
  // Local cache
  devices_(), 
  piaResets_(), 
  feds_(), 
  connections_(), 
  dcuDetIdMap_(), 
  dcuConversionFactors_(),
  // Reset flags
  resetDevices_(true), 
  resetPiaResets_(true), 
  resetFeds_(true), 
  resetConnections_(true), 
  resetDcuDetIdMap_(true), 
  resetDcuConvs_(true),
  // Misc
  usingStrips_(true)
{
  cntr_++;
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Constructing object..."
    << " (Class instance: " << cntr_ << ")";
  
  dbParams_.reset();
  dbParams_.usingDb_ = true; 
  dbParams_.confdb( user, passwd, path );
  dbParams_.partition_ = partition; 
  dbParams_.major_ = major;
  dbParams_.minor_ = minor;

  stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters:" << endl
     << dbParams_ << endl;
  edm::LogVerbatim(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( string input_module_xml,
				  string input_dcuinfo_xml,
				  vector<string> input_fec_xml,
				  vector<string> input_fed_xml,
				  string output_module_xml,
				  string output_dcuinfo_xml,
				  string output_fec_xml,
				  string output_fed_xml ) : 
  factory_(0), 
  dbParams_(),
  // Local cache
  devices_(), 
  piaResets_(), 
  feds_(), 
  connections_(), 
  dcuDetIdMap_(),
  dcuConversionFactors_(),
  // Reset flags
  resetDevices_(true), 
  resetPiaResets_(true), 
  resetFeds_(true), 
  resetConnections_(true), 
  resetDcuDetIdMap_(true),
  resetDcuConvs_(true),
  // Misc
  usingStrips_(true)
{
  cntr_++;
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Constructing object..."
    << " (Class instance: " << cntr_ << ")";

  dbParams_.reset();
  dbParams_.usingDb_ = false; 
  dbParams_.inputModuleXml_ = input_module_xml; 
  dbParams_.inputDcuInfoXml_ = input_dcuinfo_xml; 
  dbParams_.inputFecXml_ = input_fec_xml; 
  dbParams_.inputFedXml_ = input_fed_xml; 
  dbParams_.outputModuleXml_ = output_module_xml; 
  dbParams_.outputDcuInfoXml_ = output_dcuinfo_xml; 
  dbParams_.outputFecXml_ = output_fec_xml; 
  dbParams_.outputFedXml_ = output_fed_xml; 

  stringstream ss;
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Database connection parameters:" << endl
     << dbParams_ << endl;
  edm::LogVerbatim(mlConfigDb_) << ss.str();

}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  //closeDbConnection();
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
  partition_(null_), 
  major_(0),
  minor_(0),
  inputModuleXml_(""),
  inputDcuInfoXml_(""),
  inputFecXml_(),
  inputFedXml_(),
  inputDcuConvXml_(""),
  outputModuleXml_("/tmp/module.xml"),
  outputDcuInfoXml_("/tmp/dcuinfo.xml"),
  outputFecXml_("/tmp/fec.xml"),
  outputFedXml_("/tmp/fed.xml")
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
  usingDb_ = true;
  confdb_ = null_;
  confdb( confdb_ );
  partition_ = null_;
  major_ = 0;
  minor_ = 0;
  inputModuleXml_ = "";
  inputDcuInfoXml_ = "";
  inputFecXml_ = vector<string>(1,"");
  inputFedXml_ = vector<string>(1,"");
  inputDcuConvXml_ = "";
  outputModuleXml_ = "";
  outputDcuInfoXml_ = "";
  outputFecXml_ = "";
  outputFedXml_ = "";
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::setParams( const edm::ParameterSet& pset ) {
  reset();
  usingDb_ = pset.getUntrackedParameter<bool>("UsingDb",true); 
  confdb( pset.getUntrackedParameter<string>("ConfDb","") );
  partition_ = pset.getUntrackedParameter<string>("Partition","");
  major_ = pset.getUntrackedParameter<unsigned int>("MajorVersion",0);
  minor_ = pset.getUntrackedParameter<unsigned int>("MinorVersion",0);
  inputModuleXml_ = pset.getUntrackedParameter<string>("InputModuleXml","");
  inputDcuInfoXml_ = pset.getUntrackedParameter<string>("InputDcuInfoXml",""); 
  inputFecXml_ = pset.getUntrackedParameter< vector<string> >( "InputFecXml", vector<string>(1,"") ); 
  inputFedXml_ = pset.getUntrackedParameter< vector<string> >( "InputFedXml", vector<string>(1,"") ); 
  inputDcuConvXml_ = pset.getUntrackedParameter<string>( "InputDcuConvXml","" );
  outputModuleXml_ = pset.getUntrackedParameter<string>("OutputModuleXml","/tmp/module.xml");
  outputDcuInfoXml_ = pset.getUntrackedParameter<string>("OutputDcuInfoXml","/tmp/dcuinfo.xml");
  outputFecXml_ = pset.getUntrackedParameter<string>( "OutputFecXml", "/tmp/fec.xml" );
  outputFedXml_ = pset.getUntrackedParameter<string>( "OutputFedXml", "/tmp/fed.xml" );
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::confdb( const string& confdb ) {
  confdb_ = confdb;
  uint32_t ipass = confdb.find("/");
  uint32_t ipath = confdb.find("@");
  if ( ipass != string::npos && 
       ipath != string::npos ) {
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
void SiStripConfigDb::DbParams::confdb( const string& user,
					const string& passwd,
					const string& path ) {
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
void SiStripConfigDb::DbParams::print( stringstream& ss ) const {
  ss << " Using database            : " << usingDb_ << endl
     << " ConfDb                    : " << confdb_ << endl
     << " User/Passwd@Path          : " << user_ << "/" << passwd_ << "@" << path_ << endl
     << " Partition                 : " << partition_ << endl
     << " Major/minor versions      : " << major_ << "/" << minor_ << endl;
  // Input
  ss << " Input \"module.xml\" file   : " << inputModuleXml_ << endl
     << " Input \"dcuinfo.xml\" file  : " << inputDcuInfoXml_ << endl
     << " Input \"fec.xml\" file(s)   : ";
  vector<string>::const_iterator ifec = inputFecXml_.begin();
  for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << *ifec << ", "; }
  ss << endl;
  ss << " Input \"fed.xml\" file(s)   : ";
  vector<string>::const_iterator ifed = inputFedXml_.begin();
  for ( ; ifed != inputFedXml_.end(); ifed++ ) { ss << *ifed << ", "; }
  ss << endl;
  // Output 
  ss << " Output \"module.xml\" file  : " << outputModuleXml_ << endl
     << " Output \"dcuinfo.xml\" file : " << outputDcuInfoXml_ << endl
     << " Output \"fec.xml\" file(s)  : " << outputFecXml_ << endl
     << " Output \"fed.xml\" file(s)  : " << outputFedXml_ << endl;
}

// -----------------------------------------------------------------------------
// 
ostream& operator<< ( ostream& os, const SiStripConfigDb::DbParams& params ) {
  stringstream ss;
  params.print(ss);
  os << ss.str();
  return os;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::openDbConnection() {
  
  // Establish database connection
  if ( dbParams_.usingDb_ ) { 
    usingDatabase(); 
  } else { 
    usingXmlFiles(); 
  }
  
  // Refresh local cache
  refreshLocalCaches();

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::closeDbConnection() {
  try { 
    if ( factory_ ) { delete factory_; }
  } catch (...) { handleException( __func__, "Attempting to close database connection..." ); }
  factory_ = 0; 
}

// -----------------------------------------------------------------------------
//
DeviceFactory* const SiStripConfigDb::deviceFactory( string method_name ) const { 
  if ( factory_ ) { return factory_; }
  else { 
    if ( method_name != "" ) { 
      stringstream ss;
      ss << "[SiStripConfigDb::" << method_name << "]"
	 << " NULL pointer to DeviceFactory!";
      edm::LogError(mlConfigDb_) << ss.str();
    }
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {
  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Using a database account...";
  
  // Check TNS_ADMIN env var
  string tns_admin = "TNS_ADMIN";
  string env_var = "/afs/cern.ch/project/oracle/admin";
  if ( getenv(tns_admin.c_str()) != NULL ) { 
    string tmp = getenv(tns_admin.c_str()); 
    if ( tmp == "." ) { 
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Env. var. TNS_ADMIN is set to 'pwd'!"
	<< " Setting to '" << env_var << "'...";
      setenv(tns_admin.c_str(),env_var.c_str(),1); 
    } else if ( tmp != env_var ) { 
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Env. var. TNS_ADMIN is set to '" << tmp
	<< "'! Setting to '" << env_var << "'...";
      setenv(tns_admin.c_str(),env_var.c_str(),1); 
    } else {
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Env. var. TNS_ADMIN is set to: " << tmp;
    }
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Env. var. TNS_ADMIN is not set!"
      << " Setting to '" << env_var << "'...";
    setenv(tns_admin.c_str(),env_var.c_str(),1); 
  }
  
  // Retrieve connection params from CONFDB env. var. and overwrite .cfg values 
  string user = "";
  string passwd = "";
  string path = "";
  DbAccess::getDbConfiguration( user, passwd, path );
  if ( user != "" && passwd != "" && path != "" ) {
    stringstream ss;
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
    LogTrace(mlConfigDb_) << ss.str() << endl;
    dbParams_.confdb( user, passwd, path );
  } else if ( dbParams_.user_ != null_ && 
	      dbParams_.passwd_ != null_ && 
	      dbParams_.path_ != null_ ) { 
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Setting \"user/passwd@path\" to \""
      << user << "/" << passwd << "@" << path
      << "\" using ConfDb read from .cfg file";
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unable to retrieve 'user/passwd@path' parameters"
      << " from 'CONFDB' environmental variable or .cfg file"
      << " (present value is \"" 
      << dbParams_.user_ << "/" 
      << dbParams_.passwd_ << "@" 
      << dbParams_.path_ 
      << "\"). Aborting connection to database...";
    return;
  }
  
  // Retrieve partition name from ENV_CMS_TK_PARTITION env. var. and overwrite .cfg value
  string partition = "ENV_CMS_TK_PARTITION";
  if ( getenv(partition.c_str()) != NULL ) { 
    stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " Setting \"PARTITION\" to \""
       << getenv( partition.c_str() )
       << "\" using 'ENV_CMS_TK_PARTITION' environmental variable";
    if ( dbParams_.partition_ != "" ) { 
      ss << " (Overwriting existing value of \""
	 << dbParams_.partition_
	 << "\" read from .cfg file)";
    }
    LogTrace(mlConfigDb_) << ss.str() << endl;
    dbParams_.partition_ = getenv( partition.c_str() ); 
  } else if ( dbParams_.partition_ != "" ) {
    LogTrace(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Setting \"partition\" to \""
      << getenv( partition.c_str() )
      << "\" using 'Partition' read from .cfg file";
  } else { 
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unable to retrieve 'partition' parameter"
      << " from 'CONFDB' environmental variable or .cfg file"
      << " (present value is \"" 
      << dbParams_.partition_
      << "\"). Aborting connection to database...";
    return;
  } 
  
  // Create device factory object
  try { 
    factory_ = new DeviceFactory( dbParams_.user_, 
				  dbParams_.passwd_, 
				  dbParams_.path_ ); 
  } catch (...) { 
    stringstream ss; 
    ss << "Failed to connect to database using parameters '" 
       << dbParams_.user_ << "/" 
       << dbParams_.passwd_ << "@" 
       << dbParams_.path_ 
       << "' and partition '" 
       << dbParams_.partition_ << "'";
    handleException( __func__, ss.str() );
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__func__) ) { 
    stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << ", using database account with parameters '" 
       << dbParams_.user_ << "/" 
       << dbParams_.passwd_ << "@" 
       << dbParams_.path_
       << "' and partition '" 
       << dbParams_.partition_ << "'";
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database using connection parameters '" 
      << dbParams_.user_ << "/" 
      << dbParams_.passwd_ << "@" 
      << dbParams_.path_
      << "' and partition '" 
      << dbParams_.partition_ << "'";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( dbParams_.usingDb_ ); 
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  // DCU-DetId 
  try { 
    deviceFactory(__func__)->addDetIdPartition( dbParams_.partition_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addDetIdPartition; for partition: " << dbParams_.partition_;
    handleException( __func__, ss.str() );
  }

  // FED-FEC connections
  try { 
    deviceFactory(__func__)->createInputDBAccess();
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
  }

  try {
    deviceFactory(__func__)->setInputDBVersion( dbParams_.partition_,
						dbParams_.major_,
						dbParams_.minor_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'setInputDBVersion; for partition: " << dbParams_.partition_;
    handleException( __func__, ss.str() ); 
  }
  
  // DCU conversion factors
  try {
    //deviceFactory(__func__)->addConversionPartition( dbParams_.partition_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addConversionPartition; for partition: " << dbParams_.partition_;
    handleException( __func__, ss.str() ); 
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
    stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << ", using XML description files";
    LogTrace(mlConfigDb_) << ss.str();
    cout << ss.str() << endl;
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
    deviceFactory(__func__)->createInputFileAccess();
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
	deviceFactory(__func__)->setFedFecConnectionInputFileName( dbParams_.inputModuleXml_ ); 
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
    vector<string>::iterator iter = dbParams_.inputFecXml_.begin();
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
    vector<string>::iterator iter = dbParams_.inputFedXml_.begin();
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
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
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
void SiStripConfigDb::refreshLocalCaches() {
  resetDeviceDescriptions();
  resetFedDescriptions();
  resetFedConnections();
  resetPiaResetDescriptions();
  resetDcuConversionFactors();
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createPartition( const string& partition_name,
				       const SiStripFecCabling& fec_cabling ) {
  
  // Set partition name and version
  dbParams_.partition_ = partition_name;
  dbParams_.major_ = 0;
  dbParams_.minor_ = 0;

  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Creating partition " << dbParams_.partition_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fec_" << dbParams_.partition_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->createPartition( devices,
						resets, 
						&dbParams_.major_, 
						&dbParams_.minor_, 
						&dbParams_.major_,
						&dbParams_.minor_,
						dbParams_.partition_,
						dbParams_.partition_ );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create new partition with name "
	 << dbParams_.partition_ << " and version " 
	 << dbParams_.major_ << "." << dbParams_.minor_;
      handleException( __func__, ss.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/dcuconv_" << dbParams_.partition_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << dbParams_.partition_ << " and version " 
	 << dbParams_.major_ << "." << dbParams_.minor_;
      handleException( __func__, ss.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fed_" << dbParams_.partition_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      uint16_t major = static_cast<uint16_t>(dbParams_.major_);
      uint16_t minor = static_cast<uint16_t>(dbParams_.minor_);
      deviceFactory(__func__)->setFed9UDescriptions( feds,
						     dbParams_.partition_,
						     &major,
						     &minor,
						     1 ); // new major version
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FED descriptions"
	 << " to partition with name "
	 << dbParams_.partition_ << " and version " 
	 << dbParams_.major_ << "." << dbParams_.minor_;
      handleException( __func__, ss.str() );
    }
  }

  // Create and upload FED connections
  const FedConnections& conns = createFedConnections( fec_cabling );
  if ( !conns.empty() ) {
    FedConnections::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) { 
      try {
	deviceFactory(__func__)->addFedChannelConnection( *iconn );
      } catch(...) {
	stringstream ss; 
	ss << "Failed to add FedChannelConnectionDescription!";
	handleException( __func__, ss.str() );
      }
    }
    try {
      stringstream ss; 
      ss << "/tmp/module_" << dbParams_.partition_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->write();
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << dbParams_.partition_ << " and version " 
	 << dbParams_.major_ << "." << dbParams_.minor_;
      
      handleException( __func__, ss.str() );
    }
  }

  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Finished creating partition " << dbParams_.partition_;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const string& method_name,
				       const string& extra_info ) { //throw (cms::Exception) {

  stringstream ss;
  try {
    //throw; // rethrow caught exception to be dealt with below
  } 

  catch ( const cms::Exception& e ) { 
    ss << " Caught cms::Exception in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw e; // rethrow cms::Exception
  }
  
  catch ( const oracle::occi::SQLException& e ) { 
    ss << " Caught oracle::occi::SQLException in method "
       << method_name << " with message: " << endl 
       << e.getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << endl;
  }

  catch ( const FecExceptionHandler& e ) {
    ss << " Caught FecExceptionHandler exception in method "
       << method_name << " with message: " << endl 
       << const_cast<FecExceptionHandler&>(e).getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << endl;
  }

  catch ( const ICUtils::ICException& e ) {
    ss << " Caught ICUtils::ICException in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << endl;
  }

  catch ( const exception& e ) {
    ss << " Caught std::exception in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << endl;
  }

  catch (...) {
    ss << " Caught unknown exception in method "
       << method_name << " (No message) " << endl;
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << ss.str() << endl;
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






  
