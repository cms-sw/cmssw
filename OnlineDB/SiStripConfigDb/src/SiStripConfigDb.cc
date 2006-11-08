// Last commit: $Id: SiStripConfigDb.cc,v 1.22 2006/11/07 10:24:04 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/SiStripConfigDb.cc,v $

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
  usingDb_(true), 
  // Database connection params
  user_(""), 
  passwd_(""), 
  path_(""), 
  partition_(), 
  // Input XML
  inputModuleXml_(""), 
  inputDcuInfoXml_(""), 
  inputFecXml_(), 
  inputFedXml_(), 
  inputDcuConvXml_(""),
  // Output XML
  outputModuleXml_("/tmp/module.xml"), 
  outputDcuInfoXml_("/tmp/dcuinfo.xml"), 
  outputFecXml_("/tmp/fec.xml"), 
  outputFedXml_("/tmp/fed.xml"),
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
  
  // Set all DB params in struct
  dbParams_.usingDb_ = pset.getUntrackedParameter<bool>("UsingDb",true); 
  dbParams_.confdb( pset.getUntrackedParameter<string>("ConfDb","") );
  dbParams_.partition_ = pset.getUntrackedParameter<string>("Partition","");
  dbParams_.major_ = pset.getUntrackedParameter<unsigned int>("MajorVersion",0);
  dbParams_.minor_ = pset.getUntrackedParameter<unsigned int>("MinorVersion",0);
  dbParams_.inputModuleXml_ = pset.getUntrackedParameter<string>("InputModuleXml","");
  dbParams_.inputDcuInfoXml_ = pset.getUntrackedParameter<string>("InputDcuInfoXml",""); 
  dbParams_.inputFecXml_ = pset.getUntrackedParameter< vector<string> >( "InputFecXml", vector<string>(1,"") ); 
  dbParams_.inputFedXml_ = pset.getUntrackedParameter< vector<string> >( "InputFedXml", vector<string>(1,"") ); 
  dbParams_.inputDcuConvXml_ = pset.getUntrackedParameter<string>( "InputDcuConvXml","" );
  dbParams_.outputModuleXml_ = pset.getUntrackedParameter<string>("OutputModuleXml","/tmp/module.xml");
  dbParams_.outputDcuInfoXml_ = pset.getUntrackedParameter<string>("OutputDcuInfoXml","/tmp/dcuinfo.xml");
  dbParams_.outputFecXml_ = pset.getUntrackedParameter<string>( "OutputFecXml", "/tmp/fec.xml" );
  dbParams_.outputFedXml_ = pset.getUntrackedParameter<string>( "OutputFedXml", "/tmp/fed.xml" );
  
  // Copy values to private member data 
  usingDb_ = dbParams_.usingDb_;
  user_ = dbParams_.user_;
  passwd_ = dbParams_.passwd_;
  path_ = dbParams_.path_;
  partition_.name_ = dbParams_.partition_;
  partition_.major_ = dbParams_.major_;
  partition_.minor_ = dbParams_.minor_;
  inputModuleXml_ = dbParams_.inputModuleXml_;
  inputDcuInfoXml_ =dbParams_.inputDcuInfoXml_;
  inputFecXml_ = dbParams_.inputFecXml_;
  inputFedXml_ = dbParams_.inputFedXml_;
  inputDcuConvXml_ = dbParams_.inputDcuConvXml_;
  outputModuleXml_ = dbParams_.outputModuleXml_;
  outputDcuInfoXml_ = dbParams_.outputDcuInfoXml_;
  outputFecXml_ = dbParams_.outputFecXml_;
  outputFedXml_ = dbParams_.outputFedXml_;

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
  usingDb_(true), 
  // Database connection params
  user_(""), 
  passwd_(""), 
  path_(""), 
  partition_(), 
  // Input XML
  inputModuleXml_(""), 
  inputDcuInfoXml_(""), 
  inputFecXml_(), 
  inputFedXml_(), 
  inputDcuConvXml_(""),
  // Output XML
  outputModuleXml_("/tmp/module.xml"), 
  outputDcuInfoXml_("/tmp/dcuinfo.xml"), 
  outputFecXml_("/tmp/fec.xml"), 
  outputFedXml_("/tmp/fed.xml"),
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
  
  uint32_t ipass = confdb.find("/");
  uint32_t ipath = confdb.find("@");
  if ( ipass != string::npos && 
       ipath != string::npos ) {
    user_   = confdb.substr(0,ipass); 
    passwd_ = confdb.substr(ipass+1,ipath-ipass-1); 
    path_   = confdb.substr(ipath+1,confdb.size());
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Unexpected value for 'ConfDb' configurable: " << confdb;
  }
  partition_.name_ = partition;
  partition_.major_ = major;
  partition_.minor_ = minor;
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
  usingDb_(true), 
  // Database connection params
  user_( user ), 
  passwd_( passwd ), 
  path_( path ), 
  partition_(), 
  // Input XML
  inputModuleXml_(""), 
  inputDcuInfoXml_(""), 
  inputFecXml_(), 
  inputFedXml_(), 
  inputDcuConvXml_(""),
  // Output XML
  outputModuleXml_("/tmp/module.xml"), 
  outputDcuInfoXml_("/tmp/dcuinfo.xml"), 
  outputFecXml_("/tmp/fec.xml"), 
  outputFedXml_("/tmp/fed.xml"),
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
  
  partition_.name_ = partition;
  partition_.major_ = major;
  partition_.minor_ = minor;
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
  usingDb_(false), 
  // Database connection params
  user_(""), 
  passwd_(""), 
  path_(""), 
  partition_(),
  // Input XML
  inputModuleXml_( input_module_xml ), 
  inputDcuInfoXml_( input_dcuinfo_xml ), 
  inputFecXml_( input_fec_xml ), 
  inputFedXml_( input_fed_xml ),
  inputDcuConvXml_(""),
  // Output XML
  outputModuleXml_( output_module_xml ), 
  outputDcuInfoXml_( output_dcuinfo_xml ), 
  outputFecXml_( output_fec_xml ), 
  outputFedXml_( output_fed_xml ),
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
  user_(""),
  passwd_(""),
  path_(""),
  partition_(""), 
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
  inputFecXml_.clear();
  inputFedXml_.clear();
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DbParams::~DbParams() {
  inputFecXml_.clear();
  inputFedXml_.clear();
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DbParams::print( stringstream& ss ) const {
  ss << "[SiStripConfigDb::DbParams::" << __func__ << "]"
     << " Using database: " << usingDb_ << endl
     << " User/Passwd@Path: " 
     << user_ << "/" << passwd_ << "/" << path_ << endl
     << " Partition: " << partition_ << endl
     << " Major/minor versions: " 
     << major_ << "/" << minor_ << endl;
  // Input
  ss << " Input \"module.xml\" file: " << inputModuleXml_ << endl
     << " Input \"dcuinfo.xml\" file: " << inputDcuInfoXml_ << endl
     << " Input \"fec.xml\" file(s): ";
  vector<string>::const_iterator ifec = inputFecXml_.begin();
  for ( ; ifec != inputFecXml_.end(); ifec++ ) { ss << "\"" << *ifec << "\", "; }
  ss << endl;
  ss << " Input \"fed.xml\" file(s): ";
  vector<string>::const_iterator ifed = inputFedXml_.begin();
  for ( ; ifed != inputFedXml_.end(); ifed++ ) { ss << "\"" << *ifed << "\", "; }
  ss << endl;
  // Output 
  ss << " Output \"module.xml\" file: " << outputModuleXml_ << endl
     << " Output \"dcuinfo.xml\" file: " << outputDcuInfoXml_ << endl
     << " Output \"fec.xml\" file(s): "
     << "\"" << outputFecXml_ << "\", " << endl
     << " Output \"fed.xml\" file(s): "
     << "\"" << outputFedXml_ << "\", " << endl;
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
//   } else {
//     edm::LogWarning(mlConfigDb_)
//       << "[SiStripConfigDb::DbConfdb::" << __func__ << "]"
//       << " Unexpected value for \"confdb\": \"" << confdb_ << "\"";
  }
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
  if ( usingDb_ ) { 
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

  // Retrieve db connection parameters
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL database connection parameter(s): user/passwd@path: " 
      << user_ << "/" << passwd_ << "@" << path_ 
      << " Attempting to retrieve parameters from CONFDB env var...";
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      edm::LogError(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " NULL data connection parameter(s) from CONFDB env var: user/passwd@path: " 
	<< user_ << "/" << passwd_ << "@" << path_ 
	<< " Aborting connection to database...";
      return;
    }
  }

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
    } else {
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Env. var. TNS_ADMIN is set to: " << env_var;
    }
  } else {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Env. var. TNS_ADMIN is not set!"
      << " Setting to '" << env_var << "'...";
    setenv(tns_admin.c_str(),env_var.c_str(),1); 
  }
  
  // Retrieve partition name
  string partition = "ENV_CMS_TK_PARTITION";
  if ( partition_.name_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " Database partition name not specified!"
      << " Attempting to read 'ENV_CMS_TK_PARTITION' env. var...";
    if ( getenv(partition.c_str()) != NULL ) { 
      partition_.name_ = getenv(partition.c_str()); 
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Database partition name set using '"
	<< partition << "' env. var: "
	<< partition_.name_;
    } 
    else { 
      edm::LogError(mlConfigDb_) 
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Unable to retrieve database partition name!"
	<< " '" << partition << "' env var not specified!"
	<< " Aborting connection to database...";
      return;
    } 
  } 
  
  // Create device factory object
  try { 
    factory_ = new DeviceFactory( user_, passwd_, path_ ); 
  } catch (...) { 
    stringstream ss; 
    ss << "Failed to connect to database using parameters '" 
       << user_ << "/" << passwd_ << "@" << path_ 
       << "' and partition '" << partition_.name_ << "'";
    handleException( __func__, ss.str() );
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__func__) ) { 
    stringstream ss;
    ss << "[SiStripConfigDb::" << __func__ << "]"
       << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << " using database connection parameters '" 
       << user_ << "/" << passwd_ << "@" << path_
       << "' and partition '" << partition_.name_ << "'";
    LogTrace(mlConfigDb_) << ss.str();
  } else {
    edm::LogError(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database using connection parameters '" 
      << user_ << "/" << passwd_ << "@" << path_
      << "' and partition '" << partition_.name_ << "'";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( usingDb_ ); 
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  // DCU-DetId 
  try { 
    deviceFactory(__func__)->addDetIdPartition( partition_.name_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addDetIdPartition; for partition: " << partition_.name_;
    handleException( __func__, ss.str() );
  }

  // FED-FEC connections
  try { 
    deviceFactory(__func__)->createInputDBAccess();
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
  }

  try {
    deviceFactory(__func__)->setInputDBVersion( partition_.name_,
						partition_.major_,
						partition_.minor_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'setInputDBVersion; for partition: " << partition_.name_;
    handleException( __func__, ss.str() ); 
  }
  
  // DCU conversion factors
  try {
    //deviceFactory(__func__)->addConversionPartition( partition_.name_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addConversionPartition; for partition: " << partition_.name_;
    handleException( __func__, ss.str() ); 
  }
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingXmlFiles() {

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
       << " using xml files";
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
    deviceFactory(__func__)->setUsingDb( usingDb_ );
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }

  try { 
    deviceFactory(__func__)->createInputFileAccess();
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputFileAccess'" ); 
  }
  
  // Input module.xml file
  if ( inputModuleXml_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to input 'module.xml' file!";
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	deviceFactory(__func__)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { 
	handleException( __func__ ); 
      }
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Added input 'module.xml' file: " << inputModuleXml_;
    } else {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " No 'module.xml' file found at " << inputModuleXml_;
      inputModuleXml_ = ""; 
    }
  }
  
  // Input dcuinfo.xml file
  if ( inputDcuInfoXml_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(__func__)->setTkDcuInfoInputFileName( inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( __func__ ); 
      }
      LogTrace(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " Added 'dcuinfo.xml' file: " << inputDcuInfoXml_;
    } else {
      edm::LogWarning(mlConfigDb_)
	<< "[SiStripConfigDb::" << __func__ << "]"
	<< " No 'dcuinfo.xml' file found at " << inputDcuInfoXml_;
      inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( inputFecXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFecXml_.begin();
    for ( ; iter != inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL path to input 'fec.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFecXml_.size() == 1 ) {
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
  if ( inputFedXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL paths to input 'fed.xml' files!";
  } else {
    vector<string>::iterator iter = inputFedXml_.begin();
    for ( ; iter != inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_) 
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL path to input 'fed.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFedXml_.size() == 1 ) {
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
  if ( outputModuleXml_ == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'module.xml' file!"
      << " Setting to '/tmp/module.xml'...";
    outputModuleXml_ = "/tmp/module.xml"; 
  } else {
    try { 
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( outputModuleXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'module.xml' file!" ); 
    }
  }

  // Output dcuinfo.xml file
  if ( outputDcuInfoXml_ == "" ) { 
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'dcuinfo.xml' file!"
      << " Setting to '/tmp/dcuinfo.xml'...";
    outputModuleXml_ = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( outputDcuInfoXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'dcuinfo.xml' file!" ); 
    }
  }

  // Output fec.xml file
  if ( outputFecXml_ == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fec.xml' file!"
      << " Setting to '/tmp/fec.xml'...";
    outputFecXml_ = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( outputFecXml_ ); 
    } catch (...) { 
      handleException( __func__, "Problems setting output 'fec.xml' file!" ); 
    }
  }

  // Output fed.xml file
  if ( outputFedXml_ == "" ) {
    edm::LogWarning(mlConfigDb_) 
      << "[SiStripConfigDb::" << __func__ << "]"
      << " NULL path to output 'fed.xml' file!"
      << " Setting to '/tmp/fed.xml'...";
    outputFedXml_ = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( outputFedXml_ ); 
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
  partition_.name_ = partition_name;
  partition_.major_ = 0;
  partition_.minor_ = 0;

  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Creating partition " << partition_.name_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fec_" << partition_.name_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->createPartition( devices,
						resets, 
						&partition_.major_, 
						&partition_.minor_, 
						&partition_.major_,
						&partition_.minor_,
						partition_.name_,
						partition_.name_ );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create new partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      handleException( __func__, ss.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/dcuconv_" << partition_.name_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      handleException( __func__, ss.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fed_" << partition_.name_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->setFed9UDescriptions( feds,
						     partition_.name_,
						     &(uint16_t)partition_.major_,
						     &(uint16_t)partition_.minor_,
						     1 ); // new major version
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FED descriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
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
      ss << "/tmp/module_" << partition_.name_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__func__)->write();
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      
      handleException( __func__, ss.str() );
    }
  }

  LogTrace(mlConfigDb_)
    << "[SiStripConfigDb::" << __func__ << "]"
    << " Finished creating partition " << partition_.name_;
  
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






  
