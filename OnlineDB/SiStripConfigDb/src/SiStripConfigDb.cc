// Last commit: $Id: SiStripConfigDb.cc,v 1.18 2006/09/07 20:31:19 bainbrid Exp $
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
  inputDcuConvXml_(),
  // Output XML
  outputModuleXml_(""), 
  outputDcuInfoXml_(""), 
  outputFecXml_(""), 
  outputFedXml_(""),
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
  edm::LogInfo(mlConfigDb_)
    << __func__ << " Constructing object..."
    << " (Class instance: " << cntr_ << ")";
  
  uint32_t ipass = confdb.find("/");
  uint32_t ipath = confdb.find("@");
  if ( ipass != string::npos && 
       ipath != string::npos ) {
    user_   = confdb.substr(0,ipass); 
    passwd_ = confdb.substr(ipass+1,ipath-ipass-1); 
    path_   = confdb.substr(ipath+1,confdb.size());
  } else {
    edm::LogError(mlConfigDb_)
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
  inputDcuConvXml_(),
  // Output XML
  outputModuleXml_(""), 
  outputDcuInfoXml_(""), 
  outputFecXml_(""), 
  outputFedXml_(""),
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
  edm::LogInfo(mlConfigDb_)
    << __func__ << " Constructing object..."
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
  inputDcuConvXml_( "" ),
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
  edm::LogInfo(mlConfigDb_)
    << __func__ << " Constructing object..."
    << " (Class instance: " << cntr_ << ")";
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  edm::LogInfo(mlConfigDb_)
    << __func__ << " Destructing object...";
  if ( cntr_ ) { cntr_--; }
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
    ostringstream os;
    os << "[" << __func__ << "]";
    if ( method_name != "" ) { os << " Method " << method_name; }
    else { os << " Unknown method"; }
    os << " trying to access NULL pointer to DeviceFactory!";
    edm::LogError(mlConfigDb_) << os;
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {

  // Retrieve db connection parameters
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << " NULL data connection parameter(s) given to constructor: " 
      << user_ << "/" << passwd_ << "@" << path_ << endl
      << " Attempting to retrieve paramters from CONFDB env var...";
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      edm::LogError(mlConfigDb_)
	<< " NULL data connection parameter(s) extracted from CONFDB env var: " 
	<< user_ << "/" << passwd_ << "@" << path_ << endl
	<< " Aborting connection to database...";
      return;
    }
  }

  // Check TNS_ADMIN env var
  if ( getenv("TNS_ADMIN") != NULL ) { 
    string tns_admin = getenv("TNS_ADMIN"); 
    if ( tns_admin == "." ) { 
      edm::LogError(mlConfigDb_)
	<< " Env. var. TNS_ADMIN is set to 'pwd'!"
	<< " Aborting connection to database...";
      return;
    } else {
      edm::LogVerbatim(mlConfigDb_)
	<< " Env. var. TNS_ADMIN is set to: " << tns_admin;
    }
  } else {
    edm::LogError(mlConfigDb_)
      << " Env. var. TNS_ADMIN is not set!"
      << " Aborting connection to database...";
    return;
  }

  // Retrieve partition name
  if ( partition_.name_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << " Database partition name not specified!"
      << " Attempting to read 'ENV_CMS_TK_PARTITION' env. var...";
    if ( getenv("ENV_CMS_TK_PARTITION") != NULL ) { 
      partition_.name_ = getenv("ENV_CMS_TK_PARTITION"); 
      edm::LogVerbatim(mlConfigDb_)
	<< " Database partition name set using 'ENV_CMS_TK_PARTITION' env. var: "
	<< partition_.name_;
    } 
    else { 
      edm::LogError(mlConfigDb_) 
	<< " Unable to retrieve database partition name!"
	<< " 'ENV_CMS_TK_PARTITION' env var not specified!"
	<< " Aborting connection to database...";
      return;
    } 
  }

  // Create device factory object
  try { 
    factory_ = new DeviceFactory( user_, passwd_, path_ ); 
  } catch (...) { 
    ostringstream os; 
    os << "Attempting to connect to database using parameters '" 
       << user_ << "/" << passwd_ << "@" << path_ 
       << "' and partition '" << partition_.name_ << "'";
    handleException( __func__, os.str() );
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__func__) ) { 
    ostringstream os;
    os << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << " using database connection parameters '" 
       << user_ << "/" << passwd_ << "@" << path_
       << "' and partition '" << partition_.name_ << "'";
    edm::LogInfo(mlConfigDb_) << os;
  } else {    
    edm::LogError(mlConfigDb_)
      << " NULL pointer to DeviceFactory!"
      << " Unable to connect to database!";
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setUsingDb( usingDb_ ); //@@ necessary?
  } catch (...) { 
    handleException( __func__, "Attempted to 'setUsingDb'" );
  }
  
  try { 
    deviceFactory(__func__)->createInputDBAccess();
  } catch (...) { 
    handleException( __func__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
  }

  // DCU-DetId 
  try { 
    deviceFactory(__func__)->addDetIdPartition( partition_.name_ );
    //deviceFactory(__func__)->addAllDetId();
  } catch (...) { 
    ostringstream os;
    os << "Attempted to 'addDetIdPartition; for partition: " << partition_.name_;
    handleException( __func__, os.str() );
  }

  // FED-FEC connections
  try {
    deviceFactory(__func__)->setInputDBVersion( partition_.name_,
						partition_.major_,
						partition_.minor_ );
  } catch (...) { 
    ostringstream os;
    os << "Attempted to 'setInputDBVersion; for partition: " << partition_.name_;
    handleException( __func__, os.str() ); 
  }
  
  // DCU conversion factors
  try {
    //deviceFactory(__func__)->addConversionPartition( partition_.name_ );
  } catch (...) { 
    ostringstream os;
    os << "Attempted to 'addConversionPartition; for partition: " << partition_.name_;
    handleException( __func__, os.str() ); 
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
    ostringstream os;
    os << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << " using xml files";
    edm::LogInfo(mlConfigDb_) << os;
    cout << os.str() << endl;
  } else {    
    edm::LogError(mlConfigDb_)
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
    edm::LogError(mlConfigDb_)
      << " NULL path to input 'module.xml' file!";
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	deviceFactory(__func__)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { handleException( __func__ ); }
      edm::LogInfo(mlConfigDb_)
	<< " Added input 'module.xml' file: " << inputModuleXml_;
    } else {
      edm::LogError(mlConfigDb_)
	<< " No 'module.xml' file found at " << inputModuleXml_;
      inputModuleXml_ = ""; 
    }
  }
  
  // Input dcuinfo.xml file
  if ( inputDcuInfoXml_ == "" ) {
    edm::LogWarning(mlConfigDb_)
      << " NULL path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(__func__)->setTkDcuInfoInputFileName( inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( __func__ ); 
      }
      edm::LogInfo(mlConfigDb_)
	<< " Added 'dcuinfo.xml' file: " << inputDcuInfoXml_;
    } else {
      edm::LogError(mlConfigDb_)
	<< " No 'dcuinfo.xml' file found at " << inputDcuInfoXml_;
      inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( inputFecXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << " NULL paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFecXml_.begin();
    for ( ; iter != inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_)
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
	  edm::LogInfo(mlConfigDb_) 
	    << " Added 'fec.xml' file: " << *iter;
	} else {
	  edm::LogError(mlConfigDb_) 
	    << " No 'fec.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }
    
  // Input FED xml files
  if ( inputFedXml_.empty() ) {
    edm::LogWarning(mlConfigDb_) 
      << " NULL paths to input 'fed.xml' files!";
  } else {
    vector<string>::iterator iter = inputFedXml_.begin();
    for ( ; iter != inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(mlConfigDb_) 
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
	  edm::LogInfo(mlConfigDb_) 
	    << " Added 'fed.xml' file: " << *iter;
	} else {
	  edm::LogError(mlConfigDb_) 
	    << " No 'fed.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }

  // Output module.xml file
  if ( outputModuleXml_ == "" ) { 
    edm::LogWarning(mlConfigDb_) 
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

  edm::LogInfo(mlConfigDb_)
    << " Creating partition " << partition_.name_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      ostringstream os; 
      os << "/tmp/fec_" << partition_.name_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( os.str() );
      deviceFactory(__func__)->createPartition( devices,
						resets, 
						&partition_.major_, 
						&partition_.minor_, 
						&partition_.major_,
						&partition_.minor_,
						partition_.name_,
						partition_.name_ );
    } catch (...) { 
      ostringstream os; 
      os << "Failed to create new partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(mlConfigDb_) << os;
      handleException( __func__, os.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      ostringstream os; 
      os << "/tmp/dcuconv_" << partition_.name_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( os.str() );
      deviceFactory(__func__)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      ostringstream os; 
      os << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(mlConfigDb_) << os;
      handleException( __func__, os.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      ostringstream os; 
      os << "/tmp/fed_" << partition_.name_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( os.str() );
      deviceFactory(__func__)->setFed9UDescriptions( feds,
							 partition_.name_,
							 &(uint16_t)partition_.major_,
							 &(uint16_t)partition_.minor_,
							 1 ); // new major version
    } catch(...) {
      ostringstream os; 
      os << "Failed to create and upload FED descriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(mlConfigDb_) << os;
      handleException( __func__, os.str() );
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
	ostringstream os; 
	os << "Failed to add FedChannelConnectionDescription!";
	handleException( __func__, os.str() );
      }
    }
    try {
      ostringstream os; 
      os << "/tmp/module_" << partition_.name_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(__func__);
      factory->setOutputFileName( os.str() );
      deviceFactory(__func__)->write();
    } catch(...) {
      ostringstream os; 
      os << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      
      handleException( __func__, os.str() );
    }
  }

  edm::LogInfo("FedCabling") << __func__ << " Finished!";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const string& method_name,
				       const string& extra_info ) { //throw (cms::Exception) {

  ostringstream os;
  try {
    //throw; // rethrow caught exception to be dealt with below
  } 

  catch ( const cms::Exception& e ) { 
    os << " Caught cms::Exception in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw e; // rethrow cms::Exception
  }
  
  catch ( const oracle::occi::SQLException& e ) { 
    os << " Caught oracle::occi::SQLException in method "
       << method_name << " with message: " << endl 
       << e.getMessage();
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << os.str() << endl;
  }

  catch ( const FecExceptionHandler& e ) {
    os << " Caught FecExceptionHandler exception in method "
       << method_name << " with message: " << endl 
       << const_cast<FecExceptionHandler&>(e).getMessage();
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << os.str() << endl;
  }

  catch ( const ICUtils::ICException& e ) {
    os << " Caught ICUtils::ICException in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << os.str() << endl;
  }

  catch ( const exception& e ) {
    os << " Caught std::exception in method "
       << method_name << " with message: " << endl 
       << e.what();
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << os.str() << endl;
  }

  catch (...) {
    os << " Caught unknown exception in method "
       << method_name << " (No message) " << endl;
    if ( extra_info != "" ) { os << "Additional info: " << extra_info << endl; }
    //throw cms::Exception(mlConfigDb_) << os.str() << endl;
  }
  
  // Message
  edm::LogError(mlConfigDb_) << os;

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






  
