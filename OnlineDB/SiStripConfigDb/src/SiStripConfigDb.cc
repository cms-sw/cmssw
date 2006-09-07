// Last commit: $Id: SiStripConfigDb.cc,v 1.17 2006/08/31 19:49:41 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/SiStripConfigDb.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const string SiStripConfigDb::logCategory_ = "SiStrip|ConfigDb";
uint32_t SiStripConfigDb::cntr_ = 0;

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
  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
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
  stringstream ss;
  ss << "[" << __PRETTY_FUNCTION__ << "]"
     << " Constructing object..."
     << " (Class instance: " << cntr_ << ")";
  edm::LogInfo(logCategory_) << ss.str();
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  stringstream ss;
  ss << "[" << __PRETTY_FUNCTION__ << "]"
     << " Destructing object...";
  edm::LogInfo(logCategory_) << ss.str();
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
  } catch (...) { handleException( __FUNCTION__, "Attempting to close database connection..." ); }
  factory_ = 0; 
}

// -----------------------------------------------------------------------------
//
DeviceFactory* const SiStripConfigDb::deviceFactory( string method_name ) const { 
  if ( factory_ ) { return factory_; }
  else { 
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]";
    if ( method_name != "" ) { ss << " Method " << method_name; }
    else { ss << " Unknown method"; }
    ss << " trying to access NULL pointer to DeviceFactory!";
    edm::LogError(logCategory_) << ss.str();
    return 0;
  }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {

  // Retrieve db connection parameters
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " NULL data connection parameter(s) given to constructor: " 
       << user_ << "/" << passwd_ << "@" << path_
       << " Attempting to retrieve paramters from CONFDB env var...";
    edm::LogWarning(logCategory_) << ss.str();
    
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      stringstream ss;
      ss << "[" << __PRETTY_FUNCTION__ << "]"
	 << " NULL data connection parameter(s) extracted from CONFDb env var: " 
	 << user_ << "/" << passwd_ << "@" << path_
	 << " Aborting connection to database...";
      edm::LogError(logCategory_) << ss.str();
      return;
    }
  }

  // Check TNS_ADMIN env var
  if ( getenv("TNS_ADMIN") != NULL ) { 
    string tns_admin = getenv("TNS_ADMIN"); 
    if ( tns_admin == "." ) { 
      stringstream ss;
      ss << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Env. var. TNS_ADMIN is set to '.' (pwd)!"
	 << " Aborting connection to database...";
      edm::LogError(logCategory_) << ss.str();
      return;
    } else {
      stringstream ss;
      ss << "[" << __PRETTY_FUNCTION__ << "]"
	 << " Env. var. TNS_ADMIN is set to: " << tns_admin;
      edm::LogVerbatim(logCategory_) << ss.str();
    }
  } else {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Env. var. TNS_ADMIN is not set!"
       << " Aborting connection to database...";
    edm::LogError(logCategory_) << ss.str();
    return;
  }

  // Retrieve partition name
  if ( partition_.name_ == "" ) {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Database partition name not specified!"
       << " Attempting to read 'ENV_CMS_TK_PARTITION' env. var...";
    edm::LogWarning(logCategory_) << ss.str(); 
    if ( getenv("ENV_CMS_TK_PARTITION") != NULL ) { 
      partition_.name_ = getenv("ENV_CMS_TK_PARTITION"); 
      stringstream ss;
      ss  << "[" << __PRETTY_FUNCTION__ << "]"
	  << " Database partition name set using 'ENV_CMS_TK_PARTITION' env. var: "
	  << partition_.name_;
      edm::LogVerbatim(logCategory_) << ss.str();
    } 
    else { 
      stringstream ss;
      ss  << "[" << __PRETTY_FUNCTION__ << "]"
	  << " Unable to retrieve database partition name!"
	  << " 'ENV_CMS_TK_PARTITION' env var not specified!"
	  << " Aborting connection to database...";
      edm::LogError(logCategory_) << ss.str();
      return;
    } 
  }

  // Create device factory object
  try { 
    factory_ = new DeviceFactory( user_, passwd_, path_ ); 
  } catch (...) { 
    stringstream ss; 
    ss << "Attempting to connect to database using parameters '" 
       << user_ << "/" << passwd_ << "@" << path_ 
       << "' and partition '" << partition_.name_ << "'";
    handleException( __FUNCTION__, ss.str() );
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__FUNCTION__) ) { 
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << " using database connection parameters '" 
       << user_ << "/" << passwd_ << "@" << path_
       << "' and partition '" << partition_.name_ << "'";
    edm::LogInfo(logCategory_) << ss.str();
  } else {    
    stringstream ss; 
    ss << "[" << __PRETTY_FUNCTION__ << "]" 
       << " NULL pointer to DeviceFactory!"
       << " Unable to connect to database!";
    edm::LogError(logCategory_) << ss.str();
    return; 
  }
  
  try { 
    deviceFactory(__FUNCTION__)->setUsingDb( usingDb_ ); //@@ necessary?
  } catch (...) { 
    handleException( __FUNCTION__, "Attempted to 'setUsingDb'" );
  }
  
  try { 
    deviceFactory(__FUNCTION__)->createInputDBAccess();
  } catch (...) { 
    handleException( __FUNCTION__, "Attempted to 'createInputDBAccess' for FED-FEC connections!" );
  }

  // DCU-DetId 
  try { 
    deviceFactory(__FUNCTION__)->addDetIdPartition( partition_.name_ );
    //deviceFactory(__FUNCTION__)->addAllDetId();
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addDetIdPartition; for partition: " << partition_.name_;
    handleException( __FUNCTION__, ss.str() );
  }

  // FED-FEC connections
  try {
    deviceFactory(__FUNCTION__)->setInputDBVersion( partition_.name_,
						    partition_.major_,
						    partition_.minor_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'setInputDBVersion; for partition: " << partition_.name_;
    handleException( __FUNCTION__, ss.str() ); 
  }
  
  // DCU conversion factors
  try {
    //deviceFactory(__FUNCTION__)->addConversionPartition( partition_.name_ );
  } catch (...) { 
    stringstream ss;
    ss << "Attempted to 'addConversionPartition; for partition: " << partition_.name_;
    handleException( __FUNCTION__, ss.str() ); 
  }
  
}  

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingXmlFiles() {

  // Create device factory object
  try { 
    factory_ = new DeviceFactory(); 
  } catch (...) { 
    handleException( __FUNCTION__, "Attempting to create DeviceFactory for use with xml files" );
  }
  
  // Check for valid pointer to DeviceFactory
  if ( deviceFactory(__FUNCTION__) ) { 
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " DeviceFactory created at address 0x" 
       << hex << setw(8) << setfill('0') << factory_ << dec
       << " using xml files";
    edm::LogInfo(logCategory_) << ss.str();
  } else {    
    stringstream ss; 
    ss << "[" << __PRETTY_FUNCTION__ << "]" 
       << " NULL pointer to DeviceFactory!"
       << " Unable to connect to database!";
    return; 
  }
  
  try { 
    deviceFactory(__FUNCTION__)->setUsingDb( usingDb_ );
  } catch (...) { 
    handleException( __FUNCTION__, "Attempted to 'setUsingDb'" );
  }

  try { 
    deviceFactory(__FUNCTION__)->createInputFileAccess();
  } catch (...) { 
    handleException( __FUNCTION__, "Attempted to 'createInputFileAccess'" ); 
  }
  
  // Input module.xml file
  if ( inputModuleXml_ == "" ) {
    edm::LogError(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to input 'module.xml' file!";
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	deviceFactory(__FUNCTION__)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { handleException( __FUNCTION__ ); }
      edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " Added input 'module.xml' file: " << inputModuleXml_;
    } else {
      edm::LogError(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " No 'module.xml' file found at " << inputModuleXml_;
      inputModuleXml_ = ""; 
    }
  }
  
  // Input dcuinfo.xml file
  if ( inputDcuInfoXml_ == "" ) {
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(__FUNCTION__)->setTkDcuInfoInputFileName( inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( __FUNCTION__ ); 
      }
      edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " Added 'dcuinfo.xml' file: " << inputDcuInfoXml_;
    } else {
      edm::LogError(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " No 'dcuinfo.xml' file found at " << inputDcuInfoXml_;
      inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( inputFecXml_.empty() ) {
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFecXml_.begin();
    for ( ; iter != inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to input 'fec.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFecXml_.size() == 1 ) {
	      deviceFactory(__FUNCTION__)->setFecInputFileName( *iter ); 
	    } else {
	      deviceFactory(__FUNCTION__)->addFecFileName( *iter ); 
	    }
	  } catch (...) { handleException( __FUNCTION__ ); }
	  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " Added 'fec.xml' file: " << *iter;
	} else {
	  edm::LogError(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " No 'fec.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }
    
  // Input FED xml files
  if ( inputFedXml_.empty() ) {
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL paths to input 'fed.xml' files!";
  } else {
    vector<string>::iterator iter = inputFedXml_.begin();
    for ( ; iter != inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to input 'fed.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFedXml_.size() == 1 ) {
	      deviceFactory(__FUNCTION__)->setFedInputFileName( *iter ); 
	    } else {
	      deviceFactory(__FUNCTION__)->addFedFileName( *iter ); 
	    }
	  } catch (...) { 
	    handleException( __FUNCTION__ ); 
	  }
	  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " Added 'fed.xml' file: " << *iter;
	} else {
	  edm::LogError(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " No 'fed.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }

  // Output module.xml file
  if ( outputModuleXml_ == "" ) { 
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to output 'module.xml' file! Setting to '/tmp/module.xml'...";
    outputModuleXml_ = "/tmp/module.xml"; 
  } else {
    try { 
      FedFecConnectionDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( outputModuleXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'module.xml' file!";
      handleException( __FUNCTION__, info ); 
    }
  }

  // Output dcuinfo.xml file
  if ( outputDcuInfoXml_ == "" ) { 
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to output 'dcuinfo.xml' file! Setting to '/tmp/dcuinfo.xml'...";
    outputModuleXml_ = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( outputDcuInfoXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'dcuinfo.xml' file!";
      handleException( __FUNCTION__, info ); 
    }
  }

  // Output fec.xml file
  if ( outputFecXml_ == "" ) {
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to output 'fec.xml' file! Setting to '/tmp/fec.xml'...";
    outputFecXml_ = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( outputFecXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fec.xml' file!";
      handleException( __FUNCTION__, info ); 
    }
  }

  // Output fed.xml file
  if ( outputFedXml_ == "" ) {
    edm::LogWarning(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]" << " NULL path to output 'fed.xml' file! Setting to '/tmp/fed.xml'...";
    outputFedXml_ = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( outputFedXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fed.xml' file!";
      handleException( __FUNCTION__, info ); 
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

  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
			     << " Creating partition " << partition_.name_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fec_" << partition_.name_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__FUNCTION__)->createPartition( devices,
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
      edm::LogError(logCategory_) << ss.str() << "\n";
      handleException( __FUNCTION__, ss.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/dcuconv_" << partition_.name_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__FUNCTION__)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(logCategory_) << ss.str() << "\n";
      handleException( __FUNCTION__, ss.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fed_" << partition_.name_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__FUNCTION__)->setFed9UDescriptions( feds,
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
      edm::LogError(logCategory_) << ss.str() << "\n";
      handleException( __FUNCTION__, ss.str() );
    }
  }    

  // Create and upload FED connections
  const FedConnections& conns = createFedConnections( fec_cabling );
  if ( !conns.empty() ) {
    FedConnections::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) { 
      try {
	deviceFactory(__FUNCTION__)->addFedChannelConnection( *iconn );
      } catch(...) {
	stringstream ss; 
	ss << "Failed to add FedChannelConnectionDescription!";
	handleException( __FUNCTION__, ss.str() );
      }
    }
    try {
      stringstream ss; 
      ss << "/tmp/module_" << partition_.name_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(__FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__FUNCTION__)->write();
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      
      handleException( __FUNCTION__, ss.str() );
    }
  }

  edm::LogInfo("FedCabling") << "[" << __PRETTY_FUNCTION__ << "]" << " Finished!";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const string& method_name,
				       const string& extra_info ) { //throw (cms::Exception) {

  stringstream ss;
  ss << "[" << __PRETTY_FUNCTION__ << "]";

  try {
    //throw; // rethrow caught exception to be dealt with below
  } 
 
  catch ( const cms::Exception& e ) { 
    ss << " Caught cms::Exception in method "
       << method_name << " with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw e; // rethrow cms::Exception
  }
  
  catch ( const oracle::occi::SQLException& e ) { 
    ss << " Caught oracle::occi::SQLException in method "
       << method_name << " with message: \n" 
       << e.getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }

  catch ( const FecExceptionHandler& e ) {
    ss << " Caught FecExceptionHandler exception in method "
       << method_name << " with message: \n" 
       << const_cast<FecExceptionHandler&>(e).getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }

  catch ( const ICUtils::ICException& e ) {
    ss << " Caught ICUtils::ICException in method "
       << method_name << " with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }

  catch ( const exception& e ) {
    ss << " Caught std::exception in method "
       << method_name << " with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }

  catch (...) {
    ss << " Caught unknown exception in method "
       << method_name << " (No message) \n";
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info << "\n"; }
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  
  // Message
  edm::LogError(logCategory_) << ss.str();

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






  
