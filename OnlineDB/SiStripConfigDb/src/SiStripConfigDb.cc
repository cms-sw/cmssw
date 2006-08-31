// Last commit: $Id: SiStripConfigDb.cc,v 1.15 2006/08/03 07:04:08 bainbrid Exp $
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
  edm::LogInfo(logCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
			     << " Constructing object..."
			     << " (Class instance: " << cntr_ << ")";
  
  partition_.name_ = partition;
  partition_.major_ = major;
  partition_.minor_ = minor;

  // If partition name is not set, attempt to use environmental variable
  if ( partition == "" ) {
    edm::LogWarning(logCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
				  << " Database partition not specified!"
				  << " Attempting to read 'ENV_CMS_TK_PARTITION' environmental variable...";
    
    char* cpath = getenv( "ENV_CMS_TK_PARTITION" );
    if ( cpath == 0 ) { 
      edm::LogError(logCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
				  << " 'ENV_CMS_TK_PARTITION' environmental variable not specified!";
      partition_.name_ = "UNKNOWN";
    } else {
      string confdb(cpath);
      partition_.name_ = confdb;
      edm::LogVerbatim(logCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
				     << " Database partition set using 'ENV_CMS_TK_PARTITION' environmental variable: "
				     << confdb;
    }
  }

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
  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__
			     << " Constructing object..."
			     << " (Class instance: " << cntr_ << ")";
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__
			     << " Destructing object...";
  if ( cntr_ ) { cntr_--; }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::openDbConnection() {
  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__;

  // Establish database connection
  if ( usingDb_ ) { 
    usingDatabase(); 
  } else { 
    usingXmlFiles(); 
  }
  
  // Refresh local cache
  //refreshLocalCaches();

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::closeDbConnection() {
  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__;
  try { 
    if ( factory_ ) { delete factory_; }
  } catch (...) { handleException( __PRETTY_FUNCTION__ ); }
  factory_ = 0; 
}

// -----------------------------------------------------------------------------
//
DeviceFactory* const SiStripConfigDb::deviceFactory( string method_name ) const { 
  if ( !factory_ ) { 
    stringstream ss;
    if ( method_name != "" ) { ss << method_name; }
    else { ss << __PRETTY_FUNCTION__; }
    ss << " Null pointer to DeviceFactory API! \n";
    edm::LogError(logCategory_) << ss.str();
    //throw cms::Exception(logCategory_) << ss.str();
    return 0;
  } else { return factory_; }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {

  // Check on whether not using database
  if ( !usingDb_ ) {
    stringstream ss;
    ss << __PRETTY_FUNCTION__
       << " Attempting to use xml files when configured to use database!";
    edm::LogError(logCategory_) << ss.str();
    //throw cms::Exception(logCategory_) << ss.str();
  }
  
  // Retrieve db connection parameters
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      string confdb = "";
      if ( getenv(CONFDB) != NULL ) { confdb = getenv(CONFDB); } 
      stringstream ss;
      ss << __PRETTY_FUNCTION__
	 << " NULL database connection parameter(s)!" 
	 << " Extracted from .cfg file: " 
	 << user_ << "/" << passwd_ << "@" << path_
	 << " Extracted from CONFDB: " << confdb;
      edm::LogError(logCategory_) << ss.str();
      //throw cms::Exception(logCategory_) << ss.str();
    }
  }
  
  // Create device factory object
  try { 
    factory_ = new DeviceFactory( user_, passwd_, path_ ); 
    deviceFactory(__PRETTY_FUNCTION__)->setUsingDb( usingDb_ ); //@@ necessary?
  } catch (...) { 
    stringstream ss; 
    ss << "Attempting to use database connection parameters '" 
       << user_ << "/" << passwd_ << "@" << path_ 
       << "' and partition '" << partition_.name_ << "'";
    handleException( __PRETTY_FUNCTION__, ss.str() );
  }

  // Database access for FED-FEC connections
  try { 
    deviceFactory(__PRETTY_FUNCTION__)->createInputDBAccess();
  } catch (...) { 
    stringstream ss; 
    ss << "Attempting to use database for FED-FEC connections!";
    handleException( __PRETTY_FUNCTION__, ss.str() );
  }

  // DCU-DetId 
  try { 
    deviceFactory(__PRETTY_FUNCTION__)->addDetIdPartition( partition_.name_ );
    //deviceFactory(__PRETTY_FUNCTION__)->addAllDetId();
  } catch (...) { 
    stringstream ss; 
    ss << "DCU-DetId map!"; 
    handleException( __PRETTY_FUNCTION__, ss.str() );
  }
  
  stringstream ss;
  ss << __PRETTY_FUNCTION__
     << " DeviceFactory created at address 0x" 
     << hex << setw(8) << setfill('0') << factory_ << dec
     << " using database connection parameters '" 
     << user_ << "/" << passwd_ << "@" << path_
     << "' and partition '" << partition_.name_ << "'";
  edm::LogInfo(logCategory_) << ss.str();
  
}  

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingXmlFiles() {

  // Check on whether not using database
  if ( usingDb_ ) {
    stringstream ss;
    ss << __PRETTY_FUNCTION__
       << " Attempting to use database when configured to use xml files! \n";
    edm::LogError(logCategory_) << ss.str();
    //throw cms::Exception(logCategory_) << ss.str();
  }

  // Set "using file"
  try { 
    factory_ = new DeviceFactory(); 
    deviceFactory(__PRETTY_FUNCTION__)->setUsingDb( usingDb_ ); //@@ necessary?
    deviceFactory(__PRETTY_FUNCTION__)->createInputFileAccess(); //@@ necessary?
  } catch (...) { handleException( __PRETTY_FUNCTION__ ); }
  
  // Input module.xml file
  if ( inputModuleXml_ == "" ) {
    edm::LogError(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to input 'module.xml' file!";
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	//deviceFactory(__PRETTY_FUNCTION__)->addFileName( inputModuleXml_ ); //@@ obsolete?
	deviceFactory(__PRETTY_FUNCTION__)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { handleException( __PRETTY_FUNCTION__ ); }
      edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__ << " Added input 'module.xml' file: " << inputModuleXml_;
    } else {
      edm::LogError(logCategory_) << __PRETTY_FUNCTION__ << " No 'module.xml' file found at " << inputModuleXml_;
      inputModuleXml_ = ""; 
      //throw cms::Exception(logCategory_) << __PRETTY_FUNCTION__ << " No 'module.xml' file found at " << inputModuleXml_;
    }
  }
  
  // Input dcuinfo.xml file
  if ( inputDcuInfoXml_ == "" ) {
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(__PRETTY_FUNCTION__)->setTkDcuInfoInputFileName( inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( __PRETTY_FUNCTION__ ); 
      }
      edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__ << " Added 'dcuinfo.xml' file: " << inputDcuInfoXml_;
    } else {
      edm::LogError(logCategory_) << __PRETTY_FUNCTION__ << " No 'dcuinfo.xml' file found at " << inputDcuInfoXml_;
      inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( inputFecXml_.empty() ) {
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFecXml_.begin();
    for ( ; iter != inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to input 'fec.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFecXml_.size() == 1 ) {
	      deviceFactory(__PRETTY_FUNCTION__)->setFecInputFileName( *iter ); 
	    } else {
	      deviceFactory(__PRETTY_FUNCTION__)->addFecFileName( *iter ); 
	    }
	  } catch (...) { handleException( __PRETTY_FUNCTION__ ); }
	  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__ << " Added 'fec.xml' file: " << *iter;
	} else {
	  edm::LogError(logCategory_) << __PRETTY_FUNCTION__ << " No 'fec.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }
    
  // Input FED xml files
  if ( inputFedXml_.empty() ) {
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL paths to input 'fed.xml' files!";
  } else {
    vector<string>::iterator iter = inputFedXml_.begin();
    for ( ; iter != inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to input 'fed.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFedXml_.size() == 1 ) {
	      deviceFactory(__PRETTY_FUNCTION__)->setFedInputFileName( *iter ); 
	    } else {
	      deviceFactory(__PRETTY_FUNCTION__)->addFedFileName( *iter ); 
	    }
	  } catch (...) { 
	    handleException( __PRETTY_FUNCTION__ ); 
	  }
	  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__ << " Added 'fed.xml' file: " << *iter;
	} else {
	  edm::LogError(logCategory_) << __PRETTY_FUNCTION__ << " No 'fed.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }

  // Output module.xml file
  if ( outputModuleXml_ == "" ) { 
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to output 'module.xml' file! Setting to '/tmp/module.xml'...";
    outputModuleXml_ = "/tmp/module.xml"; 
  } else {
    try { 
      FedFecConnectionDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( outputModuleXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'module.xml' file!";
      handleException( __PRETTY_FUNCTION__, info ); 
    }
  }

  // Output dcuinfo.xml file
  if ( outputDcuInfoXml_ == "" ) { 
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to output 'dcuinfo.xml' file! Setting to '/tmp/dcuinfo.xml'...";
    outputModuleXml_ = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( outputDcuInfoXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'dcuinfo.xml' file!";
      handleException( __PRETTY_FUNCTION__, info ); 
    }
  }

  // Output fec.xml file
  if ( outputFecXml_ == "" ) {
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to output 'fec.xml' file! Setting to '/tmp/fec.xml'...";
    outputFecXml_ = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( outputFecXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fec.xml' file!";
      handleException( __PRETTY_FUNCTION__, info ); 
    }
  }

  // Output fed.xml file
  if ( outputFedXml_ == "" ) {
    edm::LogWarning(logCategory_) << __PRETTY_FUNCTION__ << " NULL path to output 'fed.xml' file! Setting to '/tmp/fed.xml'...";
    outputFedXml_ = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( outputFedXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fed.xml' file!";
      handleException( __PRETTY_FUNCTION__, info ); 
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

  getDeviceDescriptions();
  getFedDescriptions();
  getFedConnections();
  getPiaResetDescriptions();
  getDcuConversionFactors();

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createPartition( const string& partition_name,
				       const SiStripFecCabling& fec_cabling ) {
  
  // Set partition name and version
  partition_.name_ = partition_name;
  partition_.major_ = 0;
  partition_.minor_ = 0;

  edm::LogInfo(logCategory_) << __PRETTY_FUNCTION__
			     << " Creating partition " << partition_.name_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fec_" << partition_.name_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__PRETTY_FUNCTION__)->createPartition( devices,
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
      handleException( __PRETTY_FUNCTION__, ss.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/dcuconv_" << partition_.name_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__PRETTY_FUNCTION__)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(logCategory_) << ss.str() << "\n";
      handleException( __PRETTY_FUNCTION__, ss.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fed_" << partition_.name_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__PRETTY_FUNCTION__)->setFed9UDescriptions( feds,
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
      handleException( __PRETTY_FUNCTION__, ss.str() );
    }
  }    

  // Create and upload FED connections
  const FedConnections& conns = createFedConnections( fec_cabling );
  if ( !conns.empty() ) {
    FedConnections::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) { 
      try {
	deviceFactory(__PRETTY_FUNCTION__)->addFedChannelConnection( *iconn );
      } catch(...) {
	stringstream ss; 
	ss << "Failed to add FedChannelConnectionDescription!";
	handleException( __PRETTY_FUNCTION__, ss.str() );
      }
    }
    try {
      stringstream ss; 
      ss << "/tmp/module_" << partition_.name_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(__PRETTY_FUNCTION__);
      factory->setOutputFileName( ss.str() );
      deviceFactory(__PRETTY_FUNCTION__)->write();
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      
      handleException( __PRETTY_FUNCTION__, ss.str() );
    }
  }

  edm::LogInfo("FedCabling") << __PRETTY_FUNCTION__ << " Finished!";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const string& method_name,
				       const string& extra_info ) { //throw (cms::Exception) {
  try {
    //throw; // rethrow caught exception to be dealt with below
  } 
  catch ( const cms::Exception& e ) { 
    //throw e; // rethrow cms::Exception to be caught by framework
  }
  catch ( const oracle::occi::SQLException& e ) { 
    stringstream ss;
    ss << "Caught oracle::occi::SQLException in ["
       << method_name << "] with message: \n" 
       << e.getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  catch ( const FecExceptionHandler& e ) {
    stringstream ss;
    ss << "Caught FecExceptionHandler exception in ["
       << method_name << "] with message: \n" 
       << const_cast<FecExceptionHandler&>(e).getMessage(); //@@ Fred?
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  catch ( const ICUtils::ICException& e ) {
    stringstream ss;
    ss << "Caught ICUtils::ICException in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  catch ( const exception& e ) {
    stringstream ss;
    ss << "Caught std::exception in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
  catch (...) {
    stringstream ss;
    ss << "Caught unknown exception in ["
       << method_name << "]";
    if ( extra_info != "" ) { ss << "\n" << "Additional info: " << extra_info; }
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }
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







//@@ DYNAMIC CASTING

//     FecFactory::deleteVector( descriptions ); // clears vector
//     deviceVector::iterator idevice;
//     for ( idevice = all_descriptions.begin() ; idevice != all_descriptions.end() ; idevice++ ) {
//       if ( device_type == APV ) {
// 	apvDescription* apv = dynamic_cast<apvDescription*>( *idevice );
// 	if ( apv ) { descriptions.push_back( apv ); }
//       } else if ( device_type == PLL ) {
// 	pllDescription* pll = dynamic_cast<pllDescription*>( *idevice );
// 	if ( pll ) { descriptions.push_back( pll ); }
//       } else if ( device_type == MUX ) {
// 	muxDescription* mux = dynamic_cast<muxDescription*>( *idevice );
// 	if ( mux ) { descriptions.push_back( mux ); }
//       } else if ( device_type == DCU ) {
// 	dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idevice );
// 	if ( dcu ) { descriptions.push_back( dcu ); }
//       } else if ( device_type == LLD ) {
// 	laserdriverDescription* lld = dynamic_cast<laserdriverDescription*>( *idevice );
// 	if ( lld ) { descriptions.push_back( lld ); }
//       }	else { 
// 	edm::LogError(logCategory_) << "[SiStripConfigDb::getDeviceDescriptions]"
// 				  << " Unknown device!!";
//       }
//     }










// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::feDevices( enumDeviceType device_type,
// 				 deviceVector& devices ) {
  
//   string device_name("");
//   if      ( device_type == APV25 )       { device_name = "APV25"; }
//   else if ( device_type == PLL )         { device_name = "PLL"; }
//   else if ( device_type == APVMUX )      { device_name = "APVMUX"; }
//   else if ( device_type == DCU )         { device_name = "DCU"; }
//   else if ( device_type == LASERDRIVER ) { device_name = "LASERDRIVER"; }
//   else { device_name = "UNKNOWN"; }
  
//   pair<int,int> version = partitionVersion( partitionName() );
  
//   try {
//     if ( !allDevices_.empty() ) { FecFactory::deleteVector( allDevices_ ); }
//     deviceFactory(__PRETTY_FUNCTION__)->getFecDeviceDescriptions( partitionName(), 
// 					allDevices_, 
// 					version.first, 
// 					version.second );
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<"Caught unknown exception : ";
//   }
  
//   if ( allDevices_.empty() ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::getApvDevices]"
// 			      << " No FE devices found for partition " << partitionName()
// 			      << " and version " << version.first << "." << version.second;
//   } else {
//     LogDebug(logCategory_) << "[SiStripConfigDb::feDevices]"
// 			 << " Found " << allDevices_.size() << " FE devices "
// 			 << "for partition " << partitionName() 
// 			 << " and version " << version.first << "." << version.second;
//   }
  
//   deviceSummary(); 
  
// }

// // -----------------------------------------------------------------------------
// //
// void SiStripConfigDb::deviceSummary() { 
//   LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary]";
  
//   pair<int,int> version = partitionVersion( partitionName() );
  
//   if ( allDevices_.empty() ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::deviceSummary]"
// 			      << " No devices found for partition " << partitionName()
// 			      << " and version " << version.first << "." << version.second;
//   } else {
//     unsigned int apv_cntr, pll_cntr, mux_cntr, dcu_cntr, las_cntr, doh_cntr, misc_cntr;
//     apv_cntr = pll_cntr = mux_cntr = dcu_cntr = las_cntr = doh_cntr = misc_cntr = 0;
//     deviceVector::iterator idevice;
//     for ( idevice = allDevices_.begin(); idevice != allDevices_.end(); idevice++ ) {
//       if      ( (*idevice)->getDeviceType() == APV25 )       { apv_cntr++; }
//       else if ( (*idevice)->getDeviceType() == PLL )         { pll_cntr++; }
//       else if ( (*idevice)->getDeviceType() == APVMUX )      { mux_cntr++; }
//       else if ( (*idevice)->getDeviceType() == DCU )         { dcu_cntr++; }
//       else if ( (*idevice)->getDeviceType() == LASERDRIVER ) { las_cntr++; }
//       else if ( (*idevice)->getDeviceType() == DOH )         { doh_cntr++; }
//       else { misc_cntr++; }
//     }
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << apv_cntr << " APV devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << pll_cntr << " PLL devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << mux_cntr << " APVMUX devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << dcu_cntr << " DCU devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << las_cntr << " LASERDRIVER devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << doh_cntr << " DOH devices";
//     LogDebug(logCategory_) << "[SiStripConfigDb::deviceSummary] Found " << misc_cntr << " other MISCELLANEOUS devices";
//     // FecFactory::display( allDevices_ ); 
//   }
  
// }

// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::apvDescriptions( vector<apvDescription*>& apv_descriptions ) {

//   pair<int,int> version = partitionVersion( partitionName() );
  
//   deviceVector apv_devices;
//   try {
//     feDevices( APV25, apv_devices );
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught unknown exception : ";
//   }
  
//   if ( !apv_devices.empty() ) {
//     deviceVector::iterator idevice;
//     for ( idevice = apv_devices.begin() ; idevice != apv_devices.end() ; idevice++ ) {
//       apvDescription* apv = dynamic_cast<apvDescription*>( *idevice );
//       if ( apv ) { 
// 	apv_descriptions.push_back( apv ); 
// 	DeviceAddress addr = hwAddresses( *apv );
// 	LogDebug(logCategory_) << "[SiStripConfigDb::apvDescriptions] APV25 found at "
// 			     << " FEC crate: " << addr.fecCrate
// 			     << ", FEC slot: " << addr.fecSlot
// 			     << ", FEC ring: " << addr.fecRing
// 			     << ", CCU addr: " << addr.ccuAddr
// 			     << ", CCU chan: " << addr.ccuChan
// 			     << ", I2C Addr: " << addr.i2cAddr;
//       }
//     }
//   }
  
//   if ( apv_descriptions.empty() ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      << " No APV descriptions found for partition " 
// 			      << partitionName();
//   } else {
//     LogDebug(logCategory_) << "[SiStripConfigDb::apvDescriptions] "
// 			 << "Found " << apv_descriptions.size() << " APV descriptions "
// 			 << "for partition " << partitionName() 
// 			 << " and version " << version.first << "." << version.second;
//   }
  
// }

// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::dcuDescriptions( vector<dcuDescription*>& dcu_descriptions ) {
  
//   pair<int,int> version = partitionVersion( partitionName() );
  
//   deviceVector dcu_devices;
//   try {
//     deviceFactory(__PRETTY_FUNCTION__)->getDcuDescriptions( partitionName(), dcuDevices_ );
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught unknown exception : ";
//   }
  
//   if ( !dcu_devices.empty() ) {
//     deviceVector::iterator idevice;
//     for ( idevice = dcu_devices.begin() ; idevice != dcu_devices.end() ; idevice++ ) {
//       dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idevice );
//       if ( dcu ) { 
// 	dcu_descriptions.push_back( dcu ); 
// 	DeviceAddress addr = hwAddresses( *dcu );
// 	LogDebug(logCategory_) << "[SiStripConfigDb::dcuDescriptions] DCU found at "
// 			     << " FEC crate: " << addr.fecCrate
// 			     << ", FEC slot: " << addr.fecSlot
// 			     << ", FEC ring: " << addr.fecRing
// 			     << ", CCU addr: " << addr.ccuAddr
// 			     << ", CCU chan: " << addr.ccuChan
// 			     << ", I2C Addr: " << addr.i2cAddr;
//       }
//     }
//   }
  
//   if ( dcu_descriptions.empty() ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      << " No DCU descriptions found for partition " 
// 			      << partitionName();
//   } else {
//     LogDebug(logCategory_) << "[SiStripConfigDb::dcuDescriptions] "
// 			 << "Found " << dcu_descriptions.size() << " DCU descriptions "
// 			 << "for partition " << partitionName() 
// 			 << " and version " << version.first << "." << version.second;
//   }
  
// }


// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::aohDescriptions( vector<laserdriverDescription*>& aoh_descriptions ) {
//   LogDebug(logCategory_) << "[SiStripConfigDb::aohDescriptions] " 
// 		 << "Retrieving AOH descriptions from database...";

//   try {

//     // Retrieve partition name
//     string partitionName() = partitionName();

//     // Retrieve major/minor versions for given partition
//     pair<int,int> version;
//     partitionVersion( partitionName(), version );

//     // Retrieve all AOH devices 
//     deviceVector aoh_devices;
//     feDevices( LASERDRIVER, aoh_devices );

//     // Retrieve LASERDRIVER descriptions
//     if ( !aoh_devices.empty() ) {
//       deviceVector::iterator idevice;
//       for ( idevice = aoh_devices.begin() ; idevice != aoh_devices.end() ; idevice++ ) {
// 	laserdriverDescription* aoh = dynamic_cast<laserdriverDescription*>( *idevice );
// 	if ( aoh ) { aoh_descriptions.push_back( aoh ); }
// 	LogDebug(logCategory_) << "[SiStripConfigDb::aohDescriptions] "
// 		      << "LASERDRIVER (AOH) found at: "
// 		      << "FEC slot: " << getFecKey(     aoh->getKey() ) << ", "
// 		      << "FEC ring: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "CCU addr: " << getCcuKey(     aoh->getKey() ) << ", "
// 		      << "CCU chan: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "I2C Addr: " << getAddressKey( aoh->getKey() );
//       }
//     }

//     LogDebug(logCategory_) << "[SiStripConfigDb::aohDescriptions] "
// 		  << "Found " << aoh_descriptions.size() << " AOH descriptions "
// 		  << "for the partition name " << partitionName() 
// 		  << " and version " << version.first << "." << version.second
// 		 ;

//     // Check if any AOH descriptions are found
//     if ( aoh_descriptions.empty() ) {
//       edm::LogError(logCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	   << "No LASERDRIVER (AOH) descriptions found for the partition name " << partitionName()
// 	   << endl ;
//     }

//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(logCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught FecExceptionHandler exception : Problems accessing database " 
// 	 << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(logCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught exception : Problems accessing database " 
// 	 << e.what();
//   }
//   catch (...) {
//     edm::LogError(logCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught unknown exception : Problems accessing database " 
// 	;
//   }
  
// }



// // -----------------------------------------------------------------------------
// // Open connection to database
// void SiStripConfigDb::feDevices() {
//   LogDebug(logCategory_) << "[SiStripConfigDb::feDevices] "
// 		<< " ";
  
// #ifdef USING_DATABASE

//   pair<int,int> version = partitionVersion();
//   SimpleConfigurable<string> partitionName()("nil","SiStripConfigDb:PartitionName");
//   try {
//     deviceVector devices = deviceFactory(__PRETTY_FUNCTION__)->getFecDeviceDescriptions( partitionName(), version.first, version.second );
//     if ( devices.empty() ) {
//       edm::LogError(logCategory_) << "[SiStripConfigDb::feDevices] "
// 	   << "ERROR : No FE devices exist for the partition name " << partitionName()
// 	   << endl ;
//     }
//     if ( devices.size() ) {
//       LogDebug(logCategory_) << "[SiStripConfigDb::feDevices] "
// 	   << "Found " << devices.size() << " FE device descriptions "
// 	   << "for partition name " << partitionName();
//       FecFactory::display( devices ); // display devices
//       // count number of APV devices
//       unsigned int apv_cntr = 0;
//       deviceVector::iterator device;
//       for ( device = devices.begin() ; device != devices.end() ; device++ ) {
// 	deviceDescription* description = *device;
// 	if ( description->getDeviceType() == APV25 ) { 
// 	  apv_cntr++; 
// 	  description->display();
// 	}
//       }
//       LogDebug(logCategory_) << "Found " << apv_cntr << " APV ";
//     }
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(logCategory_) << "[SiStripConfigDb::feDevices] "
// 	 <<"ERROR : problems retrieving FE devices " 
// 	 << e.getMessage();
//   }

// #else
//   edm::LogError(logCategory_) << warning("[SiStripConfigDb::openConnection] ")
//        << "USING_DATABASE not defined! => not using database";        
// #endif // USING_DATABASE

// }


//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set all the devices with modification in the PLL
//       unsigned int major, minor ;
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, true) ;
//       //                                                                                  ^ major version, false = minor version
    
//       LogDebug(logCategory_) << "-------------------------- Version ---------------------" << endl ;
//       LogDebug(logCategory_) << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug(logCategory_) << "--------------------------------------------------------" << endl ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set only the PLL devices
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, false) ;
//       //                                                                                  ^ minor version, true = major version
    
//       LogDebug(logCategory_) << "-------------------------- Version ---------------------" << endl ;
//       LogDebug(logCategory_) << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug(logCategory_) << "--------------------------------------------------------" << endl ;
   
//       // ---------------------------------------------------------------------------------------------------- 
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;

//       // ---------------------------------------------------------------------------------------------------- 
//       // delete the vector of deviceDescriptions
//       FecFactory::deleteVector (devices) ;
//     }
//     else {
//       edm::LogError(logCategory_ ) << "No devices found in the database" << endl ;
//     }
