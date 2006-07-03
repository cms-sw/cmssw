// Last commit: $Id: SiStripConfigDb.cc,v 1.10 2006/06/30 06:57:52 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/SiStripConfigDb.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const string SiStripConfigDb::errorCategory_ = "SiStrip|ConfigDb";
uint32_t SiStripConfigDb::cntr_ = 0;

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb( string user, 
				  string passwd, 
				  string path,
				  string partition ) : 
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
  edm::LogInfo(errorCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
			       << " Constructing object..."
			       << " (Class instance: " << cntr_ << ")";
  partition_.name_ = partition;
  partition_.major_ = 0;
  partition_.minor_ = 0;
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
  edm::LogInfo(errorCategory_) << "[SiStripConfigDb::SiStripConfigDb]"
			       << " Constructing object..."
			       << " (Class instance: " << cntr_ << ")";
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  edm::LogInfo(errorCategory_) << "[SiStripConfigDb::~SiStripConfigDb]"
			       << " Destructing object...";
  if ( cntr_ ) { cntr_--; }
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::openDbConnection() {
  edm::LogInfo(errorCategory_) << "[SiStripConfigDb::openDbConnection]";

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
  edm::LogInfo(errorCategory_) << "[SiStripConfigDb::closeDbConnection]";
  try { 
    if ( factory_ ) { delete factory_; }
  } catch (...) { handleException( "SiStripConfigDb::closeDbConnection" ); }
  factory_ = 0; 
}

// -----------------------------------------------------------------------------
//
DeviceFactory* const SiStripConfigDb::deviceFactory( string method_name ) const { 
  if ( !factory_ ) { 
    stringstream ss;
    if ( method_name != "" ) { ss << "[" << method_name << "]"; }
    else { ss << "[SiStripConfigDb::deviceFactory]"; }
    ss << " Null pointer to DeviceFactory API! \n";
    edm::LogError(errorCategory_) << ss.str();
    throw cms::Exception(errorCategory_) << ss.str();
    return 0;
  } else { return factory_; }
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingDatabase() {
  string method = "SiStripConfigDb::usingDatabase";

  // Check on whether not using database
  if ( !usingDb_ ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " Attempting to use xml files when configured to use database!";
    edm::LogError(errorCategory_) << ss.str();
    throw cms::Exception(errorCategory_) << ss.str();
  }
  
  // Retrieve db connection parameters
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      string confdb = "";
      if ( getenv(CONFDB) != NULL ) { confdb = getenv(CONFDB); } 
      stringstream ss;
      ss << "["<<method<<"]"
	 << " NULL database connection parameter(s)!" 
	 << " Extracted from .cfg file: " 
	 << user_ << "/" << passwd_ << "@" << path_
	 << " Extracted from CONFDB: " << confdb;
      edm::LogError(errorCategory_) << ss.str();
      throw cms::Exception(errorCategory_) << ss.str();
    }
  }
  
  // Create device factory object
  try { 
    factory_ = new DeviceFactory( user_, passwd_, path_ ); 
    deviceFactory(method)->setUsingDb( usingDb_ ); //@@ necessary?
  } catch (...) { 
    stringstream ss; 
    ss << "Attempting to use database connection parameters '" 
       << user_ << "/" << passwd_ << "@" << path_ 
       << "' and partition '" << partition_.name_ << "'";
    handleException( method, ss.str() );
  }

  // Database access for FED-FEC connections
  try { 
    deviceFactory(method)->createInputDBAccess();
  } catch (...) { 
    stringstream ss; 
    ss << "Attempting to use database for FED-FEC connections!";
    handleException( method, ss.str() );
  }

//   // DCU-DetId 
//   try { 
//     deviceFactory(method)->addAllDetId();
//   } catch (...) { 
//     stringstream ss; 
//     ss << "DCU-DetId map!"; 
//     handleException( method, ss.str() );
//   }
  
  stringstream ss;
  ss << "["<<method<<"]"
     << " DeviceFactory created at address 0x" 
     << hex << setw(8) << setfill('0') << factory_ << dec
     << " using database connection parameters '" 
     << user_ << "/" << passwd_ << "@" << path_
     << "' and partition '" << partition_.name_ << "'";
  edm::LogInfo(errorCategory_) << ss.str();
  
}  

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::usingXmlFiles() {
  string method = "SiStripConfigDb::usingXmlFiles";

  // Check on whether not using database
  if ( usingDb_ ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " Attempting to use database when configured to use xml files! \n";
    edm::LogError(errorCategory_) << ss.str();
    throw cms::Exception(errorCategory_) << ss.str();
  }

  // Create DeviceFactory object
  try { 
    factory_ = new DeviceFactory(); 
    deviceFactory(method)->setUsingDb( usingDb_ ); //@@ necessary?
  } catch (...) { handleException( method ); }
  
  // Input module.xml file
  try {
    deviceFactory(method)->createInputFileAccess(); //@@ necessary?
  } catch (...) { handleException( method ); }
  if ( inputModuleXml_ == "" ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to input 'module.xml' file!";
    edm::LogError(errorCategory_) << ss.str();
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	//deviceFactory(method)->addFileName( inputModuleXml_ ); //@@ obsolete?
	deviceFactory(method)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { handleException( method ); }
      edm::LogInfo(errorCategory_) 
	<< "["<<method<<"]"
	<< " Added input 'module.xml' file found at " << inputModuleXml_;
    } else {
      stringstream ss; 
      ss << "["<<method<<"]"
	 << " Missing input file! No 'module.xml' file found at " 
	 << inputModuleXml_;
      edm::LogError(errorCategory_) <<  ss.str();
      inputModuleXml_ = ""; 
      throw cms::Exception(errorCategory_) << ss.str();
    }
  }
  
  // Input dcuinfo.xml file
  if ( inputDcuInfoXml_ == "" ) {
    edm::LogError(errorCategory_) << "["<<method<<"]"
				  << " Null path to input 'dcuinfo.xml' file!";
  } else { 
    if ( checkFileExists( inputDcuInfoXml_ ) ) { 
      try { 
	deviceFactory(method)->setTkDcuInfoInputFileName( inputDcuInfoXml_ ); 
      } catch (...) { 
	handleException( method ); 
      }
      edm::LogInfo(errorCategory_) << "["<<method<<"]"
				   << " Added 'dcuinfo.xml' file: " << inputDcuInfoXml_;
    } else {
      edm::LogError(errorCategory_) << "["<<method<<"]"
				    << " No 'dcuinfo.xml' file found at " << inputDcuInfoXml_;
      inputDcuInfoXml_ = ""; 
    } 
  }

  // Input FEC xml files
  if ( inputFecXml_.empty() ) {
    edm::LogError(errorCategory_) << "["<<method<<"]"
				  << " No paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFecXml_.begin();
    for ( ; iter != inputFecXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogError(errorCategory_) << "["<<method<<"]"
				      << " Null path to input 'fec.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFecXml_.size() == 1 ) {
	      deviceFactory(method)->setFecInputFileName( *iter ); 
	    } else {
	      deviceFactory(method)->addFecFileName( *iter ); 
	    }
	  } catch (...) { handleException( method ); }
	  edm::LogInfo(errorCategory_) << "["<<method<<"]"
				       << " Added 'fec.xml' file: " << *iter;
	} else {
	  edm::LogError(errorCategory_) << "["<<method<<"]"
					<< " No 'fec.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }
    
  // Input FED xml files
  if ( inputFedXml_.empty() ) {
    edm::LogError(errorCategory_) << "["<<method<<"]"
				  << " No paths to input 'fec.xml' files!";
  } else {
    vector<string>::iterator iter = inputFedXml_.begin();
    for ( ; iter != inputFedXml_.end(); iter++ ) {
      if ( *iter == "" ) {
	edm::LogError(errorCategory_) << "["<<method<<"]"
				      << " Null path to input 'fed.xml' file!";
      } else {
	if ( checkFileExists( *iter ) ) { 
	  try { 
	    if ( inputFecXml_.size() == 1 ) {
	      deviceFactory(method)->setFedInputFileName( *iter ); 
	    } else {
	      deviceFactory(method)->addFedFileName( *iter ); 
	    }
	  } catch (...) { 
	    handleException( method ); 
	  }
	  edm::LogInfo(errorCategory_) << "["<<method<<"]"
				       << " Added 'fed.xml' file: " << *iter;
	} else {
	  edm::LogError(errorCategory_) << "["<<method<<"]"
					<< " No 'fed.xml' file found at " << *iter;
	  *iter = ""; 
	} 
      }
    }
  }

  // Input module.xml file
  if ( inputModuleXml_ == "" ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to input 'module.xml' file!";
    edm::LogError(errorCategory_) << ss.str();
  } else {
    if ( checkFileExists( inputModuleXml_ ) ) { 
      try { 
	deviceFactory(method)->createInputFileAccess(); //@@ necessary?
	//deviceFactory(method)->addFileName( inputModuleXml_ ); //@@ obsolete?
	deviceFactory(method)->setFedFecConnectionInputFileName( inputModuleXml_ ); 
      } catch (...) { handleException( method ); }
      edm::LogInfo(errorCategory_) 
	<< "["<<method<<"]"
	<< " Added input 'module.xml' file found at " << inputModuleXml_;
    } else {
      stringstream ss; 
      ss << "["<<method<<"]"
	 << " Missing input file! No 'module.xml' file found at " 
	 << inputModuleXml_;
      edm::LogError(errorCategory_) <<  ss.str();
      inputModuleXml_ = ""; 
      throw cms::Exception(errorCategory_) << ss.str();
    }
  }

  // Output module.xml file
  if ( outputModuleXml_ == "" ) { 
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to output 'module.xml' file!"
       << " Setting to '/tmp/module.xml'...";
    edm::LogWarning(errorCategory_) << ss.str();
    outputModuleXml_ = "/tmp/module.xml"; 
  } else {
    try { 
      FedFecConnectionDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( outputModuleXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'module.xml' file!";
      handleException( method, info ); 
    }
  }

  // Output dcuinfo.xml file
  if ( outputDcuInfoXml_ == "" ) { 
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to output 'dcuinfo.xml' file!"
       << " Setting to '/tmp/dcuinfo.xml'...";
    edm::LogWarning(errorCategory_) << ss.str();
    outputModuleXml_ = "/tmp/dcuinfo.xml"; 
  } else {
    try { 
      TkDcuInfoFactory* factory = deviceFactory(method);
      factory->setOutputFileName( outputDcuInfoXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'dcuinfo.xml' file!";
      handleException( method, info ); 
    }
  }

  // Output fec.xml file
  if ( outputFecXml_ == "" ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to output 'fec.xml' file!"
       << " Setting to '/tmp/fec.xml'...";
    edm::LogWarning(errorCategory_) << ss.str();
    outputFecXml_ = "/tmp/fec.xml";
  } else {
    try { 
      FecDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( outputFecXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fec.xml' file!";
      handleException( method, info ); 
    }
  }

  // Output fed.xml file
  if ( outputFedXml_ == "" ) {
    stringstream ss;
    ss << "["<<method<<"]"
       << " NULL path to output 'fed.xml' file!"
       << " Setting to '/tmp/fed.xml'...";
    edm::LogWarning(errorCategory_) << ss.str();
    outputFedXml_ = "/tmp/fed.xml";
  } else {
    try { 
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( outputFedXml_ ); 
    } catch (...) { 
      string info = "Problems setting output 'fed.xml' file!";
      handleException( method, info ); 
    }
  }

}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::refreshLocalCaches() {
  //string method = "[SiStripConfigDb::refreshLocalCaches]";
  
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
  string method = "SiStripConfigDb::createPartition";
  
  // Set partition name and version
  partition_.name_ = partition_name;
  partition_.major_ = 0;
  partition_.minor_ = 0;

  edm::LogInfo(errorCategory_) << "["<<method<<"]"
			       << " Creating partition " << partition_.name_;

  // Create new partition based on device and PIA reset descriptions
  const DeviceDescriptions& devices = createDeviceDescriptions( fec_cabling );
  const PiaResetDescriptions& resets = createPiaResetDescriptions( fec_cabling );
  if ( !devices.empty() && !resets.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fec_" << partition_.name_ << ".xml";
      FecDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( ss.str() );
      deviceFactory(method)->createPartition( devices,
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
      edm::LogError(errorCategory_) << ss.str() << "\n";
      handleException( method, ss.str() );
    } 
  }
  
  // Create and upload DCU conversion factors
  const DcuConversionFactors& dcu_convs = createDcuConversionFactors( fec_cabling );
  if ( !dcu_convs.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/dcuconv_" << partition_.name_ << ".xml";
      TkDcuConversionFactory* factory = deviceFactory(method);
      factory->setOutputFileName( ss.str() );
      deviceFactory(method)->setTkDcuConversionFactors( dcu_convs );
    } catch (...) { 
      stringstream ss; 
      ss << "Failed to create and upload DCU conversion factors"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      edm::LogError(errorCategory_) << ss.str() << "\n";
      handleException( method, ss.str() );
    }
  }
  
  // Create and upload FED descriptions
  const FedDescriptions& feds = createFedDescriptions( fec_cabling );
  if ( !feds.empty() ) {
    try {
      stringstream ss; 
      ss << "/tmp/fed_" << partition_.name_ << ".xml";
      Fed9U::Fed9UDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( ss.str() );
      deviceFactory(method)->setFed9UDescriptions( feds,
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
      edm::LogError(errorCategory_) << ss.str() << "\n";
      handleException( method, ss.str() );
    }
  }    

  // Create and upload FED connections
  const FedConnections& conns = createFedConnections( fec_cabling );
  if ( !conns.empty() ) {
    FedConnections::const_iterator iconn = conns.begin();
    for ( ; iconn != conns.end(); iconn++ ) { 
      try {
	deviceFactory(method)->addFedChannelConnection( *iconn );
      } catch(...) {
	stringstream ss; 
	ss << "Failed to add FedChannelConnectionDescription!";
	handleException( method, ss.str() );
      }
    }
    try {
      stringstream ss; 
      ss << "/tmp/module_" << partition_.name_ << ".xml";
      FedFecConnectionDeviceFactory* factory = deviceFactory(method);
      factory->setOutputFileName( ss.str() );
      deviceFactory(method)->write();
    } catch(...) {
      stringstream ss; 
      ss << "Failed to create and upload FedChannelConnectionDescriptions"
	 << " to partition with name "
	 << partition_.name_ << " and version " 
	 << partition_.major_ << "." << partition_.minor_;
      
      handleException( method, ss.str() );
    }
  }

  edm::LogInfo("FedCabling") << "["<<method<<"] Finished!";
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::handleException( const string& method_name,
				       const string& extra_info ) throw (cms::Exception) {
  try {
    throw; // rethrow caught exception to be dealt with below
  } 
  catch ( const cms::Exception& e ) { 
    throw e; // rethrow cms::Exception to be caught by framework
  }
  catch ( const oracle::occi::SQLException& e ) { 
    stringstream ss;
    ss << "Caught oracle::occi::SQLException in ["
       << method_name << "] with message: \n" 
       << e.getMessage();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(errorCategory_) << ss.str() << "\n";
    throw cms::Exception(errorCategory_) << ss.str() << "\n";
  }
  catch ( const FecExceptionHandler& e ) {
    stringstream ss;
    ss << "Caught FecExceptionHandler exception in ["
       << method_name << "] with message: \n" 
       << const_cast<FecExceptionHandler&>(e).getMessage(); //@@ Fred?
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(errorCategory_) << ss.str() << "\n";
    throw cms::Exception(errorCategory_) << ss.str() << "\n";
  }
  catch ( const ICUtils::ICException& e ) {
    stringstream ss;
    ss << "Caught ICUtils::ICException in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(errorCategory_) << ss.str() << "\n";
    throw cms::Exception(errorCategory_) << ss.str() << "\n";
  }
  catch ( const exception& e ) {
    stringstream ss;
    ss << "Caught std::exception in ["
       << method_name << "] with message: \n" 
       << e.what();
    if ( extra_info != "" ) { ss << "Additional info: " << extra_info; }
    edm::LogError(errorCategory_) << ss.str() << "\n";
    throw cms::Exception(errorCategory_) << ss.str() << "\n";
  }
  catch (...) {
    stringstream ss;
    ss << "Caught unknown exception in ["
       << method_name << "]";
    if ( extra_info != "" ) { ss << "\n" << "Additional info: " << extra_info; }
    edm::LogError(errorCategory_) << ss.str() << "\n";
    throw cms::Exception(errorCategory_) << ss.str() << "\n";
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
// 	edm::LogError(errorCategory_) << "[SiStripConfigDb::getDeviceDescriptions]"
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
//     deviceFactory(method)->getFecDeviceDescriptions( partitionName(), 
// 					allDevices_, 
// 					version.first, 
// 					version.second );
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::feDevices]"
// 			      <<"Caught unknown exception : ";
//   }
  
//   if ( allDevices_.empty() ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::getApvDevices]"
// 			      << " No FE devices found for partition " << partitionName()
// 			      << " and version " << version.first << "." << version.second;
//   } else {
//     LogDebug(errorCategory_) << "[SiStripConfigDb::feDevices]"
// 			 << " Found " << allDevices_.size() << " FE devices "
// 			 << "for partition " << partitionName() 
// 			 << " and version " << version.first << "." << version.second;
//   }
  
//   deviceSummary(); 
  
// }

// // -----------------------------------------------------------------------------
// //
// void SiStripConfigDb::deviceSummary() { 
//   LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary]";
  
//   pair<int,int> version = partitionVersion( partitionName() );
  
//   if ( allDevices_.empty() ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::deviceSummary]"
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
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << apv_cntr << " APV devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << pll_cntr << " PLL devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << mux_cntr << " APVMUX devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << dcu_cntr << " DCU devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << las_cntr << " LASERDRIVER devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << doh_cntr << " DOH devices";
//     LogDebug(errorCategory_) << "[SiStripConfigDb::deviceSummary] Found " << misc_cntr << " other MISCELLANEOUS devices";
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
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      <<" Caught unknown exception : ";
//   }
  
//   if ( !apv_devices.empty() ) {
//     deviceVector::iterator idevice;
//     for ( idevice = apv_devices.begin() ; idevice != apv_devices.end() ; idevice++ ) {
//       apvDescription* apv = dynamic_cast<apvDescription*>( *idevice );
//       if ( apv ) { 
// 	apv_descriptions.push_back( apv ); 
// 	DeviceAddress addr = hwAddresses( *apv );
// 	LogDebug(errorCategory_) << "[SiStripConfigDb::apvDescriptions] APV25 found at "
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
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::apvDescriptions]"
// 			      << " No APV descriptions found for partition " 
// 			      << partitionName();
//   } else {
//     LogDebug(errorCategory_) << "[SiStripConfigDb::apvDescriptions] "
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
//     deviceFactory(method)->getDcuDescriptions( partitionName(), dcuDevices_ );
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught FecExceptionHandler exception : " 
// 			      << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught exception : " << e.what();
//   }
//   catch (...) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      <<" Caught unknown exception : ";
//   }
  
//   if ( !dcu_devices.empty() ) {
//     deviceVector::iterator idevice;
//     for ( idevice = dcu_devices.begin() ; idevice != dcu_devices.end() ; idevice++ ) {
//       dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idevice );
//       if ( dcu ) { 
// 	dcu_descriptions.push_back( dcu ); 
// 	DeviceAddress addr = hwAddresses( *dcu );
// 	LogDebug(errorCategory_) << "[SiStripConfigDb::dcuDescriptions] DCU found at "
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
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::dcuDescriptions]"
// 			      << " No DCU descriptions found for partition " 
// 			      << partitionName();
//   } else {
//     LogDebug(errorCategory_) << "[SiStripConfigDb::dcuDescriptions] "
// 			 << "Found " << dcu_descriptions.size() << " DCU descriptions "
// 			 << "for partition " << partitionName() 
// 			 << " and version " << version.first << "." << version.second;
//   }
  
// }


// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::aohDescriptions( vector<laserdriverDescription*>& aoh_descriptions ) {
//   LogDebug(errorCategory_) << "[SiStripConfigDb::aohDescriptions] " 
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
// 	LogDebug(errorCategory_) << "[SiStripConfigDb::aohDescriptions] "
// 		      << "LASERDRIVER (AOH) found at: "
// 		      << "FEC slot: " << getFecKey(     aoh->getKey() ) << ", "
// 		      << "FEC ring: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "CCU addr: " << getCcuKey(     aoh->getKey() ) << ", "
// 		      << "CCU chan: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "I2C Addr: " << getAddressKey( aoh->getKey() );
//       }
//     }

//     LogDebug(errorCategory_) << "[SiStripConfigDb::aohDescriptions] "
// 		  << "Found " << aoh_descriptions.size() << " AOH descriptions "
// 		  << "for the partition name " << partitionName() 
// 		  << " and version " << version.first << "." << version.second
// 		 ;

//     // Check if any AOH descriptions are found
//     if ( aoh_descriptions.empty() ) {
//       edm::LogError(errorCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	   << "No LASERDRIVER (AOH) descriptions found for the partition name " << partitionName()
// 	   << endl ;
//     }

//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(errorCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught FecExceptionHandler exception : Problems accessing database " 
// 	 << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError(errorCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught exception : Problems accessing database " 
// 	 << e.what();
//   }
//   catch (...) {
//     edm::LogError(errorCategory_) << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught unknown exception : Problems accessing database " 
// 	;
//   }
  
// }



// // -----------------------------------------------------------------------------
// // Open connection to database
// void SiStripConfigDb::feDevices() {
//   LogDebug(errorCategory_) << "[SiStripConfigDb::feDevices] "
// 		<< " ";
  
// #ifdef USING_DATABASE

//   pair<int,int> version = partitionVersion();
//   SimpleConfigurable<string> partitionName()("nil","SiStripConfigDb:PartitionName");
//   try {
//     deviceVector devices = deviceFactory(method)->getFecDeviceDescriptions( partitionName(), version.first, version.second );
//     if ( devices.empty() ) {
//       edm::LogError(errorCategory_) << "[SiStripConfigDb::feDevices] "
// 	   << "ERROR : No FE devices exist for the partition name " << partitionName()
// 	   << endl ;
//     }
//     if ( devices.size() ) {
//       LogDebug(errorCategory_) << "[SiStripConfigDb::feDevices] "
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
//       LogDebug(errorCategory_) << "Found " << apv_cntr << " APV ";
//     }
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError(errorCategory_) << "[SiStripConfigDb::feDevices] "
// 	 <<"ERROR : problems retrieving FE devices " 
// 	 << e.getMessage();
//   }

// #else
//   edm::LogError(errorCategory_) << warning("[SiStripConfigDb::openConnection] ")
//        << "USING_DATABASE not defined! => not using database";        
// #endif // USING_DATABASE

// }


//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set all the devices with modification in the PLL
//       unsigned int major, minor ;
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, true) ;
//       //                                                                                  ^ major version, false = minor version
    
//       LogDebug(errorCategory_) << "-------------------------- Version ---------------------" << endl ;
//       LogDebug(errorCategory_) << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug(errorCategory_) << "--------------------------------------------------------" << endl ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set only the PLL devices
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, false) ;
//       //                                                                                  ^ minor version, true = major version
    
//       LogDebug(errorCategory_) << "-------------------------- Version ---------------------" << endl ;
//       LogDebug(errorCategory_) << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug(errorCategory_) << "--------------------------------------------------------" << endl ;
   
//       // ---------------------------------------------------------------------------------------------------- 
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;

//       // ---------------------------------------------------------------------------------------------------- 
//       // delete the vector of deviceDescriptions
//       FecFactory::deleteVector (devices) ;
//     }
//     else {
//       edm::LogError(errorCategory_ ) << "No devices found in the database" << endl ;
//     }
