#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <sstream>

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::SiStripConfigDb(  string user, 
				   string passwd, 
				   string path, 
				   string partition ) : 
  factory_(0), 
  user_( user ),
  passwd_( passwd ), 
  path_( path ),
  fromXml_(false),
  xmlFile_(""),
  partition_( partition ),
  allDevices_(), 
  apvDevices_(), //@@ needed?
  dcuDevices_(), //@@ needed?
  fedConnections_()
{
  edm::LogInfo("ConfigDb") << "[SiStripConfigDb::SiStripConfigDb] Constructing object...";
  openDbConnection();
}

// -----------------------------------------------------------------------------
//
SiStripConfigDb::~SiStripConfigDb() {
  edm::LogInfo("ConfigDb") << "[SiStripConfigDb::~SiStripConfigDb] Destructing object...";
  if ( !allDevices_.empty() ) { FecFactory::deleteVector( allDevices_ ); }
  if ( !apvDevices_.empty() ) { FecFactory::deleteVector( apvDevices_ ); }
  if ( !dcuDevices_.empty() ) { FecFactory::deleteVector( dcuDevices_ ); }
  allDevices_.clear();
  apvDevices_.clear();
  dcuDevices_.clear();
  closeDbConnection();
}

// -----------------------------------------------------------------------------
// 
bool SiStripConfigDb::openDbConnection() {
  
  if ( user_ == "" || passwd_ == "" || path_ == "" ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
			      << " Problem with DB connection parameters!"
			      << " Attempting to use env. var. CONFDB...";
    DbAccess::getDbConfiguration( user_, passwd_, path_ );
    if ( user_ == "" || passwd_ == "" || path_ == "" ) {
      edm::LogError("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
				<< " Env. var. CONFDB set incorrectly!";
      return false;
    }
  }
  
  edm::LogInfo("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
			   << " Connection parameters:"
			   << "  user_: " << user_ 
			   << "  passwd_: " << passwd_ 
			   << "  path_: " << path_;
  
  try {
    // factory_ = new DeviceFactory( user_, passwd_, path_ );
    factory_ = new DeviceFactory();
    LogDebug("ConfigDb") << "[SiStripConfigDb::openDbConnection] "
			 << "DeviceFactory created at address: " << factory_ << " "
			 << "with connection parameters " 
			 << user_ << "/" << passwd_ << "@" << path_;
  } 
  catch ( oracle::occi::SQLException e ) { 
    edm::LogError("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
			      << " Caught oracle::occi::SQLException : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
			      << " Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::openDbConnection]"
			      <<" Caught unknown exception : ";
  }
  return true;
}

// -----------------------------------------------------------------------------
//
bool SiStripConfigDb::closeDbConnection() {
  
  try { 
    if ( factory_ ) { delete factory_; }
  }
  catch ( oracle::occi::SQLException e ) { 
    edm::LogError("ConfigDb") << "[SiStripConfigDb::closeDbConnection]"
			      << " Caught oracle::occi::SQLException : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::closeDbConnection]"
			      << " Caught exception : " 
			      << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::closeDbConnection]"
			      <<" Caught unknown exception : ";
  }
  return true;
}

// -----------------------------------------------------------------------------
// Setter for partition name
string SiStripConfigDb::partitionName() { 
  LogDebug("ConfigDb") << "[SiStripConfigDb::partitionName]";
  if ( partition_ == "" ) { partition_ = "TEMP_VALUE!"; } //@@ retrieve from where? configuration?
  LogDebug("ConfigDb") << "[SiStripConfigDb::partitionName]"
		       << " Partition name is " << partition_;
  return partition_;
  
}

// -----------------------------------------------------------------------------
// 
pair<int16_t,int16_t> SiStripConfigDb::partitionVersion( string partition_name ) {
  
  pair<int16_t,int16_t> version(-1,-1);

  // Retrieve list containing partition versions
  list<unsigned int*>* partition_version = 0;
  try {
    partition_version = factory_->getFecDevicePartitionVersion( partition_name );
  } 
  catch ( FecExceptionHandler e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::partitionVersion]"
			      <<" Caught FecExceptionHandler exception : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::partitionVersion]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::partitionVersion]"
			      << " Caught unknown exception : ";
  }
  
  // Check if version list is empty
  if ( partition_version->empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::partitionVersion]"
			      << " No versions exist for partition " << partition_name;
    return version;
  }
  
  // Check if more than one version is found
  if ( partition_version->size() > 1 ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::partitionVersion]"
			      << " Several versions (" << partition_version->size() 
			      << ") exist for partition " << partition_name;
  }
  
  // Iterate through list of versions
  list<unsigned int*>::iterator begin = partition_version->begin();
  list<unsigned int*>::iterator end = partition_version->end();
  list<unsigned int*>::iterator iter;
  for ( iter = begin; iter != end; iter++ ) {
    unsigned int* value = *iter;
    LogDebug("ConfigDb") << "[SiStripConfigDb::partitionVersion] "
			 << "Partition name: " << factory_->getPartitionName(value[0]) 
			 << ", Partition ID: " << value[0] 
			 << ", Partition version: " << value[1] << "." << value[2];
    if ( iter == begin ) {
      version.first = static_cast<int16_t>( value[1] );
      version.second = static_cast<int16_t>( value[2] );
    }
  }
  
  return version;
  
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress SiStripConfigDb::hwAddresses( deviceDescription& description ) {
  
  DeviceAddress addr;
  keyType key;
  try {
    key = description.getKey();
  }
  catch ( FecExceptionHandler e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::hwAddresses]"
			      <<" Caught FecExceptionHandler exception : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::hwAddresses]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::hwAddresses]"
			      <<" Caught unknown exception : ";
  }

  // Extract hardware addresses
  addr.fecCrate = static_cast<int16_t>( 0/*getCrateKey(key)*/ ); //@@ <- needs implementing!
  addr.fecSlot  = static_cast<int16_t>( getFecKey(key) );
  addr.fecRing  = static_cast<int16_t>( getRingKey(key) );
  addr.ccuAddr  = static_cast<int16_t>( getCcuKey(key) );
  addr.ccuChan  = static_cast<int16_t>( getChannelKey(key) );
  addr.i2cAddr  = static_cast<int16_t>( getAddressKey(key) );
  
  LogDebug("ConfigDb") << "[SiStripConfigDb::hwAddresses]"
		       << " deviceDescription*: " << &description
		       << ", key: " << key
		       << ", fec_crate: " << addr.fecCrate
		       << ", fec_slot: " << addr.fecSlot
		       << ", fec_ring: " << addr.fecRing
		       << ", ccu_addr: " << addr.ccuAddr
		       << ", ccu_chan: " << addr.ccuChan
		       << ", i2c_addr: " << addr.i2cAddr;
  
  return addr;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::feDevices( enumDeviceType device_type,
				 deviceVector& devices ) {
  
  string device_name("");
  if      ( device_type == APV25 )       { device_name = "APV25"; }
  else if ( device_type == PLL )         { device_name = "PLL"; }
  else if ( device_type == APVMUX )      { device_name = "APVMUX"; }
  else if ( device_type == DCU )         { device_name = "DCU"; }
  else if ( device_type == LASERDRIVER ) { device_name = "LASERDRIVER"; }
  else { device_name = "UNKNOWN"; }
  
  pair<int,int> version = partitionVersion( partitionName() );
  
  try {
    if ( !allDevices_.empty() ) { FecFactory::deleteVector( allDevices_ ); }
    factory_->getFecDeviceDescriptions( partitionName(), 
					allDevices_, 
					version.first, 
					version.second );
  }
  catch ( FecExceptionHandler e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::feDevices]"
			      <<" Caught FecExceptionHandler exception : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::feDevices]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::feDevices]"
			      <<"Caught unknown exception : ";
  }
  
  if ( allDevices_.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::getApvDevices]"
			      << " No FE devices found for partition " << partitionName()
			      << " and version " << version.first << "." << version.second;
  } else {
    LogDebug("ConfigDb") << "[SiStripConfigDb::feDevices]"
			 << " Found " << allDevices_.size() << " FE devices "
			 << "for partition " << partitionName() 
			 << " and version " << version.first << "." << version.second;
  }
  
  deviceSummary(); 
  
}

// -----------------------------------------------------------------------------
//
void SiStripConfigDb::deviceSummary() { 
  LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary]";
  
  pair<int,int> version = partitionVersion( partitionName() );
  
  if ( allDevices_.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::deviceSummary]"
			      << " No devices found for partition " << partitionName()
			      << " and version " << version.first << "." << version.second;
  } else {
    unsigned int apv_cntr, pll_cntr, mux_cntr, dcu_cntr, las_cntr, doh_cntr, misc_cntr;
    apv_cntr = pll_cntr = mux_cntr = dcu_cntr = las_cntr = doh_cntr = misc_cntr = 0;
    deviceVector::iterator idevice;
    for ( idevice = allDevices_.begin(); idevice != allDevices_.end(); idevice++ ) {
      if      ( (*idevice)->getDeviceType() == APV25 )       { apv_cntr++; }
      else if ( (*idevice)->getDeviceType() == PLL )         { pll_cntr++; }
      else if ( (*idevice)->getDeviceType() == APVMUX )      { mux_cntr++; }
      else if ( (*idevice)->getDeviceType() == DCU )         { dcu_cntr++; }
      else if ( (*idevice)->getDeviceType() == LASERDRIVER ) { las_cntr++; }
      else if ( (*idevice)->getDeviceType() == DOH )         { doh_cntr++; }
      else { misc_cntr++; }
    }
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << apv_cntr << " APV devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << pll_cntr << " PLL devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << mux_cntr << " APVMUX devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << dcu_cntr << " DCU devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << las_cntr << " LASERDRIVER devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << doh_cntr << " DOH devices";
    LogDebug("ConfigDb") << "[SiStripConfigDb::deviceSummary] Found " << misc_cntr << " other MISCELLANEOUS devices";
    // FecFactory::display( allDevices_ ); 
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::apvDescriptions( vector<apvDescription*>& apv_descriptions ) {

  pair<int,int> version = partitionVersion( partitionName() );
  
  deviceVector apv_devices;
  try {
    feDevices( APV25, apv_devices );
  }
  catch ( FecExceptionHandler e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::apvDescriptions]"
			      <<" Caught FecExceptionHandler exception : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::apvDescriptions]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::apvDescriptions]"
			      <<" Caught unknown exception : ";
  }
  
  if ( !apv_devices.empty() ) {
    deviceVector::iterator idevice;
    for ( idevice = apv_devices.begin() ; idevice != apv_devices.end() ; idevice++ ) {
      apvDescription* apv = dynamic_cast<apvDescription*>( *idevice );
      if ( apv ) { 
	apv_descriptions.push_back( apv ); 
	DeviceAddress addr = hwAddresses( *apv );
	LogDebug("ConfigDb") << "[SiStripConfigDb::apvDescriptions] APV25 found at "
			     << " FEC crate: " << addr.fecCrate
			     << ", FEC slot: " << addr.fecSlot
			     << ", FEC ring: " << addr.fecRing
			     << ", CCU addr: " << addr.ccuAddr
			     << ", CCU chan: " << addr.ccuChan
			     << ", I2C Addr: " << addr.i2cAddr;
      }
    }
  }
  
  if ( apv_descriptions.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::apvDescriptions]"
			      << " No APV descriptions found for partition " 
			      << partitionName();
  } else {
    LogDebug("ConfigDb") << "[SiStripConfigDb::apvDescriptions] "
			 << "Found " << apv_descriptions.size() << " APV descriptions "
			 << "for partition " << partitionName() 
			 << " and version " << version.first << "." << version.second;
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::dcuDescriptions( vector<dcuDescription*>& dcu_descriptions ) {
  
  pair<int,int> version = partitionVersion( partitionName() );
  
  deviceVector dcu_devices;
  try {
    factory_->getDcuDescriptions( partitionName(), dcuDevices_ );
  }
  catch ( FecExceptionHandler e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::dcuDescriptions]"
			      <<" Caught FecExceptionHandler exception : " 
			      << e.getMessage();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::dcuDescriptions]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::dcuDescriptions]"
			      <<" Caught unknown exception : ";
  }
  
  if ( !dcu_devices.empty() ) {
    deviceVector::iterator idevice;
    for ( idevice = dcu_devices.begin() ; idevice != dcu_devices.end() ; idevice++ ) {
      dcuDescription* dcu = dynamic_cast<dcuDescription*>( *idevice );
      if ( dcu ) { 
	dcu_descriptions.push_back( dcu ); 
	DeviceAddress addr = hwAddresses( *dcu );
	LogDebug("ConfigDb") << "[SiStripConfigDb::dcuDescriptions] DCU found at "
			     << " FEC crate: " << addr.fecCrate
			     << ", FEC slot: " << addr.fecSlot
			     << ", FEC ring: " << addr.fecRing
			     << ", CCU addr: " << addr.ccuAddr
			     << ", CCU chan: " << addr.ccuChan
			     << ", I2C Addr: " << addr.i2cAddr;
      }
    }
  }
  
  if ( dcu_descriptions.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::dcuDescriptions]"
			      << " No DCU descriptions found for partition " 
			      << partitionName();
  } else {
    LogDebug("ConfigDb") << "[SiStripConfigDb::dcuDescriptions] "
			 << "Found " << dcu_descriptions.size() << " DCU descriptions "
			 << "for partition " << partitionName() 
			 << " and version " << version.first << "." << version.second;
  }
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::fedDescriptions( vector<Fed9U::Fed9UDescription*>& fed_descriptions ) {

  pair<int16_t,int16_t> version = partitionVersion( partitionName() );
  
  vector<Fed9U::Fed9UDescription*>* descriptions;
  try {
    LogDebug("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			 << " getUsingDb(): " << factory_->getUsingDb();
    LogDebug("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			 << " getUsingStrips(): " << factory_->getUsingStrips();
    descriptions = factory_->getFed9UDescriptions( partitionName(), 
						   version.first, 
						   version.second );
    fed_descriptions = *descriptions;
  }
  catch ( ICUtils::ICException& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			      <<" Caught ICUtils::ICException : " << e.what();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			      <<" Caught unknown exception";
  }
  
  if ( fed_descriptions.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			      << " No FED descriptions found for partition " << partitionName()
			      << " and version " << version.first << "." << version.second;
  } else {
    LogDebug("ConfigDb") << "[SiStripConfigDb::fedDescriptions]"
			 << " Found " << fed_descriptions.size() << " FED descriptions "
			 << "for partition " << partitionName()
			 << " and version " << version.first << "." << version.second;
  }
  
}

// -----------------------------------------------------------------------------
/** */ 
void SiStripConfigDb::fedIds( vector<unsigned short>& fed_ids ) {
  
  pair<int16_t,int16_t> version = partitionVersion( partitionName() );
  
  vector<Fed9U::Fed9UDescription*> feds;
  try {
    factory_->setUsingStrips( false );
    fedDescriptions( feds );
    factory_->setUsingStrips( true );
  }
  catch ( ICUtils::ICException& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedIds]"
			      <<" Caught ICUtils::ICException : " << e.what();
  }
  catch ( exception& e ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedIds]"
			      <<" Caught exception : " << e.what();
  }
  catch (...) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedIds]"
			      <<" Caught unknown exception";
  }
  
  vector<Fed9U::Fed9UDescription*>::iterator ifed;
  for ( ifed = feds.begin(); ifed != feds.end(); ifed++ ) {
    fed_ids.push_back( (*ifed)->getFedId() );
  }
  
  LogDebug("ConfigDb") << "[SiStripConfigDb::fedIds] "
		       << "Found " << fed_ids.size() << " FED ids "
		       << "for the partition name " << partitionName()
		       << " and version " << version.first << "." << version.second;
  
  if ( !fed_ids.empty() ) {
    stringstream ss;
    ss << "[SiStripConfigDb::fedIds] FED ids: ";
    vector<unsigned short>::iterator ifed;
    for ( ifed = fed_ids.begin(); ifed != fed_ids.end(); ifed++ ) { ss << *ifed << ", "; }
    LogDebug("ConfigDb") << ss.str();
  }
  
  if ( fed_ids.empty() ) {
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedIds]"
			      << " No FED ids found for the partition name " 
			      << partitionName();
  }
  
}

// -----------------------------------------------------------------------------
// 
vector<FedChannelConnectionDescription*>& SiStripConfigDb::fedConnections( bool reset_cache ) {
  
  if ( reset_cache ) {
    vector<FedChannelConnectionDescription*>::iterator iconn;
    for ( iconn = fedConnections_.begin(); 
	  iconn != fedConnections_.end();
	  iconn++ ) { delete (*iconn); }
    fedConnections_.clear(); 
  }
  
  if ( fedConnections_.empty() ) {
    
    try {
      if ( fromXml_ ) { 
	if ( xmlFile_ == "" ) {
	  edm::LogError("ConfigDb") << "[SiStripConfigDb::fedConnections]"
				    <<" No 'module.xml' specified!"
				    << " Check 'xmlFileName' configurable is set!";
	}
	factory_->createInputFileAccess();
	factory_->addFileName( xmlFile_ ); 
	LogDebug("ConfigDb") << "[SiStripConfigDb::fedConnections]"
			     << " Parsing of file '" << xmlFile_ << "' completed";
      } else {
	factory_->createInputDBAccess();
	factory_->setInputDBVersion( partitionName() );
	LogDebug("ConfigDb") << "[SiStripConfigDb::fedConnections]"
			     << " Parsing of data from partition '" 
			     << partitionName() << "' completed";
      }
    }
    catch ( FecExceptionHandler e ) {
      edm::LogError("ConfigDb") << "[SiStripConfigDb::fedConnections]"
				<<" Caught FecExceptionHandler exception : Problems accessing database " 
				<< e.getMessage();
    }
    catch ( exception& e ) {
      edm::LogError("ConfigDb") << "[SiStripConfigDb::fedConnections]"
				<<" Caught exception : Problems accessing database " << e.what();
    }
    catch (...) {
      edm::LogError("ConfigDb") << "[SiStripConfigDb::fedConnections]"
				<<" Caught unknown exception : Problems accessing database ";
    }

  }
  for ( int iconn = 0; iconn < factory_->getNumberOfFedChannel(); iconn++ ) {
    fedConnections_.push_back( factory_->getFedChannelConnection( iconn ) ); 
    LogDebug("ConfigDb") << "[SiStripConfigDb::fedConnections]"
			 << " FED connection number " << iconn 
			 << " (total of " << factory_->getNumberOfFedChannel()
			 << ") with parameters: ";
    stringstream ss; factory_->getFedChannelConnection(iconn)->toXML( ss );
    LogDebug("ConfigDb") << ss.str();
  }

  stringstream ss; 
  ss << "[SiStripConfigDb::fedConnections] "
     << "Found " << fedConnections_.size() 
     << " FED connections";
  if ( fromXml_ ) { 
    ss << " in input (xml) file '" << xmlFile_ << "'";
  } else { 
    pair<int,int> version = partitionVersion( partitionName() );
    ss << " in database partition '" << partitionName()
       << "' and version " << version.first << "." << version.second;
  }
  LogDebug("ConfigDb") << ss.str();
    
  if ( fedConnections_.empty() ) {
    stringstream ss; ss << "No FED connections found";
    if ( fromXml_ ) { ss << " in input xml file " << xmlFile_; }
    else { ss << " in database partition " << partitionName(); }
    edm::LogError("ConfigDb") << "[SiStripConfigDb::fedConnections]" << ss.str();
  }
  
  return fedConnections_;
  
}














// // -----------------------------------------------------------------------------
// // 
// void SiStripConfigDb::aohDescriptions( vector<laserdriverDescription*>& aoh_descriptions ) {
//   LogDebug("ConfigDb") << "[SiStripConfigDb::aohDescriptions] " 
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
// 	LogDebug("ConfigDb") << "[SiStripConfigDb::aohDescriptions] "
// 		      << "LASERDRIVER (AOH) found at: "
// 		      << "FEC slot: " << getFecKey(     aoh->getKey() ) << ", "
// 		      << "FEC ring: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "CCU addr: " << getCcuKey(     aoh->getKey() ) << ", "
// 		      << "CCU chan: " << getChannelKey( aoh->getKey() ) << ", "
// 		      << "I2C Addr: " << getAddressKey( aoh->getKey() );
//       }
//     }

//     LogDebug("ConfigDb") << "[SiStripConfigDb::aohDescriptions] "
// 		  << "Found " << aoh_descriptions.size() << " AOH descriptions "
// 		  << "for the partition name " << partitionName() 
// 		  << " and version " << version.first << "." << version.second
// 		 ;

//     // Check if any AOH descriptions are found
//     if ( aoh_descriptions.empty() ) {
//       edm::LogError("ConfigDb") << warning("[SiStripConfigDb::aohDescriptions] ")
// 	   << "No LASERDRIVER (AOH) descriptions found for the partition name " << partitionName()
// 	   << endl ;
//     }

//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError("ConfigDb") << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught FecExceptionHandler exception : Problems accessing database " 
// 	 << e.getMessage();
//   }
//   catch ( exception& e ) {
//     edm::LogError("ConfigDb") << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught exception : Problems accessing database " 
// 	 << e.what();
//   }
//   catch (...) {
//     edm::LogError("ConfigDb") << warning("[SiStripConfigDb::aohDescriptions] ")
// 	 <<"Caught unknown exception : Problems accessing database " 
// 	;
//   }
  
// }



// // -----------------------------------------------------------------------------
// // Open connection to database
// void SiStripConfigDb::feDevices() {
//   LogDebug("ConfigDb") << "[SiStripConfigDb::feDevices] "
// 		<< " ";
  
// #ifdef USING_DATABASE

//   pair<int,int> version = partitionVersion();
//   SimpleConfigurable<string> partitionName()("nil","SiStripConfigDb:PartitionName");
//   try {
//     deviceVector devices = factory_->getFecDeviceDescriptions( partitionName(), version.first, version.second );
//     if ( devices.empty() ) {
//       edm::LogError("ConfigDb") << "[SiStripConfigDb::feDevices] "
// 	   << "ERROR : No FE devices exist for the partition name " << partitionName()
// 	   << endl ;
//     }
//     if ( devices.size() ) {
//       LogDebug("ConfigDb") << "[SiStripConfigDb::feDevices] "
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
//       LogDebug("ConfigDb") << "Found " << apv_cntr << " APV ";
//     }
//   }
//   catch ( FecExceptionHandler e ) {
//     edm::LogError("ConfigDb") << "[SiStripConfigDb::feDevices] "
// 	 <<"ERROR : problems retrieving FE devices " 
// 	 << e.getMessage();
//   }

// #else
//   edm::LogError("ConfigDb") << warning("[SiStripConfigDb::openConnection] ")
//        << "USING_DATABASE not defined! => not using database";        
// #endif // USING_DATABASE

// }








//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set all the devices with modification in the PLL
//       unsigned int major, minor ;
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, true) ;
//       //                                                                                  ^ major version, false = minor version
    
//       LogDebug("ConfigDb") << "-------------------------- Version ---------------------" << endl ;
//       LogDebug("ConfigDb") << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug("ConfigDb") << "--------------------------------------------------------" << endl ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;
    
//       // ----------------------------------------------------------------------------------------------------
//       // Upload in database => create a version major with a modification on the PLL
//       // set only the PLL devices
//       deviceFactory->setFecDeviceDescriptions (devices, partitionName, &major, &minor, false) ;
//       //                                                                                  ^ minor version, true = major version
    
//       LogDebug("ConfigDb") << "-------------------------- Version ---------------------" << endl ;
//       LogDebug("ConfigDb") << "Upload the version " << dec << major << "." << minor << " in the database" << endl ;
//       LogDebug("ConfigDb") << "--------------------------------------------------------" << endl ;
   
//       // ---------------------------------------------------------------------------------------------------- 
//       // set the version as to be downloaded
//       deviceFactory->setFecDevicePartitionVersion ( partitionName, major, minor ) ;

//       // ---------------------------------------------------------------------------------------------------- 
//       // delete the vector of deviceDescriptions
//       FecFactory::deleteVector (devices) ;
//     }
//     else {
//       edm::LogError("ConfigDb") << "No devices found in the database" << endl ;
//     }





