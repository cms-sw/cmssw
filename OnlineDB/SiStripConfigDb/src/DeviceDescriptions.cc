// Last commit: $Id: DeviceDescriptions.cc,v 1.4 2006/08/31 19:49:41 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DeviceDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::getDeviceDescriptions( const enumDeviceType& device_type,	
										   bool all_devices_except ) {
  stringstream ss;
  ss << "[" << __PRETTY_FUNCTION__ << "]";
  if ( !all_devices_except ) { 
    ss << " Retrieving " << deviceType( device_type ) << " descriptions..."; 
  } else {
    ss << " Retrieving descriptions for all devices except " 
       << deviceType( device_type ) << "...";
  }
  edm::LogInfo(logCategory_) << ss.str();
  
  // Use static object to hold device description of a particular type
  static SiStripConfigDb::DeviceDescriptions descriptions;
  descriptions.clear();

  // Retrieve device descriptions if necessary
  if ( devices_.empty() ) { getDeviceDescriptions(); }
  
  if ( !devices_.empty() ) {
    DeviceDescriptions::iterator idevice = devices_.begin();
    for ( ; idevice != devices_.end(); idevice++ ) {
      deviceDescription* desc = *idevice;
      // Extract devices of given type from descriptions found in local cache  
      if ( !all_devices_except && desc->getDeviceType() == device_type ) { descriptions.push_back( desc ); }
      // Extract all devices EXCEPT those of given type from descriptions found in local cache  
      if ( all_devices_except && desc->getDeviceType() != device_type ) { descriptions.push_back( desc ); }
    }
  }

  stringstream sss; 
  if ( descriptions.empty() ) { 
    sss << "[" << __PRETTY_FUNCTION__ << "]";
    if ( !all_devices_except ) {
      sss << " No " << deviceType( device_type ) << " descriptions found";
    } else {
      sss << " No descriptions found (for all devices except " 
	  << deviceType( device_type ) << ")";
    }
    edm::LogError(logCategory_) << sss.str();
  } else {
    sss << "[" << __PRETTY_FUNCTION__ << "]"
	<< " Found " << descriptions.size() << " descriptions (for";
    if ( !all_devices_except ) { sss << " " << deviceType( device_type ) << ")"; }
    else { sss << " all devices except " << deviceType( device_type ) << ")"; }
    edm::LogInfo(logCategory_) << sss.str();
  }
  
  return descriptions;
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::getDeviceDescriptions() {
  edm::LogInfo(logCategory_) << "[" << __PRETTY_FUNCTION__ << "]"
			     << " Retrieving device descriptions...";
  
  if ( !deviceFactory(__FUNCTION__) ) { return devices_; }
  if ( !resetDevices_ ) { return devices_; }

  try { 
    
    if ( !usingDb_ ) {
      resetPiaResetDescriptions();
      getPiaResetDescriptions();
    }
    
    deviceFactory(__FUNCTION__)->getFecDeviceDescriptions( partition_.name_, 
							   devices_,
							   partition_.major_,
							   partition_.minor_ );
//     deviceFactory(__FUNCTION__)->getDcuDescriptions( partition_.name_, 
// 						     devices_ );
    resetDevices_ = false;
    
  }
  catch (...) { handleException( __FUNCTION__ ); }
  
  stringstream ss; 
  if ( devices_.empty() ) {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " No device descriptions found";
    if ( !usingDb_ ) { ss << " in " << inputFecXml_.size() << " 'fec.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(logCategory_) << ss.str();
  } else {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Found " << devices_.size() << " device descriptions";
    if ( !usingDb_ ) { ss << " in " << inputFecXml_.size() << " 'fec.xml' file(s)"; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(logCategory_) << ss.str();
  }

  return devices_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::resetDeviceDescriptions() {
  //FecFactory::deleteVector( devices_ );
  devices_.clear();
  resetDevices_ = true;
}

// -----------------------------------------------------------------------------
// 
//@@ if new major, upload all desc. if not, upload just modified ones... ???
void SiStripConfigDb::uploadDeviceDescriptions( bool new_major_version ) {

  if ( !deviceFactory(__FUNCTION__) ) { return; }
  
  try { 
    
    if ( !usingDb_ ) {
      deviceFactory(__FUNCTION__)->setPiaResetDescriptions( piaResets_, 
							    partition_.name_ );
    }
    
    deviceFactory(__FUNCTION__)->setFecDeviceDescriptions( getDeviceDescriptions( DCU, true ), // all devices except DCUs
							   partition_.name_, 
							   &partition_.major_,
							   &partition_.minor_,
							   new_major_version );
    
//     deviceFactory(__FUNCTION__)->setDcuDescriptions( partition_.name_, 
// 						     getDeviceDescriptions( DCU ) );
    
  }
  catch (...) { 
    handleException( __FUNCTION__ ); 
  }
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::createDeviceDescriptions( const SiStripFecCabling& fec_cabling ) {

  // Static container
  static DeviceDescriptions static_device_descriptions;
  static_device_descriptions.clear();
  
  // Default settings for APV, DOH, AOH, MUX and PLL
  apvDescription apv_default( (uint8_t)0x2B, (uint8_t)0x64, (uint8_t)0x04, (uint8_t)0x73,
			      (uint8_t)0x3C, (uint8_t)0x32, (uint8_t)0x32, (uint8_t)0x32,
			      (uint8_t)0x50, (uint8_t)0x32, (uint8_t)0x50, (uint8_t)0x00,
			      (uint8_t)0x43, (uint8_t)0x43, (uint8_t)0x14, (uint8_t)0xFB,
			      (uint8_t)0xFE, (uint8_t)0x00 );
  tscType8 doh_bias[3] = {24,24,24}; laserdriverDescription doh_default(2,doh_bias);
  tscType8 aoh_bias[3] = {23,23,23}; laserdriverDescription aoh_default(2,aoh_bias);
  muxDescription mux_default( (uint16_t)0xFF );
  pllDescription pll_default(6,1);
  
  // Unique key (within partition)
  keyType index;
  
  // Iterate through control system, create descriptions and populate containers 
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      
      // FEC hardware id (encodes FEC crate and slot)
      stringstream fec_hardware_id; 
      fec_hardware_id << setw(4) << setfill('0') << 100 * icrate->fecCrate() + ifec->fecSlot();
      
      for ( vector<SiStripRing>::const_iterator iring = ifec->rings().begin(); iring != ifec->rings().end(); iring++ ) {
	
	index = buildCompleteKey( ifec->fecSlot(), 
				  iring->fecRing(), 
				  0x7F,  // CCU address
				  0x10,  // CCU channel
				  0x0 ); // I2C address

	uint32_t dcu_id = SiStripControlKey::key( icrate->fecCrate(), 
						  ifec->fecSlot(), 
						  iring->fecRing(), 
						  0x7F,  // CCU address
						  0x10,  // CCU channel
						  0x0 ); // I2C address

	// Add DCU (to "dummy" CCU) at FEC ring level
	dcuDescription* dcu = new dcuDescription( index, // access key
						  0,     // timestamp
						  dcu_id,
						  0,0,0,0,0,0,0,0 ); // DCU channels
	dcu->setFecHardwareId( fec_hardware_id.str() );
	static_device_descriptions.push_back( dcu );
	edm::LogInfo(logCategory_)
	  << "[" << __PRETTY_FUNCTION__ << "]" 
	  << " Added DCU to 'dummy' CCU at 'FEC ring' level, with address 0x" 
	  << hex << setw(8) << setfill('0') << index << dec;
	
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  
	  index = buildCompleteKey( ifec->fecSlot(), 
				    iring->fecRing(), 
				    iccu->ccuAddr(), 
				    0x10,  // CCU channel
				    0x0 ); // I2C address

	  uint32_t dcu_id = SiStripControlKey::key( icrate->fecCrate(), 
						    ifec->fecSlot(), 
						    iring->fecRing(), 
						    iccu->ccuAddr(), 
						    0x10,  // CCU channel
						    0x0 ); // I2C address

	  // Add DCU description at CCU level
	  dcuDescription* dcu = new dcuDescription( index, // access key
						    0,     // timestamp
						    dcu_id,
						    0,0,0,0,0,0,0,0 ); // DCU channels
	  dcu->setFecHardwareId( fec_hardware_id.str() );
	  static_device_descriptions.push_back( dcu );
	  edm::LogInfo(logCategory_)
	    << "[" << __PRETTY_FUNCTION__ << "]" 
	    << " Added DCU at 'CCU level', with address 0x" 
	    << hex << setw(8) << setfill('0') << index << dec;

	  // Add two DOH description at CCU level (for CCU = 1 or 2)
	  if ( iccu->ccuAddr() == 1 || iccu->ccuAddr() == 2 ) {
	    laserdriverDescription* doh = new laserdriverDescription( doh_default ) ;
	    index = buildCompleteKey( ifec->fecSlot(), 
				      iring->fecRing(), 
				      iccu->ccuAddr(), 
				      0x10,   // CCU channel
				      0x70 ); // I2C address
	    doh->setAccessKey( index ) ;
	    doh->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( doh ) ;
	    edm::LogInfo(logCategory_)
	      << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " Added DOH at 'CCU level' with address 0x" 
	      << hex << setw(8) << setfill('0') << index << dec;
	  }
	  
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    
	    index = buildCompleteKey( ifec->fecSlot(), 
				      iring->fecRing(), 
				      iccu->ccuAddr(), 
				      imod->ccuChan(), 
				      0x0 ); // I2C address
	  
	    vector<uint16_t> apvs = imod->activeApvs();
	    vector<uint16_t>::const_iterator iapv = apvs.begin();
	    for ( ; iapv != apvs.end(); iapv++ ) {
	      // Add APV descriptions at module level
	      apvDescription* apv = new apvDescription( apv_default );
	      apv->setAccessKey( index | setAddressKey(*iapv) ) ;
	      apv->setFecHardwareId( fec_hardware_id.str() );
	      static_device_descriptions.push_back( apv );
	      edm::LogInfo(logCategory_)
		<< "[" << __PRETTY_FUNCTION__ << "]" 
		<< " Added APV at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | setAddressKey(*iapv) ) << dec;
	    }
	    
	    // Add DCU description at module level
	    dcuDescription* dcu = new dcuDescription( index, // access key
						      0,     // timestamp
						      imod->dcuId(),
						      0,0,0,0,0,0,0,0 ); // DCU channels
	    dcu->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( dcu ) ;
	    edm::LogInfo(logCategory_)
	      << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " Added DCU at 'module' level, with address 0x"
	      << hex << setw(8) << setfill('0') << index << dec;

	    // Add MUX description at module level
	    muxDescription* mux = new muxDescription( mux_default );
	    mux->setAccessKey( index | 0x43 );
	    mux->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( mux );
	    edm::LogInfo(logCategory_)
	      << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " Added MUX at 'module' level, with address 0x"
	      << hex << setw(8) << setfill('0') << uint32_t( index | 0x43 ) << dec;

	    // Add PLL description at module level
	    pllDescription* pll = new pllDescription( pll_default );
	    pll->setAccessKey( index | 0x44 );
	    pll->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( pll );
	    edm::LogInfo(logCategory_)
	      << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " Added PLL at 'module' level, with address 0x"
	      << hex << setw(8) << setfill('0') << uint32_t( index | 0x44 ) << dec;

	    // Add AOH description at module level
	    laserdriverDescription* aoh = new laserdriverDescription( aoh_default ) ;
	    aoh->setAccessKey( index | 0x60 ) ;
	    aoh->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( aoh ) ;
	    edm::LogInfo(logCategory_)
	      << "[" << __PRETTY_FUNCTION__ << "]" 
	      << " Added AOH at 'module' level, with address 0x"
	      << hex << setw(8) << setfill('0') << uint32_t( index | 0x60 ) << dec;

	  }
	}
      }
    }
  }

  if ( static_device_descriptions.empty() ) {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "] No device descriptions created!";
    edm::LogError(logCategory_) << ss.str() << "\n";
  } 
  
  return static_device_descriptions;
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceAddress& SiStripConfigDb::deviceAddress( const deviceDescription& description ) {

  // Set default values
  static SiStripConfigDb::DeviceAddress addr;
  static uint16_t all_ = 0xFFFF;
  addr.fecCrate_ = all_;
  addr.fecSlot_  = all_;
  addr.fecRing_  = all_;
  addr.ccuAddr_  = all_;
  addr.ccuChan_  = all_;
  addr.i2cAddr_  = all_;
  
  // Retrieve FEC key
  keyType key;
  try { key = const_cast<deviceDescription&>(description).getKey(); }
  catch (...) { handleException( __FUNCTION__ ); }
  
  // Extract hardware addresses
  addr.fecCrate_ = static_cast<uint16_t>( 0 ); //@@ getCrateKey(key) );
  addr.fecSlot_  = static_cast<uint16_t>( getFecKey(key) );
  addr.fecRing_  = static_cast<uint16_t>( getRingKey(key) );
  addr.ccuAddr_  = static_cast<uint16_t>( getCcuKey(key) );
  addr.ccuChan_  = static_cast<uint16_t>( getChannelKey(key) );
  addr.i2cAddr_  = static_cast<uint16_t>( getAddressKey(key) );
  
  return addr;
}

// -----------------------------------------------------------------------------
//
string SiStripConfigDb::deviceType( const enumDeviceType& device_type ) const {
  if      ( device_type == PLL )         { return "PLL"; }
  else if ( device_type == LASERDRIVER ) { return "LLD"; }
  else if ( device_type == DOH )         { return "DOH"; }
  else if ( device_type == APVMUX )      { return "MUX"; }
  else if ( device_type == APV25 )       { return "APV"; }
  else if ( device_type == PIARESET )    { return "PIA RESET"; }
  else if ( device_type == GOH )         { return "GOH"; }
  else { return "UNKNOWN DEVICE!"; }
}
