// Last commit: $Id: DeviceDescriptions.cc,v 1.9 2006/11/24 11:41:58 bainbrid Exp $
// Latest tag:  $Name:  $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DeviceDescriptions.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::getDeviceDescriptions( SiStripConfigDb::DeviceDescriptions& descriptions,
					     const enumDeviceType& device_type,	
					     bool all_devices_except ) {
  
  // Retrieve device descriptions if necessary
  if ( devices_.empty() ) { getDeviceDescriptions(); }
  
  // Container to hold descriptions of a particular device type
  descriptions.clear();
  
  // Extract only devices of given type from descriptions found in local cache  
  // OR extract all devices EXCEPT those of given type found in local cache  
  if ( !devices_.empty() ) {
    DeviceDescriptions::iterator idevice = devices_.begin();
    for ( ; idevice != devices_.end(); idevice++ ) {
      deviceDescription* desc = *idevice;
      deviceAddress( *desc );
      if ( !all_devices_except && desc->getDeviceType() == device_type ) { descriptions.push_back( desc ); }
      if ( all_devices_except && desc->getDeviceType() != device_type ) { descriptions.push_back( desc ); }
    }
  }
  
  // Debug
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]";
  if ( descriptions.empty() ) { ss << " Found no device descriptions (for"; }
  else { ss << " Found " << descriptions.size() << " device descriptions (for"; }
  if ( !all_devices_except ) { ss << " devices of type " << deviceType( device_type ) << ")"; }
  else { ss << " all devices NOT of type " << deviceType( device_type ) << ")"; }
  if ( descriptions.empty() ) { edm::LogWarning(mlConfigDb_) << ss; }
  else { LogTrace(mlConfigDb_) << ss; }

}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::getDeviceDescriptions() {
  
  if ( !deviceFactory(__func__) ) { return devices_; }
  if ( !resetDevices_ ) { return devices_; }

  // Retrieve descriptions
  try { 
    if ( !dbParams_.usingDb_ ) {
      resetPiaResetDescriptions();
      getPiaResetDescriptions();
    }
    deviceFactory(__func__)->getFecDeviceDescriptions( dbParams_.partition_, 
						       devices_,
						       dbParams_.major_,
						       dbParams_.minor_ );
    deviceFactory(__func__)->getDcuDescriptions( dbParams_.partition_, 
						 devices_ );
    resetDevices_ = false;
  }
  catch (...) { handleException( __func__ ); }

  // Debug 
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]";
  if ( devices_.empty() ) { ss << " Found no device descriptions"; }
  else { ss << " Found " << devices_.size() << " device descriptions"; }
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.inputFecXml_.size() << " 'fec.xml' file(s)"; }
  else { ss << " in database partition '" << dbParams_.partition_ << "'"; }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss; }
  else { LogTrace(mlConfigDb_) << ss; }

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

  if ( !deviceFactory(__func__) ) { return; }
  
  try { 
    
    if ( !dbParams_.usingDb_ ) {
      deviceFactory(__func__)->setPiaResetDescriptions( piaResets_, 
							dbParams_.partition_ );
    }

    // Retrieve all devices except DCUs
    DeviceDescriptions devices;
    getDeviceDescriptions( devices, DCU, true );

    // Upload devices
    deviceFactory(__func__)->setFecDeviceDescriptions( devices,
						       dbParams_.partition_, 
						       &dbParams_.major_,
						       &dbParams_.minor_,
						       new_major_version );
    
  }
  catch (...) { 
    handleException( __func__ ); 
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

	uint32_t dcu_id = SiStripFecKey::key( icrate->fecCrate(), 
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

	stringstream ss;
	ss << "[SiStripConfigDb::" << __func__ << "]"
	   << " Added DCU to 'dummy' CCU at 'FEC ring' level, with address 0x" 
	   << hex << setw(8) << setfill('0') << index << dec;
	LogTrace(mlConfigDb_) << ss;
	
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  
	  index = buildCompleteKey( ifec->fecSlot(), 
				    iring->fecRing(), 
				    iccu->ccuAddr(), 
				    0x10,  // CCU channel
				    0x0 ); // I2C address

	  uint32_t dcu_id = SiStripFecKey::key( icrate->fecCrate(), 
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
	  stringstream ss1;
	  ss1 << "[SiStripConfigDb::" << __func__ << "]"
	      << " Added DCU at 'CCU level', with address 0x" 
	      << hex << setw(8) << setfill('0') << index << dec;
	  LogTrace(mlConfigDb_) << ss1;
  
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
	    stringstream ss2;
	    ss2 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added DOH at 'CCU level' with address 0x" 
		<< hex << setw(8) << setfill('0') << index << dec;
	    LogTrace(mlConfigDb_) << ss2;
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
	      stringstream ss3;
	      ss3 << "[SiStripConfigDb::" << __func__ << "]"
		  << " Added APV at 'module' level, with address 0x"
		  << hex << setw(8) << setfill('0') << uint32_t( index | setAddressKey(*iapv) ) << dec;
	      LogTrace(mlConfigDb_) << ss3;
	    }
	    
	    // Add DCU description at module level
	    dcuDescription* dcu = new dcuDescription( index, // access key
						      0,     // timestamp
						      imod->dcuId(),
						      0,0,0,0,0,0,0,0 ); // DCU channels
	    dcu->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( dcu ) ;
	    stringstream ss4;
	    ss4 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added DCU at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << index << dec;
	    LogTrace(mlConfigDb_) << ss4;

	    // Add MUX description at module level
	    muxDescription* mux = new muxDescription( mux_default );
	    mux->setAccessKey( index | 0x43 );
	    mux->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( mux );
	    stringstream ss5;
	    ss5 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added MUX at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | 0x43 ) << dec;
	    LogTrace(mlConfigDb_) << ss5;

	    // Add PLL description at module level
	    pllDescription* pll = new pllDescription( pll_default );
	    pll->setAccessKey( index | 0x44 );
	    pll->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( pll );
	    stringstream ss6;
	    ss6 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added PLL at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | 0x44 ) << dec;
	    LogTrace(mlConfigDb_) << ss6;

	    // Add AOH description at module level
	    laserdriverDescription* aoh = new laserdriverDescription( aoh_default ) ;
	    aoh->setAccessKey( index | 0x60 ) ;
	    aoh->setFecHardwareId( fec_hardware_id.str() );
	    static_device_descriptions.push_back( aoh ) ;
	    stringstream ss7;
	    ss7 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added AOH at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | 0x60 ) << dec;
	    LogTrace(mlConfigDb_) << ss7;
	      
	  }
	}
      }
    }
  }

  if ( static_device_descriptions.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No device descriptions created!";
  } 
  
  return static_device_descriptions;
  
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress::DeviceAddress() : 
  fecCrate_(sistrip::invalid_), 
  fecSlot_(sistrip::invalid_), 
  fecRing_(sistrip::invalid_), 
  ccuAddr_(sistrip::invalid_), 
  ccuChan_(sistrip::invalid_), 
  i2cAddr_(sistrip::invalid_) { reset(); }

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::DeviceAddress::reset() { 
  fecCrate_ = sistrip::invalid_; 
  fecSlot_ = sistrip::invalid_; 
  fecRing_ = sistrip::invalid_; 
  ccuAddr_ = sistrip::invalid_; 
  ccuChan_ = sistrip::invalid_; 
  i2cAddr_ = sistrip::invalid_;
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceAddress& SiStripConfigDb::deviceAddress( const deviceDescription& description ) {

  // Set default values
  static SiStripConfigDb::DeviceAddress addr;
  addr.reset();
  
  // Retrieve FEC key
  keyType key;
  try { key = const_cast<deviceDescription&>(description).getKey(); }
  catch (...) { handleException( __func__ ); }
  
  // Extract hardware addresses
  addr.fecCrate_ = static_cast<uint16_t>( 0 ); //@@ always zero???
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
  else if ( device_type == DCU )         { return "DCU"; }
  else if ( device_type == PIARESET )    { return "PIA RESET"; }
  else if ( device_type == GOH )         { return "GOH"; }
  else { return "UNKNOWN DEVICE!"; }
}
