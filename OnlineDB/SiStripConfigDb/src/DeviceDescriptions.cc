// Last commit: $Id: DeviceDescriptions.cc,v 1.21 2008/04/14 05:44:33 bainbrid Exp $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::getDeviceDescriptions( const enumDeviceType& device_type ) {
  
  devices_.clear();

  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return devices_; }
  
  try { 

    DeviceDescriptions all_devices;

    if ( !dbParams_.usingDbCache_ ) { 

      deviceFactory(__func__)->getFecDeviceDescriptions( dbParams_.partitions_.begin()->second.partitionName_, 
							 all_devices,
							 dbParams_.partitions_.begin()->second.fecVersion_.first,
							 dbParams_.partitions_.begin()->second.fecVersion_.second,
							 false ); //@@ do not get DISABLED modules (ie, those removed from cabling). 
      devices_ = FecFactory::getDeviceFromDeviceVector( all_devices, device_type );

    } else {

#ifdef USING_NEW_DATABASE_MODEL
      DeviceDescriptions* tmp = databaseCache(__func__)->getDevices();
      if ( tmp ) { all_devices = *tmp; }
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to DeviceDescriptions vector!";
      }
#endif

      devices_ = FecFactory::getDeviceFromDeviceVector( all_devices, device_type );
   
    }
    
  } catch (...) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << devices_.size()
     << " device descriptions (for devices of type " 
     << deviceType( device_type ) << ")"; 
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.partitions_.begin()->second.inputFecXml_.size() << " 'fec.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partitions_.begin()->second.partitionName_ << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }

  return devices_;
  
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DeviceDescriptions& SiStripConfigDb::getDeviceDescriptions() {

  devices_.clear();
  
  if ( ( !dbParams_.usingDbCache_ && !deviceFactory(__func__) ) ||
       (  dbParams_.usingDbCache_ && !databaseCache(__func__) ) ) { return devices_; }
  
  try { 

    if ( !dbParams_.usingDbCache_ ) { 

      deviceFactory(__func__)->getFecDeviceDescriptions( dbParams_.partitions_.begin()->second.partitionName_, 
							 devices_,
							 dbParams_.partitions_.begin()->second.fecVersion_.first,
							 dbParams_.partitions_.begin()->second.fecVersion_.second,
							 false ); //@@ do not get DISABLED modules (ie, those removed from cabling). 
      
    } else { 
      
#ifdef USING_NEW_DATABASE_MODEL
      DeviceDescriptions* tmp = databaseCache(__func__)->getDevices();
      if ( tmp ) { devices_ = *tmp; }
      else {
	edm::LogWarning(mlConfigDb_)
	  << "[SiStripConfigDb::" << __func__ << "]"
	  << " NULL pointer to DeviceDescriptions vector!";
      }
#endif

    }

  } catch (...) { handleException( __func__ ); }
  
  stringstream ss; 
  ss << "[SiStripConfigDb::" << __func__ << "]"
     << " Found " << devices_.size()
     << " device descriptions";
  if ( !dbParams_.usingDb_ ) { ss << " in " << dbParams_.partitions_.begin()->second.inputFecXml_.size() << " 'fec.xml' file(s)"; }
  else { if ( !dbParams_.usingDbCache_ )  { ss << " in database partition '" << dbParams_.partitions_.begin()->second.partitionName_ << "'"; } 
  else { ss << " from shared memory name '" << dbParams_.sharedMemory_ << "'"; } }
  if ( devices_.empty() ) { edm::LogWarning(mlConfigDb_) << ss.str(); }
  else { LogTrace(mlConfigDb_) << ss.str(); }
  
  return devices_;

}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDeviceDescriptions( bool new_major_version ) {

  if ( dbParams_.usingDbCache_ ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]" 
      << " Using database cache! No uploads allowed!"; 
    return;
  }
  
  if ( !deviceFactory(__func__) ) { return; }
  
  if ( devices_.empty() ) { 
    stringstream ss; 
    ss << "[SiStripConfigDb::" << __func__ << "]" 
       << " Found no cached device descriptions, therefore no upload!"; 
    edm::LogWarning(mlConfigDb_) << ss.str(); 
    return; 
  }
  
  try { 
    deviceFactory(__func__)->setFecDeviceDescriptions( devices_,
						       dbParams_.partitions_.begin()->second.partitionName_, 
						       &(dbParams_.partitions_.begin()->second.fecVersion_.first),
						       &(dbParams_.partitions_.begin()->second.fecVersion_.second),
						       new_major_version );
  } catch (...) { handleException( __func__ ); }
  
  allowCalibUpload_ = true;
  
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::createDeviceDescriptions( const SiStripFecCabling& fec_cabling ) {
  
  // Clear cache
  devices_.clear();
  
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

	uint32_t dcu_id = SiStripFecKey( icrate->fecCrate(), 
					 ifec->fecSlot(), 
					 iring->fecRing(), 
					 0x7F,  // CCU address
					 0x10,  // CCU channel
					 0x0 ).key(); // I2C address
	
	// Add DCU (to "dummy" CCU) at FEC ring level
	dcuDescription* dcu = new dcuDescription( index, // access key
						  0,     // timestamp
						  dcu_id,
						  0,0,0,0,0,0,0,0 ); // DCU channels
	dcu->setFecHardwareId( fec_hardware_id.str() );
	devices_.push_back( dcu );

	stringstream ss;
	ss << "[SiStripConfigDb::" << __func__ << "]"
	   << " Added DCU to 'dummy' CCU at 'FEC ring' level, with address 0x" 
	   << hex << setw(8) << setfill('0') << index << dec;
	LogTrace(mlConfigDb_) << ss.str();
	
	for ( vector<SiStripCcu>::const_iterator iccu = iring->ccus().begin(); iccu != iring->ccus().end(); iccu++ ) {
	  
	  index = buildCompleteKey( ifec->fecSlot(), 
				    iring->fecRing(), 
				    iccu->ccuAddr(), 
				    0x10,  // CCU channel
				    0x0 ); // I2C address

	  uint32_t dcu_id = SiStripFecKey( icrate->fecCrate(), 
					   ifec->fecSlot(), 
					   iring->fecRing(), 
					   iccu->ccuAddr(), 
					   0x10,  // CCU channel
					   0x0 ).key(); // I2C address
	  
	  // Add DCU description at CCU level
	  dcuDescription* dcu = new dcuDescription( index, // access key
						    0,     // timestamp
						    dcu_id,
						    0,0,0,0,0,0,0,0 ); // DCU channels
	  dcu->setFecHardwareId( fec_hardware_id.str() );
	  devices_.push_back( dcu );
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
	    devices_.push_back( doh ) ;
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
	      devices_.push_back( apv );
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
	    devices_.push_back( dcu ) ;
	    stringstream ss4;
	    ss4 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added DCU at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << index << dec;
	    LogTrace(mlConfigDb_) << ss4;

	    // Add MUX description at module level
	    muxDescription* mux = new muxDescription( mux_default );
	    mux->setAccessKey( index | 0x43 );
	    mux->setFecHardwareId( fec_hardware_id.str() );
	    devices_.push_back( mux );
	    stringstream ss5;
	    ss5 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added MUX at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | 0x43 ) << dec;
	    LogTrace(mlConfigDb_) << ss5;

	    // Add PLL description at module level
	    pllDescription* pll = new pllDescription( pll_default );
	    pll->setAccessKey( index | 0x44 );
	    pll->setFecHardwareId( fec_hardware_id.str() );
	    devices_.push_back( pll );
	    stringstream ss6;
	    ss6 << "[SiStripConfigDb::" << __func__ << "]"
		<< " Added PLL at 'module' level, with address 0x"
		<< hex << setw(8) << setfill('0') << uint32_t( index | 0x44 ) << dec;
	    LogTrace(mlConfigDb_) << ss6;

	    // Add AOH description at module level
	    laserdriverDescription* aoh = new laserdriverDescription( aoh_default ) ;
	    aoh->setAccessKey( index | 0x60 ) ;
	    aoh->setFecHardwareId( fec_hardware_id.str() );
	    devices_.push_back( aoh ) ;
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

  if ( devices_.empty() ) {
    edm::LogWarning(mlConfigDb_)
      << "[SiStripConfigDb::" << __func__ << "]"
      << " No device descriptions created!";
  } 
  
}

// -----------------------------------------------------------------------------
// 
SiStripConfigDb::DeviceAddress SiStripConfigDb::deviceAddress( const deviceDescription& description ) {
  
  deviceDescription& desc = const_cast<deviceDescription&>(description); 
  
  DeviceAddress addr;
  try {
#ifdef USING_NEW_DATABASE_MODEL
    addr.fecCrate_ = static_cast<uint16_t>( desc.getCrateSlot() + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset?
#else
    addr.fecCrate_ = static_cast<uint16_t>( 0 + sistrip::FEC_CRATE_OFFSET ); //@@ temporary offset?
#endif
    addr.fecSlot_  = static_cast<uint16_t>( desc.getFecSlot() );
    addr.fecRing_  = static_cast<uint16_t>( desc.getRingSlot() + sistrip::FEC_RING_OFFSET ); //@@ temporary offset?
    addr.ccuAddr_  = static_cast<uint16_t>( desc.getCcuAddress() );
    addr.ccuChan_  = static_cast<uint16_t>( desc.getChannel() );
    addr.i2cAddr_  = static_cast<uint16_t>( desc.getAddress() );
  } catch (...) { handleException( __func__ ); }
  
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
