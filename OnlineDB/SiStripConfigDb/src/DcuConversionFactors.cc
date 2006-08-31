// Last commit: $Id: DcuConversionFactors.cc,v 1.3 2006/07/26 11:27:19 bainbrid Exp $
// Latest tag:  $Name: V00-01-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/src/DcuConversionFactors.cc,v $

#include "OnlineDB/SiStripConfigDb/interface/SiStripConfigDb.h"

using namespace std;

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuConversionFactors& SiStripConfigDb::getDcuConversionFactors() {

  if ( !deviceFactory(__FUNCTION__) ) { return dcuConversionFactors_; }
  if ( !resetDcuConvs_ ) { return dcuConversionFactors_; }
  
  
  try {
    dcuConversionFactors_ = deviceFactory(__FUNCTION__)->getConversionFactors();
    resetDcuConvs_ = false;
  }
  catch (...) { 
    handleException( __FUNCTION__, "Problems retrieving DCU conversion factors!" ); 
  }
  
  stringstream ss; 
  if ( dcuConversionFactors_.empty() ) {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " No DCU conversion factors found";
    if ( !usingDb_ ) { ss << " in input 'dcuconv.xml' file " << inputDcuConvXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogError(logCategory_) << ss.str();
    throw cms::Exception(logCategory_) << ss.str();
  } else {
    ss << "[" << __PRETTY_FUNCTION__ << "]"
       << " Found " << dcuConversionFactors_.size() << " DCU conversion factors";
    if ( !usingDb_ ) { ss << " in input 'dcuconv.xml' file " << inputDcuConvXml_; }
    else { ss << " in database partition '" << partition_.name_ << "'"; }
    edm::LogInfo(logCategory_) << ss.str();
  }
  
  return dcuConversionFactors_;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::resetDcuConversionFactors() {
  dcuConversionFactors_.clear();
  resetDcuConvs_ = true;
}


// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::setDcuConversionFactors( const DcuConversionFactors& dcu_convs ) {
  dcuConversionFactors_ = dcu_convs;
}

// -----------------------------------------------------------------------------
// 
void SiStripConfigDb::uploadDcuConversionFactors() {
  //dcuConversionFactors_ = dcu_convs;
}

// -----------------------------------------------------------------------------
// 
const SiStripConfigDb::DcuConversionFactors& SiStripConfigDb::createDcuConversionFactors( const SiStripFecCabling& fec_cabling ) {
  
  // Static container
  static DcuConversionFactors static_dcu_conversions;
  static_dcu_conversions.clear();
  
  // Default settings for DCU conversion factors 
  TkDcuConversionFactors dcu_conv_default ( 0, "", DCUCCU ) ;
  dcu_conv_default.setAdcGain0(2.144) ;
  dcu_conv_default.setAdcOffset0(0) ;
  dcu_conv_default.setAdcCal0(false) ;
  dcu_conv_default.setAdcInl0(0) ;
  dcu_conv_default.setAdcInl0OW(true) ;
  dcu_conv_default.setI20(0.02122);
  dcu_conv_default.setI10(.01061);
  dcu_conv_default.setICal(false) ;
  dcu_conv_default.setKDiv(0.56) ;
  dcu_conv_default.setKDivCal(false) ;
  dcu_conv_default.setTsGain(8.9) ;
  dcu_conv_default.setTsOffset(2432) ;
  dcu_conv_default.setTsCal(false) ;
  dcu_conv_default.setR68(0) ;
  dcu_conv_default.setR68Cal(false) ;
  dcu_conv_default.setAdcGain2(0) ;
  dcu_conv_default.setAdcOffset2(0) ;
  dcu_conv_default.setAdcCal2(false) ;
  dcu_conv_default.setAdcGain3(0) ;
  dcu_conv_default.setAdcCal3(false) ;
  
  // Unique key (within a partition)
  keyType index; 
  
  // Iterate through control system, create descriptions and populate containers 
  for ( vector<SiStripFecCrate>::const_iterator icrate = fec_cabling.crates().begin(); icrate != fec_cabling.crates().end(); icrate++ ) {
    for ( vector<SiStripFec>::const_iterator ifec = icrate->fecs().begin(); ifec != icrate->fecs().end(); ifec++ ) {
      
      // FEC hardware id (encodes crate + slot numbers)
      stringstream fec_hardware_id; 
      fec_hardware_id << setw(4) << setfill('0') << 100 * icrate->fecCrate() + ifec->fecSlot();

      // Set sub-detector string for DCU conversion factor
      dcu_conv_default.setSubDetector( "ROB" );
      
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
      
	// Add DCU conversion factors (for DCU of dummy CCU) at FEC ring level
	TkDcuConversionFactors* dcu_conv = new TkDcuConversionFactors( dcu_conv_default );
	dcu_conv->setDetId( 0 );
	dcu_conv->setDcuHardId( dcu_id );
	if ( static_dcu_conversions.find(dcu_id) == static_dcu_conversions.end() ) {
	  static_dcu_conversions[dcu_id] = dcu_conv;
	} else {
	  stringstream ss;
	  ss << "[" << __PRETTY_FUNCTION__ << "]" << " DCU id " << dcu_id
	     << " already exists within map of DCU conversion factors!";
	  edm::LogError(logCategory_) << ss.str() << "\n";
	  //throw cms::Exception(logCategory_) << ss.str();
	}
	edm::LogInfo(logCategory_)
	  << "[SiStripConfigDb::createPartition]" 
	  << " Added conversion factors for DCU with address 0x"
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

	  // Add DCU conversion factors at CCU level
	  TkDcuConversionFactors* dcu_conv = new TkDcuConversionFactors( dcu_conv_default );
	  dcu_conv->setDetId( 0 );
	  dcu_conv->setDcuHardId( dcu_id );
	  if ( static_dcu_conversions.find(dcu_id) == static_dcu_conversions.end() ) {
	    static_dcu_conversions[dcu_id] = dcu_conv;
	  } else {
	    stringstream ss;
	    ss << "[" << __PRETTY_FUNCTION__ << "]" << " DCU id " << dcu_id
	       << " already exists within map of DCU conversion factors!";
	    edm::LogError(logCategory_) << ss.str() << "\n";
	    //throw cms::Exception(logCategory_) << ss.str();
	  }
	  edm::LogInfo(logCategory_)
	    << "[SiStripConfigDb::createPartition]" 
	    << " Added conversion factors for DCU at address 0x"
	    << hex << setw(8) << setfill('0') << index << dec;
	  
	  for ( vector<SiStripModule>::const_iterator imod = iccu->modules().begin(); imod != iccu->modules().end(); imod++ ) {
	    
	    index = buildCompleteKey( ifec->fecSlot(), 
				      iring->fecRing(), 
				      iccu->ccuAddr(), 
				      imod->ccuChan(), 
				      0x0 ); // I2C address

	    uint32_t dcu_id = SiStripControlKey::key( icrate->fecCrate(), 
						      ifec->fecSlot(), 
						      iring->fecRing(), 
						      iccu->ccuAddr(), 
						      imod->ccuChan(),
						      0x0 );
	    
	    // Add DCU conversion factors
	    TkDcuConversionFactors* dcu_conv = new TkDcuConversionFactors( dcu_conv_default );
	    dcu_conv->setDetId( imod->detId() );
	    dcu_conv->setDcuHardId( dcu_id ); // has to be unique within tracker
	    if ( static_dcu_conversions.find(dcu_id) == static_dcu_conversions.end() ) {
	      static_dcu_conversions[dcu_id] = dcu_conv;
	    } else {
	      stringstream ss;
	      ss << "[" << __PRETTY_FUNCTION__ << "]" << " DCU id " << dcu_id
		 << " already exists within map of DCU conversion factors!";
	      edm::LogError(logCategory_) << ss.str() << "\n";
	      //throw cms::Exception(logCategory_) << ss.str();
	    }
	    edm::LogInfo(logCategory_)
	      << "[SiStripConfigDb::createPartition]" 
	      << " Added conversion factors for DCU with address 0x"
	      << hex << setw(8) << setfill('0') << index << dec;
	    
	  }
	}
      }
    }
  }

  if ( static_dcu_conversions.empty() ) {
    stringstream ss;
    ss << "[" << __PRETTY_FUNCTION__ << "] No DCU conversion factors created!";
    edm::LogError(logCategory_) << ss.str() << "\n";
    //throw cms::Exception(logCategory_) << ss.str() << "\n";
  }

  return static_dcu_conversions;
  
}

