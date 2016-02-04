// Last commit: $Id: SiStripEventSummary.cc,v 1.11 2008/11/26 16:47:10 bainbrid Exp $

#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripEventSummary::SiStripEventSummary() : 
  valid_(true),
  triggerFed_(0),
  runType_(sistrip::UNDEFINED_RUN_TYPE), 
  event_(0), 
  bx_(0),
  spillNumber_(0),
  nDataSenders_(0),
  fedReadoutMode_(sistrip::UNDEFINED_FED_READOUT_MODE),
  apvReadoutMode_(sistrip::UNDEFINED_APV_READOUT_MODE),
  apveAddress_(0),
  nApvsInSync_(0),
  nApvsOutOfSync_(0),
  nApvsErrors_(0),
  params_(5,0)
{;}

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::commissioningInfo( const uint32_t* const buffer,
					     const uint32_t& event ) {
  
  // Set RunType
  uint16_t run = static_cast<uint16_t>( buffer[10] & 0xFFFF );
  runType_ = SiStripEnumsAndStrings::runType(run);

  // Set spill number
  spillNumber_ = buffer[0];

  // Set number of DataSenders
  nDataSenders_ = buffer[20];

  // Set FED readout mode
  if      ( buffer[15] == 0 ) { fedReadoutMode_ = sistrip::FED_SCOPE_MODE; }
  else if ( buffer[15] == 1 ) { fedReadoutMode_ = sistrip::FED_VIRGIN_RAW; }
  else if ( buffer[15] == 2 ) { fedReadoutMode_ = sistrip::FED_PROC_RAW; }
  else if ( buffer[15] == 3 ) { fedReadoutMode_ = sistrip::FED_ZERO_SUPPR; }
  else if ( buffer[15] == 4 ) { fedReadoutMode_ = sistrip::FED_ZERO_SUPPR_LITE; }
  else { fedReadoutMode_ = sistrip::UNKNOWN_FED_READOUT_MODE; }
  
  // Set hardware parameters
  if ( runType_ == sistrip::CALIBRATION ||
       runType_ == sistrip::CALIBRATION_DECO ||
       runType_ == sistrip::CALIBRATION_SCAN ||
       runType_ == sistrip::CALIBRATION_SCAN_DECO ||
       runType_ == sistrip::APV_LATENCY ) { 

    params_[0] = buffer[11]; // latency
    params_[1] = buffer[12]; // cal_chan
    params_[2] = buffer[13]; // cal_sel
    params_[3] = buffer[15]; // isha
    params_[4] = buffer[16]; // vfs

  } else if ( runType_ == sistrip::OPTO_SCAN ) { 

    params_[0] = buffer[11]; // opto gain
    params_[1] = buffer[12]; // opto bias

  } else if ( runType_ == sistrip::APV_TIMING ||
	      runType_ == sistrip::FED_TIMING ) {
    params_[0] = buffer[11]; // pll coarse delay
    params_[1] = buffer[12]; // pll fine delay
    params_[2] = buffer[13]; // ttcrx delay
  } else if ( runType_ == sistrip::FINE_DELAY || //@@ layer
	      runType_ == sistrip::FINE_DELAY_PLL ||
	      runType_ == sistrip::FINE_DELAY_TTC ) { 
    params_[0] = buffer[11]; // pll coarse delay
    params_[1] = buffer[12]; // pll fine delay
    params_[2] = buffer[13]; // ttcrx delay
    params_[3] = buffer[14]; // layer (private format: DDSSLLLL, det, side, layer)

  } else if ( runType_ == sistrip::FAST_CABLING ) { 

    params_[0] = buffer[11]; // bin number
    params_[1] = buffer[12]; // fec instance
    params_[2] = buffer[13]; // fec ip
    params_[3] = buffer[14]; // dcu hard id 

  } else if ( runType_ == sistrip::FED_CABLING ||
	      runType_ == sistrip::QUITE_FAST_CABLING ) { 

    if ( runType_ == sistrip::QUITE_FAST_CABLING ) { 

      uint16_t ii = 0;
      bool found = false;
      while ( !found && ii < 20 ) {
	uint32_t dcu = buffer[21+3*ii];
	uint32_t key = buffer[21+3*ii+1];
	uint32_t evt = buffer[21+3*ii+2];
	if ( evt == event ) {
	  params_[0] = key; // device id
	  params_[1] = 0;   // process id
	  params_[2] = 0;   // process ip
	  params_[3] = dcu; // dcu hard id
	  found = true;
	} 
	ii++;
      }
      if ( !found ) { 
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[SiStripEventSummary::" << __func__ << "]"
	     << " Did not find DeviceId/DCUid for event " 
	     << event << "!";
	  edm::LogWarning(mlDigis_) << ss.str();
	}
	params_[0] = 0; 
	params_[1] = 0; 
	params_[2] = 0;
	params_[3] = 0;
      } else {
	if ( edm::isDebugEnabled() ) {
	  std::stringstream ss;
	  ss << "[SiStripEventSummary::" << __func__ << "]"
	     << " Found DeviceId/DCUid for event " 
	     << event << ": 0x" 
	     << std::hex << std::setw(8) << std::setfill('0') << params_[0] << std::dec
	     << "/0x"
	     << std::hex << std::setw(8) << std::setfill('0') << params_[3] << std::dec;
	  LogTrace(mlDigis_) << ss.str();
	}
      }

    } else {

      params_[0] = buffer[11]; // device id
      params_[1] = buffer[12]; // process id
      params_[2] = buffer[13]; // process ip
      params_[3] = buffer[14]; // dcu hard id

    }

  } else if ( runType_ == sistrip::VPSP_SCAN ) { 

    params_[0] = buffer[11]; // vpsp value
    params_[1] = buffer[12]; // ccu channel

  } else if ( runType_ == sistrip::DAQ_SCOPE_MODE ) { 

    // nothing interesting!

  } else if (  runType_ == sistrip::PHYSICS ||
	       runType_ == sistrip::PHYSICS_ZS ||
	       runType_ == sistrip::PEDESTALS ) { 

    //@@ do anything?...

  } else { 
    
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlDigis_)
	<< "[SiStripEventSummary::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< runType_;
    }

  }

}

// -----------------------------------------------------------------------------
//

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::commissioningInfo( const uint32_t& daq1,
					     const uint32_t& daq2 ) {
  
  // Extract if commissioning info are valid or not 
  uint16_t temp = static_cast<uint16_t>( (daq1>>8)&0x3 );
  if      ( temp == uint16_t(1) ) { valid_ = true; }
  else if ( temp == uint16_t(2) ) { valid_ = false; }
  else if ( temp == uint16_t(3) && 
	    daq1 == sistrip::invalid32_ ) {
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlDigis_)
	<< "[SiStripEventSummary::" << __func__ << "]"
	<< " DAQ register contents set to invalid: 0x"
	<< std::hex 
	<< std::setw(8) << std::setfill('0') << daq1 
	<< std::dec;
    }
    valid_ = false;
  } else {
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlDigis_)
	<< "[SiStripEventSummary::" << __func__ << "]"
	<< " Unexpected bit pattern set in DAQ1: 0x"
	<< std::hex 
	<< std::setw(8) << std::setfill('0') << daq1 
	<< std::dec;
    }
    valid_ = false;
  }
  
  // Set RunType
  uint16_t run = static_cast<uint16_t>( daq1&0xFF );
  runType_ = SiStripEnumsAndStrings::runType(run);
  
  // Set hardware parameters
  if        ( runType_ == sistrip::PHYSICS ) { 
  } else if ( runType_ == sistrip::PHYSICS_ZS ) { 
  } else if ( runType_ == sistrip::PEDESTALS ) { 
  } else if ( runType_ == sistrip::DAQ_SCOPE_MODE ) { 
  } else if ( runType_ == sistrip::CALIBRATION ) { 
  } else if ( runType_ == sistrip::CALIBRATION_DECO ) { 
    params_[0] = (daq2>>8)&0xFF; // latency
    params_[1] = (daq2>>4)&0x0F; // cal_chan
    params_[2] = (daq2>>0)&0x0F; // cal_sel
  } else if ( runType_ == sistrip::CALIBRATION_SCAN ) { 
    params_[0] = (daq2>>8)&0xFF; // latency
    params_[1] = (daq2>>4)&0x0F; // cal_chan
    params_[2] = (daq2>>0)&0x0F; // cal_sel
  } else if ( runType_ == sistrip::OPTO_SCAN ) { 
    params_[0] = (daq2>>8)&0xFF; // opto gain
    params_[1] = (daq2>>0)&0xFF; // opto bias
  } else if ( runType_ == sistrip::APV_TIMING ) { 
    params_[1] = (daq2>>0)&0xFF; // pll fine delay
  } else if ( runType_ == sistrip::APV_LATENCY ) { 
    params_[0] = (daq2>>0)&0xFF; // latency
  } else if ( runType_ == sistrip::FINE_DELAY_PLL ) { 
  } else if ( runType_ == sistrip::FINE_DELAY_TTC ) { 
  } else if ( runType_ == sistrip::FINE_DELAY ) { //@@ layer
    params_[2] = (daq2>>0 )&0xFFFF; // ttcrx delay
    params_[0] = params_[2]/25;   // pll coarse delay
    params_[1] = uint32_t((params_[2]%25)*24./25.); // pll fine delay
    params_[3] = (daq2>>0)&0xFFFF0000; // layer (private format: DDSSLLLL (det, side, layer)
  } else if ( runType_ == sistrip::FED_TIMING ) { 
    params_[1] = (daq2>>0)&0xFF; // pll fine delay
  } else if ( runType_ == sistrip::VPSP_SCAN ) { 
    params_[0] = (daq2>>8)&0xFF; // vpsp value
    params_[1] = (daq2>>0)&0xFF; // ccu channel
  } else if ( runType_ == sistrip::FED_CABLING ) { 
  } else if ( runType_ == sistrip::QUITE_FAST_CABLING ) { 
  } else if ( runType_ == sistrip::FAST_CABLING ) { 
    params_[0] = (daq2>>0)&0xFF; // key
  } else { 
    if ( edm::isDebugEnabled() ) {
      edm::LogWarning(mlDigis_)
	<< "[SiStripEventSummary::" << __func__ << "]"
	<< " Unexpected commissioning task: "
	<< runType_;
    }
  }

}

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::fedReadoutMode( const uint16_t& mode ) {
  if      ( mode ==  1 ) { fedReadoutMode_ = sistrip::FED_SCOPE_MODE; }
  else if ( mode ==  2 ) { fedReadoutMode_ = sistrip::FED_VIRGIN_RAW; }
  else if ( mode ==  6 ) { fedReadoutMode_ = sistrip::FED_PROC_RAW; }
  else if ( mode == 10 ) { fedReadoutMode_ = sistrip::FED_ZERO_SUPPR; }
  else if ( mode == 12 ) { fedReadoutMode_ = sistrip::FED_ZERO_SUPPR_LITE; }
  else { fedReadoutMode_ = sistrip::UNKNOWN_FED_READOUT_MODE; }
}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripEventSummary& input ) {
  return os << "[SiStripEventSummary::" << __func__ << "]" << std::endl
	    << " isSet                : " << std::boolalpha << input.isSet() << std::noboolalpha << std::endl
	    << " Trigger FED id       : " << input.triggerFed() << std::endl
	    << " isValid              : " << std::boolalpha << input.valid() << std::noboolalpha << std::endl
	    << " Run type             : " << SiStripEnumsAndStrings::runType( input.runType() ) << std::endl
	    << " Event number         : " << input.event() << std::endl 
	    << " Bunch crossing       : " << input.bx() << std::endl
	    << " FED readout mode     : " << SiStripEnumsAndStrings::fedReadoutMode( input.fedReadoutMode() ) << std::endl
	    << " APV readout mode     : " << SiStripEnumsAndStrings::apvReadoutMode( input.apvReadoutMode() ) << std::endl
	    << " Commissioning params : "
	    << input.params()[0] << ", " 
	    << input.params()[1] << ", " 
	    << input.params()[2] << ", "  
	    << input.params()[3] 
	    << std::endl;
}
