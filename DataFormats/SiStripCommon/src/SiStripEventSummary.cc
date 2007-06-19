// Last commit: $Id: SiStripEventSummary.cc,v 1.2 2007/05/24 15:27:33 bainbrid Exp $

#include "DataFormats/SiStripCommon/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/SiStripEnumsAndStrings.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace sistrip;

// -----------------------------------------------------------------------------
//
SiStripEventSummary::SiStripEventSummary() : 
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
  params_(4,0)
{;}

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::commissioningInfo( const uint32_t* const buffer,
					     const uint32_t& event ) {

  // Set RunType
  sistrip::RunType tmp = static_cast<sistrip::RunType>( buffer[10] );
  if ( buffer[10] == 11 || buffer[10] == 16 ) { tmp = sistrip::FED_CABLING; }
  std::string run_type = SiStripEnumsAndStrings::runType(tmp);
  runType_ = SiStripEnumsAndStrings::runType(run_type);

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
  if ( buffer[10] == 3  ||
       buffer[10] == 33 ||
       buffer[10] == 6  || 
       buffer[10] == 26 ) { // Calibration or latency

    params_[0] = buffer[11]; // latency
    params_[1] = buffer[12]; // cal_chan
    params_[2] = buffer[13]; // cal_sel

  } else if ( buffer[10] == 4 ) { // Laser driver tuning

    params_[0] = buffer[11]; // opto gain
    params_[1] = buffer[12]; // opto bias

  } else if ( buffer[10] == 7 ||
	      buffer[10] == 8 ||
	      buffer[10] == 5 ||
	      buffer[10] == 12 ) { // Synchronisation and delay scans

    params_[0] = buffer[11]; // pll coarse delay
    params_[1] = buffer[12]; // pll fine delay
    params_[2] = buffer[13]; // ttcrx delay

  } else if ( buffer[10] == 21 ) { // "Very fast" connection 

    params_[0] = event-1; // buffer[11]; // bin number
    params_[1] = buffer[12]; // fec instance
    params_[2] = buffer[13]; // fec ip
    params_[3] = buffer[14]; // dcu hard id 

  } else if ( buffer[10] == 11 ||
	      buffer[10] == 13 ||
	      buffer[10] == 16 ) { // Connection loop 

    if ( buffer[10] == 16 ) { // If fast connection

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
      std::stringstream ss;
      ss << "[SiStripEventSummary::" << __func__ << "]";
      if ( !found ) { 
	ss << " Did not find DeviceId/DCUid for event " 
	   << event << "!";
	edm::LogWarning(mlDigis_) << ss.str();
	params_[0] = 0; 
	params_[1] = 0; 
	params_[2] = 0;
	params_[3] = 0;
      } else {
	ss << " Found DeviceId/DCUid for event " 
	   << event << ": 0x" 
	   << std::hex << std::setw(8) << std::setfill('0') << params_[0] << std::dec
	   << "/0x"
	   << std::hex << std::setw(8) << std::setfill('0') << params_[3] << std::dec;
	LogTrace(mlDigis_) << ss.str();
      }

    } else { // If not fast connection

      params_[0] = buffer[11]; // device id
      params_[1] = buffer[12]; // process id
      params_[2] = buffer[13]; // process ip
      params_[3] = buffer[14]; // dcu hard id

    }

  } else if ( buffer[10] == 14 ) { // VPSP scan

    params_[0] = buffer[11]; // vpsp value
    params_[1] = buffer[12]; // ccu channel (I2C of module)

  } else if ( buffer[10] == 15 ) { // DAQ scope mode readout

    // nothing interesting!

  } else if (  buffer[10] == 1 ||
	       buffer[10] == 2 ) { // Physics and pedestals

    //@@ do anything?...

  } else { // Unknown commissioning task
    
    edm::LogWarning(mlDigis_)
      << "[SiStripEventSummary::" << __func__ << "]"
      << " Unknown commissioning task: "
      << buffer[10];

  }

}

// -----------------------------------------------------------------------------
//
std::ostream& operator<< ( std::ostream& os, const SiStripEventSummary& input ) {
  return os << "[SiStripEventSummary::" << __func__ << "]" << std::endl
	    << " Run type             : " << input.runType() << std::endl
	    << " Event number         : " << input.event() << std::endl 
	    << " Bunch crossing       : " << input.bx() << std::endl
	    << " FED readout mode     : " << input.fedReadoutMode() << std::endl
	    << " APV readout mode     : " << input.apvReadoutMode() << std::endl
	    << " Commissioning params : "
	    << std::hex 
	    << " 0x" << std::setw(8) << std::setfill('0') << input.params()[0] 
	    << ", 0x" << std::setw(8) << std::setfill('0') << input.params()[1] 
	    << ", 0x" << std::setw(8) << std::setfill('0') << input.params()[2] 
	    << ", 0x" << std::setw(8) << std::setfill('0') << input.params()[3] 
	    << std::dec
	    << std::endl;
}
