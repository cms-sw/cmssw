#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::commissioningInfo( const uint32_t* const buffer ) {

  // Set commissioning task
  if      ( buffer[10] == 13 ) { task_ = sistrip::FED_CABLING; }
  else if ( buffer[10] ==  5 ) { task_ = sistrip::APV_TIMING; }
  else if ( buffer[10] == 12 ) { task_ = sistrip::FED_TIMING; }
  else if ( buffer[10] ==  4 ) { task_ = sistrip::OPTO_SCAN; }
  else if ( buffer[10] == 14 ) { task_ = sistrip::VPSP_SCAN; }
  else if ( buffer[10] ==  2 ) { task_ = sistrip::PEDESTALS; }
  else if ( buffer[10] ==  6 ) { task_ = sistrip::APV_LATENCY; }
  else if ( buffer[10] ==  1 ) { task_ = sistrip::PHYSICS; }
  else if ( buffer[10] ==  0 ) { task_ = sistrip::UNDEFINED_TASK; }
  else {
    task_ = sistrip::UNKNOWN_TASK;
    edm::LogError("Commissioning") << "[SiStripEventSummary::commissioningInfo]"
				   << " Unknown commissioning task! " 
				   << buffer[10];
  }
  
  // Set FED readout mode
  if      ( buffer[15] == 0 ) { fedReadoutMode_ = sistrip::SCOPE_MODE; }
  else if ( buffer[15] == 1 ) { fedReadoutMode_ = sistrip::VIRGIN_RAW; }
  else if ( buffer[15] == 2 ) { fedReadoutMode_ = sistrip::PROC_RAW; }
  else if ( buffer[15] == 3 ) { fedReadoutMode_ = sistrip::ZERO_SUPPR; }
  else {
    fedReadoutMode_ = sistrip::UNKNOWN_FED_READOUT_MODE;
    edm::LogError("Commissioning") << "[SiStripEventSummary::commissioningInfo]"
				   << " Unknown FED readout mode! " 
				   << buffer[15];
  }
  
  // Set hardware parameters
  if ( buffer[10] == 3  ||
       buffer[10] == 33 ||
       buffer[10] == 6  || 
       // buffer[10] == 16 || 
       buffer[10] == 26 ) { // Calibration or latency

    param0_ = buffer[11]; // latency
    param1_ = buffer[12]; // cal_chan
    param2_ = buffer[13]; // cal_sel

  } else if ( buffer[10] == 4 ) { // Laser driver tuning

    param0_ = buffer[11]; // opto gain
    param1_ = buffer[12]; // opto bias

  } else if ( buffer[10] == 7 ||
	      buffer[10] == 8 ||
	      buffer[10] == 5 ||
	      buffer[10] == 12 ) { // Synchronisation and delay scans

    param0_ = buffer[11]; // pll coarse delay
    param1_ = buffer[12]; // pll fine delay
    param2_ = buffer[13]; // ttcrx delay

  } else if ( buffer[10] == 11 ||
	      buffer[10] == 13 ) { // Connection loop 

    //@@ TEMPORARY!
    if ( buffer[10] == 11 ) { task_ = static_cast<sistrip::Task>( 13 ); }

    param0_ = buffer[11]; // device id
    param1_ = buffer[12]; // process id
    param2_ = buffer[13]; // process ip
    param3_ = buffer[14]; // dcu hard id

  } else if ( buffer[10] == 14 ) { // VPSP scan

    param0_ = buffer[11]; // vpsp

  } else if (  buffer[10] == 1 ||
	       buffer[10] == 2 ) { // Physics and pedestals

    //@@ do anything?...

  } else { // Unknown commissioning task
    
    //task_ = static_cast<sistrip::Task>( 0 );
    edm::LogError("RawToDigi") << "[SiStripEventSummary::commissioningInfo]"
			       << " Unknown commissioning task! "
			       << buffer[10];
  }
  
}
