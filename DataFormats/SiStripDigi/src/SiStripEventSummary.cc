#include "DataFormats/SiStripDigi/interface/SiStripEventSummary.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace sistrip;

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::print( stringstream& ss ) const {
  ss << "SiStripEventSummary:" << endl
     << " Event: " << event_ << endl 
     << " BX: " << bx_ << endl
     << " CommissioningTask: " << task_ << endl
     << " CommissioningParams: "
     << hex 
     << " 0x" << setw(8) << setfill('0') << param0_ 
     << " 0x" << setw(8) << setfill('0') << param1_ 
     << " 0x" << setw(8) << setfill('0') << param2_ 
     << " 0x" << setw(8) << setfill('0') << param3_ 
     << dec << endl
     << " FedReadoutMode" << fedReadoutMode_ << endl;
}

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::check() const {
  if ( ( task_ == sistrip::UNDEFINED_TASK ||
	 task_ == sistrip::UNKNOWN_TASK ) &&
       ( !param0_ && !param1_ && 
	 !param2_ && !param3_ ) ) {
    stringstream ss;
    ss << "[SiStripEventSummary::" << __func__ << "]"
       << " Unknown/undefined commissioning task and NULL parameter values!"
       << " It may be that the 'trigger FED' information was not found!"; 
    edm::LogWarning(mlDigis_) << ss.str();
  }
}

// -----------------------------------------------------------------------------
//
void SiStripEventSummary::commissioningInfo( const uint32_t* const buffer ) {
  
  // Set commissioning task
  if      ( buffer[10] == 11 ||
	    buffer[10] == 13 ) { task_ = sistrip::FED_CABLING; }
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
    stringstream ss;
    ss << "[SiStripEventSummary::" << __func__ << "]"
       << " Unknown commissioning task: " 
       << buffer[10];
    edm::LogWarning(mlDigis_) << ss.str();
  }
  
  // Set FED readout mode
  if      ( buffer[15] == 0 ) { fedReadoutMode_ = sistrip::SCOPE_MODE; }
  else if ( buffer[15] == 1 ) { fedReadoutMode_ = sistrip::VIRGIN_RAW; }
  else if ( buffer[15] == 2 ) { fedReadoutMode_ = sistrip::PROC_RAW; }
  else if ( buffer[15] == 3 ) { fedReadoutMode_ = sistrip::ZERO_SUPPR; }
  else {
    fedReadoutMode_ = sistrip::UNKNOWN_FED_READOUT_MODE;
    stringstream ss;
    ss << "[SiStripEventSummary::" << __func__ << "]"
       << "[SiStripEventSummary::commissioningInfo]"
       << " Unknown FED readout mode: " 
       << buffer[15];
    edm::LogWarning(mlDigis_) << ss.str();
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
    
    edm::LogWarning(mlDigis_)
      << "[SiStripEventSummary::" << __func__ << "]"
      << " Unknown commissioning task: "
      << buffer[10];

  }

}
