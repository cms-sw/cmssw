#ifndef DataFormats_SiStripEventSummary_SiStripEventSummary_H
#define DataFormats_SiStripEventSummary_SiStripEventSummary_H

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "boost/cstdint.hpp"
#include <string>
#include <vector>

using namespace std;

/**  
     @brief A container class for generic event-related info.
*/
class SiStripEventSummary {

 public:
  
  /** Default constructor. */
  SiStripEventSummary() : 
    event_(0), bx_(0),
    task_(SiStripEventSummary::UNKNOWN_TASK), 
    fedReadoutMode_(SiStripEventSummary::UNKNOWN_FED_MODE),
    param0_(0), param1_(0), param2_(0), param3_(0),
    apveAddress_(0),
    nApvsInSync_(0),
    nApvsOutOfSync_(0),
    nApvsErrors_(0) {;}
  
  /** Default destructor. */
  ~SiStripEventSummary() {;}
  
  // ----- Enumerated types -----

  /** Commissioning tasks: physics run, calibration run, pulse shape
      tuning, pulse shape tuning, laser driver bias and gain, relative
      apv synchronisation, coarse (25ns) apv latency scan for beam,
      fine (1ns) pll scan for beam, fine (1ns) ttc scan for beam,
      multi mode operation, connection of apv pairs to fed channels
      (obsolete), relative apv synchronisation using fed delays,
      connection of apv pairs to fed channels, apv baseline scan,
      scope mode readout (debugging purposes), unknown run type. */ 
  enum Task { PHYSICS = 1, PEDESTALS = 2, PULSESHAPE_PEAK = 3, PULSESHAPE_DECON = 33,
              OPTO_SCAN = 4, APV_TIMING = 5, APV_LATENCY = 6, PLL_DELAY = 7,
              TTC_DELAY = 8, APV_MULTI = 10, CONNECTION = 11, FED_TIMING = 12,
              BARE_CONNECTION = 13, VPSP_SCAN = 14, SCOPE_MODE_READOUT = 66,
              UNKNOWN_TASK = 0 };
  
  /** */
  enum FedReadoutMode { SCOPE_MODE = 0, VIRGIN_RAW = 1, PROC_RAW = 2, ZERO_SUPPR = 3, UNKNOWN_FED_MODE = 999 };
  
  // ----- Event-related -----

  /** Sets event number. */
  inline void event( const uint32_t& event ) { event_ = event; }
  /** Sets bunch crossing number. */
  inline void bx( const uint32_t& bx ) { bx_ = bx; }

  inline const uint32_t& event() const { return event_; }
  inline const uint32_t& bx() const { return bx_; }

  // ----- Commissioning information -----

  /** Sets commissioning-related information. */
  inline void commissioningInfo( const uint32_t* const buffer );
  /** Returns commissioning task. */ 
  inline const SiStripEventSummary::Task& task() const { return task_; }
  /** Returns pair of PLL coarse and fine delay settings. */
  inline pair<uint32_t,uint32_t> pll() { return pair<uint32_t,uint32_t>(param0_,param1_); }
  inline const uint32_t& latency() const { return param0_; }
  /** Returns pair of APV calibration chan and select. */
  inline pair<uint32_t,uint32_t> calibration() { return pair<uint32_t,uint32_t>(param1_,param2_); }
  /** Returns TTCrx delay setting. */
  inline const uint32_t& ttcrx() const { return param0_; }
  /** Returns APV VPSP setting. */
  inline const uint32_t& vpsp() const { return param0_; }
  /** Returns pair of LLD gain and bias settings. */
  inline pair<uint32_t,uint32_t> opto() { return pair<uint32_t,uint32_t>(param0_,param1_); }
  /** Returns LLD device id. */
  inline const uint32_t& deviceId() const { return param0_; }
  /** Returns process id. */
  inline const uint32_t& processId() const { return param1_; }
  /** Returns process IP address. */
  inline const uint32_t& processIp() const { return param2_; }
  /** Returns DCU id. */
  inline const uint32_t& dcuId() const { return param3_; }
  
  // ----- FED-related -----

  inline const SiStripEventSummary::FedReadoutMode& fedReadoutMode() const { return fedReadoutMode_; }

  // ----- APV-related ----- 

  inline const uint16_t& apveAddress() const { return apveAddress_; }
  inline const uint32_t& nApvsInSync() const { return nApvsInSync_; }
  inline const uint32_t& nApvsOutOfSync() const { return nApvsOutOfSync_; }
  inline const uint32_t& nApvsErrors() const { return nApvsErrors_; }
  inline void apveAddress( uint16_t& addr ) { apveAddress_ = addr; }
  inline void nApvsInSync( uint32_t& in_sync ) { nApvsInSync_ = in_sync; }
  inline void nApvsOutOfSync( uint32_t& out_of_sync ) { nApvsOutOfSync_ = out_of_sync; }
  inline void nApvsErrors( uint32_t& errors ) { nApvsErrors_ = errors; }
  
 private:

  uint32_t event_;
  uint32_t bx_;

  /** Commissioning task */
  Task task_;
  /** FED readout mode (ZS, VR, PR, SM). */
  FedReadoutMode fedReadoutMode_;
  
  // Parameters relating to commissioning tasks and used by analysis.
  uint32_t param0_;
  uint32_t param1_;
  uint32_t param2_;
  uint32_t param3_;

  // APV synchronization and errors
  uint16_t apveAddress_;
  uint32_t nApvsInSync_;
  uint32_t nApvsOutOfSync_;
  uint32_t nApvsErrors_;
  
};

// ----- inline methods -----

void SiStripEventSummary::commissioningInfo( const uint32_t* const buffer ) {

  // Set commissioning task
  task_ = static_cast<SiStripEventSummary::Task>( buffer[10] );

  // Set FED readout mode
  if ( buffer[15] == 0 || 
       buffer[15] == 1 || 
       buffer[15] == 2 || 
       buffer[15] == 3 ) {
    fedReadoutMode_ = static_cast<SiStripEventSummary::FedReadoutMode>( buffer[15] );
  } else {
    fedReadoutMode_ = SiStripEventSummary::UNKNOWN_FED_MODE;
    edm::LogError("Commissioning") << "[SiStripEventSummary::commissioning]"
				   << " Unknown FED readout mode! " 
				   << buffer[15];
  }
  
  // Set hardware parameters
  if ( buffer[10] == 3  ||
       buffer[10] == 33 ||
       buffer[10] == 6  || // buffer[10] == 16 || 
       buffer[10] == 26 ) { 
    // Calibration or latency
    param0_ = buffer[11]; // latency
    param1_ = buffer[12]; // cal_chan
    param2_ = buffer[13]; // cal_sel
  } else if ( buffer[10] == 4 ) { 
    // Laser drivers 
    param0_ = buffer[11]; // opto gain
    param1_ = buffer[12]; // opto bias
  } else if ( buffer[10] == 7 ||
	      buffer[10] == 8 ||
	      buffer[10] == 5 ||
	      buffer[10] == 12 ) { 
    // Synchronisation and delay scans
    param0_ = buffer[11]; // pll coarse delay
    param1_ = buffer[12]; // pll fine delay
    param2_ = buffer[13]; // ttcrx delay
  } else if ( buffer[10] == 11 || 
	      buffer[10] == 13 ) { 
    // Connection loops 
    param0_ = buffer[11]; // device id
    param1_ = buffer[12]; // process id
    param2_ = buffer[13]; // process ip
    param3_ = buffer[14]; // dcu hard id
  } else if ( buffer[10] == 14 ) { 
    // VPSP
    param0_ = buffer[11]; // vpsp
  } else if (  buffer[10] == 1 ||
	       buffer[10] == 2 ) { 
    //@@ do anything?...
  } else {
    // Unknown commissioning task
    task_ = static_cast<SiStripEventSummary::Task>( 0 );
    edm::LogError("RawToDigi") << "[SiStripEventSummary::commissioning]"
			       << " Unknown commissioning task! "
			       << buffer[10];
  }
  
}

#endif // DataFormats_SiStripEventSummary_SiStripEventSummary_H



