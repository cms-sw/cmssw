#ifndef DataFormats_SiStripEventSummary_SiStripEventSummary_H
#define DataFormats_SiStripEventSummary_SiStripEventSummary_H

#include "boost/cstdint.hpp"

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
      multi mode operation, relative apv synchronisation using fed
      delays, connection of apv pairs to fed channels, apv baseline
      scan, scope mode readout (debugging purposes), unknown run
      type. */ 
  enum Task { PHYSICS = 1, PEDESTALS = 2, PULSESHAPE_PEAK = 3, PULSESHAPE_DECON = 33,
              OPTO_SCAN = 4, APV_TIMING = 5, APV_LATENCY = 6, PLL_DELAY = 7,
              TTC_DELAY = 8, APV_MULTI = 10, FED_TIMING = 12, FED_CABLING = 13, 
	      VPSP_SCAN = 14, SCOPE_MODE_READOUT = 66, UNKNOWN_TASK = 0 };
  
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
  void commissioningInfo( const uint32_t* const buffer );
  
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

#endif // DataFormats_SiStripEventSummary_SiStripEventSummary_H



