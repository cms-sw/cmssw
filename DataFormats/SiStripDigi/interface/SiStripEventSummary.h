#ifndef DataFormats_SiStripEventSummary_SiStripEventSummary_H
#define DataFormats_SiStripEventSummary_SiStripEventSummary_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "boost/cstdint.hpp"
#include <sstream>

/**  
     @brief A container class for generic event-related info.
*/
class SiStripEventSummary {

 public:
  
  /** Default constructor. */
  SiStripEventSummary() : 
    event_(0), bx_(0),
    task_(sistrip::UNDEFINED_TASK), 
    fedReadoutMode_(sistrip::UNDEFINED_FED_READOUT_MODE),
    param0_(0), param1_(0), param2_(0), param3_(0),
    apveAddress_(0),
    nApvsInSync_(0),
    nApvsOutOfSync_(0),
    nApvsErrors_(0) {;}
  
  /** Default destructor. */
  ~SiStripEventSummary() {;}
  
  // ----- Event-related -----
  
  /** Sets event number. */
  inline void event( const uint32_t& event ) { event_ = event; }
  /** Sets bunch crossing number. */
  inline void bx( const uint32_t& bx ) { bx_ = bx; }

  inline const uint32_t& event() const { return event_; }
  inline const uint32_t& bx() const { return bx_; }

  /** Some debug */
  void print( std::stringstream& ) const;
  void check() const;

  // ----- Commissioning information -----

  /** Sets commissioning-related information. */
  void commissioningInfo( const uint32_t* const buffer );
  
  /** Returns commissioning task. */ 
  inline const sistrip::Task& task() const { return task_; }
  /** Returns pair of PLL coarse and fine delay settings. */
  inline std::pair<uint32_t,uint32_t> pll() { return std::pair<uint32_t,uint32_t>(param0_,param1_); }
  inline const uint32_t& latency() const { return param0_; }
  /** Returns pair of APV calibration chan and select. */
  inline std::pair<uint32_t,uint32_t> calibration() { return std::pair<uint32_t,uint32_t>(param1_,param2_); }
  /** Returns TTCrx delay setting. */
  inline const uint32_t& ttcrx() const { return param0_; }
  /** Returns APV VPSP setting. */
  inline const uint32_t& vpsp() const { return param0_; }
  /** Returns pair of LLD gain and bias settings. */
  inline std::pair<uint32_t,uint32_t> opto() { return std::pair<uint32_t,uint32_t>(param0_,param1_); }
  /** Returns device id. */
  inline const uint32_t& deviceId() const { return param0_; }
  /** Returns process id. */
  inline const uint32_t& processId() const { return param1_; }
  /** Returns process IP address. */
  inline const uint32_t& processIp() const { return param2_; }
  /** Returns DCU id. */
  inline const uint32_t& dcuId() const { return param3_; }
  
  // ----- FED-related -----

  inline const sistrip::FedReadoutMode& fedReadoutMode() const { return fedReadoutMode_; }

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
  sistrip::Task task_;
  /** FED readout mode (ZS, VR, PR, SM). */
  sistrip::FedReadoutMode fedReadoutMode_;
  
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



