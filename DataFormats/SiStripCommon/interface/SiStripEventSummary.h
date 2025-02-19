// Last commit: $Id: SiStripEventSummary.h,v 1.10 2008/06/09 12:55:03 delaer Exp $

#ifndef DataFormats_SiStripEventSummary_SiStripEventSummary_H
#define DataFormats_SiStripEventSummary_SiStripEventSummary_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <sstream>
#include <vector>

class SiStripEventSummary;

/** Debug info for SiStripFedKey class. */
std::ostream& operator<< ( std::ostream&, const SiStripEventSummary& );

/**
   @class SiStripFedKey
   @author R.Bainbridge
   
   @brief A container class for generic run and event-related info,
   information required by the commissioning analyses (such as
   hardware parameters), event quality flags, error conditions, etc...
*/
class SiStripEventSummary {

 public:

  // ---------- Constructors, destructors ----------
  
  /** Default constructor. */
  SiStripEventSummary();
  
  /** Default destructor. */
  ~SiStripEventSummary() {;}
  
  // ---------- General information ----------
  
  /** Identifies if commissioning info is valid or not (when using
      global trigger to understand if HW config is being changed). */
  inline bool valid() const;
  
  /** Identifier of trigger FED (null value means not found). */
  inline uint16_t triggerFed() const;

  /** Flag to identify if commissioning info already set. */
  inline bool isSet() const;
  
  // ---------- Run and event-related info ----------
  
  /** Returns run type. */ 
  inline const sistrip::RunType& runType() const;

  /** Returns event number. */
  inline const uint32_t& event() const;

  /** Returns bunch crossing number. */
  inline const uint32_t& bx() const;
  
  // ---------- Hardware-related info ----------

  /** Returns FED readout mode. */
  inline const sistrip::FedReadoutMode& fedReadoutMode() const;

  /** Returns APV readout mode. */
  inline const sistrip::ApvReadoutMode& apvReadoutMode() const;

  /** Returns APVE golden pipeline address. */
  inline const uint16_t& apveAddress() const;

  /** Returns number of APVs that are synchronized. */
  inline const uint32_t& nApvsInSync() const;

  /** Returns number of APVs that are not synchronized. */
  inline const uint32_t& nApvsOutOfSync() const;

  /** Returns number of APVs with error flags set. */
  inline const uint32_t& nApvsErrors() const;
  
  // ---------- Commissioning info ----------

  /** Indicates whether all params are null or not. */
  inline bool nullParams() const; 

  /** Returns values of all params. */
  inline const std::vector<uint32_t>& params() const; 
  
  /** Returns bin number for very fast connection. */
  inline const uint32_t& binNumber() const; 

  /** Returns PLL coarse delay setting. */
  inline const uint32_t& pllCoarse() const; 
  
  /** Returns PLL fine delay setting. */
  inline const uint32_t& pllFine() const; 

  /** Returns APV latency setting. */
  inline const uint32_t& latency() const;
  
  /** Returns APV calibration channels (CDRV). */
  inline const uint32_t& calChan() const;
  
  /** Returns APV calibration timing (CSEL). */
  inline const uint32_t& calSel() const;
  
  /** Returns APV ISHA. */
  inline const uint32_t& isha() const;
  
  /** Returns APV VFS. */
  inline const uint32_t& vfs() const;
  
  /** Returns TTCrx delay setting. */
  inline const uint32_t& ttcrx() const;
  
  /** Returns VPSP setting. */
  inline const uint32_t& vpsp() const;
  
  /** Returns CCU channel of module being tuned for VPSP. */
  inline const uint32_t& vpspCcuChan() const;
  
  /** Returns LLD gain setting. */
  inline const uint32_t& lldGain() const;

  /** Returns LLD bias setting. */
  inline const uint32_t& lldBias() const;
  
  /** Returns device id. */
  inline const uint32_t& deviceId() const;
  
  /** Returns process id. */
  inline const uint32_t& processId() const;
  
  /** Returns process IP address. */
  inline const uint32_t& processIp() const;
  
  /** Returns DCU id. */
  inline const uint32_t& dcuId() const;
  
  /** Returns the layer mask for fine delay runs */
  inline const uint32_t& layerScanned() const;

  // ---------- Setter methods ----------
  
  /** Sets commissioning-related information. */
  void commissioningInfo( const uint32_t* const buffer,
			  const uint32_t& event );

  /** Sets DAQ register information. */
  void commissioningInfo( const uint32_t& daq_register1,
			  const uint32_t& daq_register2 );
  
  /** Sets trigger FED number. */
  inline void triggerFed( const int16_t& );
  
  /** Sets FED readout mode. */
  void fedReadoutMode( const uint16_t& );
  
  /** Sets event number. */
  inline void event( const uint32_t& );

  /** Sets bunch crossing number. */
  inline void bx( const uint32_t& );

  inline void apveAddress( uint16_t& addr );
  inline void nApvsInSync( uint32_t& napvs_in_sync );
  inline void nApvsOutOfSync( uint32_t& napvs_out_of_sync );
  inline void nApvsErrors( uint32_t& napvs_with_errors );
  
 private:

  // ---------- General info ----------

  /** Flag to signify if trigger FED information is valid. */
  bool valid_;

  /** */
  uint16_t triggerFed_;


  // ---------- Run- and event-related info ----------
  
  /** Run type. */
  sistrip::RunType runType_;

  /** Event number. */
  uint32_t event_;
  
  /** Bunch crossing number. */
  uint32_t bx_;

  /** Spill number. */
  uint32_t spillNumber_;

  /** Number of DataSenders (c.f. ReadoutUnits). */
  uint32_t nDataSenders_;
  
  // ---------- Hardware-related info ----------

  /** FED readout mode. */
  sistrip::FedReadoutMode fedReadoutMode_;

  /** APV readout mode. */
  sistrip::ApvReadoutMode apvReadoutMode_;

  /** APVE golden pipeline address. */
  uint16_t apveAddress_;

  /** Number of APVs that are synchronized. */
  uint32_t nApvsInSync_;

  /** Number of APVs that are not synchronized. */
  uint32_t nApvsOutOfSync_;

  /** Number of APVs with error flags set. */
  uint32_t nApvsErrors_;

  /** Parameters related to commissioning analysis. */
  std::vector<uint32_t> params_;

};

// ---------- inline methods ----------

bool SiStripEventSummary::valid() const { return valid_; }
uint16_t SiStripEventSummary::triggerFed() const { return triggerFed_; }
bool SiStripEventSummary::isSet() const { return ( /* triggerFed_ > 0 && */ runType_ != sistrip::UNDEFINED_RUN_TYPE ); }  

const sistrip::RunType& SiStripEventSummary::runType() const { return runType_; }
const uint32_t& SiStripEventSummary::event() const { return event_; }
const uint32_t& SiStripEventSummary::bx() const { return bx_; }

const sistrip::FedReadoutMode& SiStripEventSummary::fedReadoutMode() const { return fedReadoutMode_; }
const sistrip::ApvReadoutMode& SiStripEventSummary::apvReadoutMode() const { return apvReadoutMode_; }

const uint16_t& SiStripEventSummary::apveAddress() const { return apveAddress_; }
const uint32_t& SiStripEventSummary::nApvsInSync() const { return nApvsInSync_; }
const uint32_t& SiStripEventSummary::nApvsOutOfSync() const { return nApvsOutOfSync_; }
const uint32_t& SiStripEventSummary::nApvsErrors() const { return nApvsErrors_; }

bool SiStripEventSummary::nullParams() const { return ( !params_[0] && !params_[1] && !params_[2] && !params_[3] ); } 
const std::vector<uint32_t>& SiStripEventSummary::params() const { return params_; } 
const uint32_t& SiStripEventSummary::binNumber() const { return params_[0]; }
const uint32_t& SiStripEventSummary::pllCoarse() const { return params_[0]; }
const uint32_t& SiStripEventSummary::pllFine() const { return params_[1]; }
const uint32_t& SiStripEventSummary::latency() const { return params_[0]; }
const uint32_t& SiStripEventSummary::calChan() const { return params_[1]; }
const uint32_t& SiStripEventSummary::calSel() const { return params_[2]; }
const uint32_t& SiStripEventSummary::isha() const { return params_[3]; }
const uint32_t& SiStripEventSummary::vfs() const { return params_[4]; }
const uint32_t& SiStripEventSummary::ttcrx() const { return params_[2]; }
const uint32_t& SiStripEventSummary::vpsp() const { return params_[0]; }
const uint32_t& SiStripEventSummary::vpspCcuChan() const { return params_[1]; }
const uint32_t& SiStripEventSummary::lldGain() const { return params_[0]; }
const uint32_t& SiStripEventSummary::lldBias() const { return params_[1]; }
const uint32_t& SiStripEventSummary::deviceId() const { return params_[0]; }
const uint32_t& SiStripEventSummary::processId() const { return params_[1]; }
const uint32_t& SiStripEventSummary::processIp() const { return params_[2]; }
const uint32_t& SiStripEventSummary::dcuId() const { return params_[3]; }
const uint32_t& SiStripEventSummary::layerScanned() const { return params_[3]; }

void SiStripEventSummary::triggerFed( const int16_t& fed ) { fed < 0 ? triggerFed_ = 0 : triggerFed_ = fed; }
void SiStripEventSummary::event( const uint32_t& event ) { event_ = event; }
void SiStripEventSummary::bx( const uint32_t& bx ) { bx_ = bx; }

void SiStripEventSummary::apveAddress( uint16_t& addr ) { apveAddress_ = addr; }
void SiStripEventSummary::nApvsInSync( uint32_t& napvs_in_sync ) { nApvsInSync_ = napvs_in_sync; }
void SiStripEventSummary::nApvsOutOfSync( uint32_t& napvs_out_of_sync ) { nApvsOutOfSync_ = napvs_out_of_sync; }
void SiStripEventSummary::nApvsErrors( uint32_t& napvs_with_errors ) { nApvsErrors_ = napvs_with_errors; }

#endif // DataFormats_SiStripEventSummary_SiStripEventSummary_H



