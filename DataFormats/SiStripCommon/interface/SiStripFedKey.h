#ifndef DataFormats_SiStripCommon_SiStripFedKey_H
#define DataFormats_SiStripCommon_SiStripFedKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFedKey {
  
 public:
  
  /** Simple container class that holds parameters that uniquely
      identify a FED channel within the readout system. */
  class Path { 
  public:
    uint16_t fedCrate_; // FED crate   [1-60]
    uint16_t fedSlot_;  // FED slot    [1-21]
    uint16_t fedId_;    // FED id      [0-1023]
    uint16_t fedCh_;    // FED channel [0-95]
    uint16_t fedApv_;   // APV         [1-2]
    uint16_t feUnit_;   // FE unit     [0-7]
    uint16_t feChan_;   // FE channel  [0-11]
    Path();
    Path( const uint16_t& fed_id,
	  const uint16_t& fed_ch,
	  const uint16_t& fed_apv = uint16_t(sistrip::invalid_) );
    Path( const uint16_t& fed_id,
	  const uint16_t& fe_unit,
	  const uint16_t& fe_chan,
	  const uint16_t& fed_apv );
  };
  
  /** Returns the parameters that uniquely identify a FED channel
      within the strip tracker readout system. */
  static Path path( uint32_t key );
  
  /** Creates a 32-bit key that uniquely identifies a FED channel
      within the strip tracker readout system. */
  static uint32_t key( const Path& );
  
  /** Creates a 32-bit key that uniquely identifies a FED channel
      (using FED id and FED channel) within the readout system. */
  static uint32_t key( const uint16_t& fed_id,
		       const uint16_t& fed_ch,
		       const uint16_t& fed_apv = uint16_t(sistrip::invalid_) );

  /** Creates a 32-bit key that uniquely identifies a FED channel
      (using FED id and FED unit/chan) within the readout system. */
  static uint32_t key( const uint16_t& fed_id,
		       const uint16_t& fe_unit,
		       const uint16_t& fe_chan,
		       const uint16_t& fed_apv );
  
 private:

  static const uint16_t fedCrateOffset_ = 26;
  static const uint16_t fedSlotOffset_  = 21;
  static const uint16_t fedIdOffset_    = 11;
  static const uint16_t feUnitOffset_   =  7;
  static const uint16_t feChanOffset_   =  3;
  static const uint16_t fedApvOffset_   =  0;

  static const uint16_t fedCrateMask_ = 0x03F;
  static const uint16_t fedSlotMask_  = 0x01F;
  static const uint16_t fedIdMask_    = 0x3FF;
  static const uint16_t feUnitMask_   = 0x00F;
  static const uint16_t feChanMask_   = 0x00F;
  static const uint16_t fedApvMask_   = 0x007;
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedKey::Path& );

#endif // DataFormats_SiStripCommon_SiStripFedKey_H


