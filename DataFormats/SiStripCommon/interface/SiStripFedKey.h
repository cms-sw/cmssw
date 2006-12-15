#ifndef DataFormats_SiStripCommon_SiStripFedKey_H
#define DataFormats_SiStripCommon_SiStripFedKey_H

#include "DataFormats/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFedKey {
  
 public:
  
  /** Container class holding parameters that identify a logical
      position within the readout structure to the level of an APV. */
  class Path { 
  public:
    uint16_t fedCrate_; // FED crate   [1-60]
    uint16_t fedSlot_;  // FED slot    [1-21]
    uint16_t fedId_;    // FED id      [1-1022]
    uint16_t fedCh_;    // FED channel [0-95]
    uint16_t fedApv_;   // APV         [1-2]
    uint16_t feUnit_;   // FE unit     [1-8]
    uint16_t feChan_;   // FE channel  [1-12]
    Path();
    Path( const uint16_t& fed_id,
	  const uint16_t& fed_ch,
	  const uint16_t& fed_apv = uint16_t(sistrip::invalid_) );
    Path( const uint16_t& fed_id,
	  const uint16_t& fe_unit,
	  const uint16_t& fe_chan,
	  const uint16_t& fed_apv );
    bool isEqual( const Path& ) const;
    bool isConsistent( const Path& ) const;
    bool isInvalid() const;
    bool isInvalid( const sistrip::Granularity& = sistrip::APV ) const;
  };
  
  // ---------- Returns 32-bit keys based on paths ----------
  
  /** Returns 32-bit key based on Path object. */
  static uint32_t key( const Path& );
  
  /** Returns 32-bit key based on FED id and channel (and APV). */
  static uint32_t key( const uint16_t& fed_id,
		       const uint16_t& fed_ch,
		       const uint16_t& fed_apv = uint16_t(sistrip::invalid_) );

  /** Returns 32-bit key based on FED id, FE unit and chan and APV. */
  static uint32_t key( const uint16_t& fed_id,
		       const uint16_t& fe_unit,
		       const uint16_t& fe_chan,
		       const uint16_t& fed_apv );
  
  // ---------- Returns paths based on 32-bit keys ----------

  /** Extracts FED id, unit, channel and APV from 32-bit key. */
  static Path path( uint32_t fed_key );

  // ---------- Consistency checks between 32-bit keys ----------

  static bool isEqual( const uint32_t& first_key, 
		       const uint32_t& second_key );
  
  static bool isConsistent( const uint32_t& first_key, 
			    const uint32_t& second_key );

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


