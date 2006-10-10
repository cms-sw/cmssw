#ifndef DataFormats_SiStripDetId_SiStripFedKey_H
#define DataFormats_SiStripDetId_SiStripFedKey_H

#include "DataFormats/SiStripCommon/interface/SiStripConstants.h"
#include <boost/cstdint.hpp>
#include <ostream>

class SiStripFedKey {
  
 public:
  
  /** Simple container class that holds parameters that uniquely
      identify a FED channel within the readout system. */
  class Path { 
  public:
    uint16_t fedId_;
    uint16_t fedFe_; // FE unit [0-7]
    uint16_t fedCh_; // FE channel [0-11]
    Path() : 
      fedId_(sistrip::invalid_),
      fedFe_(sistrip::invalid_),
      fedCh_(sistrip::invalid_) {;}
    Path( uint16_t fed_id,
	  uint16_t fed_fe,
	  uint16_t fed_ch ) :
      fedId_(fed_id),
      fedFe_(fed_fe),
      fedCh_(fed_ch) {;}
  };
  
  /** Creates a 32-bit key that uniquely identifies a FED channel
      within the strip tracker readout system. */
  static uint32_t key( uint16_t fed_id = sistrip::invalid_, 
		       uint16_t fed_fe = sistrip::invalid_, 
		       uint16_t fed_ch = sistrip::invalid_ );
  
  /** Creates a 32-bit key that uniquely identifies a FED channel
      within the strip tracker readout system. */
  static uint32_t key( const Path& );
  
  /** Returns the parameters that uniquely identify a FED channel
      within the strip tracker readout system. */
  static Path path( uint32_t key );
  
 private:

  static const uint16_t fedIdOffset_  = 20;
  static const uint16_t fedFeOffset_  = 16;
  static const uint16_t fedChOffset_  = 8;
  static const uint16_t unusedOffset_ = 0;
  
  static const uint16_t fedIdMask_  = 0xFFF;
  static const uint16_t fedFeMask_  = 0xF;
  static const uint16_t fedChMask_  = 0xFF;
  static const uint16_t unusedMask_ = 0xFF;
  
};

/** Debug info for Path container class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedKey::Path& );

#endif // DataFormats_SiStripDetId_SiStripFedKey_H


