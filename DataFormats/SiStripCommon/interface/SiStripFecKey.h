// Last commit: $Id: SiStripFecKey.h,v 1.16 2008/02/21 16:51:55 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_SiStripFecKey_H
#define DataFormats_SiStripCommon_SiStripFecKey_H

#include "DataFormats/SiStripCommon/interface/ConstantsForGranularity.h"
#include "DataFormats/SiStripCommon/interface/SiStripKey.h"

/**
   @class SiStripFecKey
   @author R.Bainbridge
   
   @brief Utility class that identifies a position within the strip
   tracker control structure, down to the level of an APV25.
   
   The class allows to encode the position within a 32-bit "key" and,
   conversely, unpack a 32-bit key to provide the position.

   The class provides the following member data: 
   - FEC key (32 bits),
   - VME crate,
   - FEC (using VME slot), 
   - FEC control ring,
   - CCU module,
   - Front-End module,
   - LLD channel,
   - APV25 chip,
   - directory path,
   - "granularity".
   
   Member data (integer in type only) with values of 0xFFFF signifies
   "invalid" (ie, FecSlot = 0xFFFF means "invalid FEC slot"). Data
   with null values signifies "all" (ie, FecSlot = 0 means "all FEC
   slots").

   The class generates a "directory path" string according to the
   member data. This can be used to organise histograms / other data
   types when using DQM / root. Conversely, the member data can also
   be built using the directory path when provided as a constructor
   argument.

   The class also provides the "granularity" to which the FEC key is
   unambiguous (ie, not "invalid" or "null") in defining a position
   within the control system.
*/
class SiStripFecKey : public SiStripKey {
  
 public:
  
  // ---------- Constructors ----------

  /** Constructor using crate, FEC, ring, CCU, module and channel. */
  SiStripFecKey( const uint16_t& fec_crate, 
		 const uint16_t& fec_slot = 0, 
		 const uint16_t& fec_ring = 0, 
		 const uint16_t& ccu_addr = 0, 
		 const uint16_t& ccu_chan = 0,
		 const uint16_t& lld_chan = 0,
		 const uint16_t& i2c_addr = 0 );
  
  /** Constructor using 32-bit "FEC key". */
  SiStripFecKey( const uint32_t& fec_key );
  
  /** Constructor using directory path. */
  SiStripFecKey( const std::string& directory_path );

  /** Copy constructor. */
  SiStripFecKey( const SiStripFecKey& );

  /** Copy constructor using base class. */
  SiStripFecKey( const SiStripKey& );

  /** Copy to level specified by granularity. */
  SiStripFecKey( const SiStripKey&,
		 const sistrip::Granularity& );

  /** Default constructor */
  SiStripFecKey();
  
  // ---------- Control structure ----------
  
  /** Returns VME crate. */
  inline const uint16_t& fecCrate() const;
  
  /** Returns FEC identifier (VME slot). */
  inline const uint16_t& fecSlot() const;
  
  /** Returns FEC control ring. */
  inline const uint16_t& fecRing() const;
  
  /** Returns CCU module. */
  inline const uint16_t& ccuAddr() const;

  /** Returns Front-End module. */
  inline const uint16_t& ccuChan() const;

  /** Returns LLD channel. */
  inline const uint16_t& lldChan() const;

  /** Returns I2C address ("invalid" if inconsistent with LLD chan. */
  inline const uint16_t& i2cAddr() const;

  // ---------- Hybrid APV/LLD numbering scheme ---------- 
  
  /** Returns hybrid position (1-6) for a given I2C addr (32-37). */
  static uint16_t hybridPos( const uint16_t& i2c_addr );
  
  /** Returns I2C addr (32-37) for a given hybrid position (1-6). */
  static uint16_t i2cAddr( const uint16_t& hybrid_pos );
  
  /** Returns LLD channel (1-3) for a given APV I2C addr (32-37). */
  static uint16_t lldChan( const uint16_t& i2c_addr );
  
  /** Identifies if first APV of pair for given I2C addr (32-37). */
  static bool firstApvOfPair( const uint16_t& i2c_addr );
  
  /** Returns I2C addr (32-37) for LLD chan (1-3) and APV pos. */
  static uint16_t i2cAddr( const uint16_t& lld_chan,
			   const bool& first_apv_of_pair );
  
  // ---------- Utility methods ---------- 
  
  /** Identifies key objects with identical member data. */
  bool isEqual( const SiStripKey& ) const;
  
  /** "Consistent" means identical and/or null (ie, "all") data. */
  bool isConsistent( const SiStripKey& ) const;
  
  /** Identifies all member data as being "valid" or "all" (null). */
  bool isValid() const;
  
  /** All member data to level of "Granularity" are "valid". If
      sistrip::Granularity is "undefined", returns false. */
  bool isValid( const sistrip::Granularity& ) const;
  
  /** Identifies all member data as being "invalid". */
  bool isInvalid() const;

  /** All member data to level of "Granularity" are invalid. If
      sistrip::Granularity is "undefined", returns true.  */
  bool isInvalid( const sistrip::Granularity& ) const;

  // ---------- Print methods ----------

  /** A terse summary of the key  */
  virtual void print( std::stringstream& ss ) const;

  /** A terse summary of the key  */
  virtual void terse( std::stringstream& ss ) const;
  
 private:
  
  // ---------- Private methods ----------
  
  void initFromValue();
  void initFromKey();
  void initFromPath();
  void initGranularity();
  
  // ---------- Private member data ----------

  /** FEC crate [0,1-4,invalid]. */
  uint16_t fecCrate_;

  /** FEC slot [0,2-21,invalid]. */
  uint16_t fecSlot_;

  /** FEC ring [0,1-8,invalid]. */
  uint16_t fecRing_;

  /** CCU module [0,1-126,invalid]. */
  uint16_t ccuAddr_;

  /** FE module [0,16-31,invalid]. */
  uint16_t ccuChan_;

  /** LLD channel [0,1-3,invalid]. */
  uint16_t lldChan_;

  /** APV I2C address [0,32-37,invalid]. */
  uint16_t i2cAddr_;
  
  // Definition of bit field positions for 32-bit key 
  static const uint16_t fecCrateOffset_ = 27;
  static const uint16_t fecSlotOffset_  = 22;
  static const uint16_t fecRingOffset_  = 18;
  static const uint16_t ccuAddrOffset_  = 10;
  static const uint16_t ccuChanOffset_  =  5;
  static const uint16_t lldChanOffset_  =  2;
  static const uint16_t i2cAddrOffset_  =  0;
  
  // Definition of bit field masks for 32-bit key 
  static const uint16_t fecCrateMask_ = 0x07; // (3 bits)
  static const uint16_t fecSlotMask_  = 0x1F; // (5 bits)
  static const uint16_t fecRingMask_  = 0x0F; // (4 bits)
  static const uint16_t ccuAddrMask_  = 0xFF; // (8 bits)
  static const uint16_t ccuChanMask_  = 0x1F; // (5 bits)
  static const uint16_t lldChanMask_  = 0x07; // (3 bits)
  static const uint16_t i2cAddrMask_  = 0x03; // (2 bits)
  
};

// ---------- Inline methods ----------

const uint16_t& SiStripFecKey::fecCrate() const { return fecCrate_; }
const uint16_t& SiStripFecKey::fecSlot() const { return fecSlot_; }
const uint16_t& SiStripFecKey::fecRing() const { return fecRing_; }
const uint16_t& SiStripFecKey::ccuAddr() const { return ccuAddr_; }
const uint16_t& SiStripFecKey::ccuChan() const { return ccuChan_; }
const uint16_t& SiStripFecKey::lldChan() const { return lldChan_; }
const uint16_t& SiStripFecKey::i2cAddr() const { return i2cAddr_; }

/* const uint16_t& SiStripFecKey::fecCrate() const {  */
/*   return ( key()>>fecCrateOffset_ ) & fecCrateMask_ != fecCrateMask_ ? ( key()>>fecCrateOffset_ ) & fecCrateMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::fecSlot() const {  */
/*   return ( key()>>fecSlotOffset_ ) & fecSlotMask_ != fecSlotMask_ ? ( key()>>fecSlotOffset_ ) & fecSlotMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::fecRing() const {  */
/*   return ( key()>>fecRingOffset_ ) & fecRingMask_ != fecRingMask_ ? ( key()>>fecRingOffset_ ) & fecRingMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::ccuAddr() const {  */
/*   return ( key()>>ccuAddrOffset_ ) & ccuAddrMask_ != ccuAddrMask_ ? ( key()>>ccuAddrOffset_ ) & ccuAddrMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::ccuChan() const {  */
/*   return ( key()>>ccuChanOffset_ ) & ccuChanMask_ != ccuChanMask_ ? ( key()>>ccuChanOffset_ ) & ccuChanMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::lldChan() const {  */
/*   return ( key()>>lldChanOffset_ ) & lldChanMask_ != lldChanMask_ ? ( key()>>lldChanOffset_ ) & lldChanMask_ : sistrip::invalid_;  */
/* } */
/* const uint16_t& SiStripFecKey::i2cAddr() const {  */
/*   return ( key()>>i2cAddrOffset_ ) & i2cAddrMask_ != i2cAddrMask_ ? ( key()>>i2cAddrOffset_ ) & i2cAddrMask_ : sistrip::invalid_;  */
/* } */

std::ostream& operator<< ( std::ostream&, const SiStripFecKey& );

inline bool operator< ( const SiStripFecKey& a, const SiStripFecKey& b ) { return ( a.key() < b.key() ); }

class ConsistentWithKey {
 public: 
  explicit ConsistentWithKey( const SiStripFecKey& key );
  bool operator() ( const uint32_t&, const uint32_t& ) const;
 private:
  explicit ConsistentWithKey();
  SiStripFecKey mask_;
};

#endif // DataFormats_SiStripCommon_SiStripFecKey_H
