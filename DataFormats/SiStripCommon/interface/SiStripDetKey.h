// Last commit: $Id: SiStripDetKey.h,v 1.8 2009/07/31 09:53:47 lowette Exp $

#ifndef DataFormats_SiStripCommon_SiStripDetKey_h
#define DataFormats_SiStripCommon_SiStripDetKey_h

#include "DataFormats/SiStripCommon/interface/ConstantsForGranularity.h"
#include "DataFormats/SiStripCommon/interface/SiStripKey.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"

/**
   @class SiStripDetKey
   @author R.Bainbridge, S.Lowette

   @brief Utility class that identifies a position within the strip
   tracker geometrical structure, down to the level of an APV25 chip.

   NOTA BENE: *** NOT FINISHED ***
   
   can generate another key that is NOT DetId and packs
   sistrip-specific data in a more condensed way, so that all levels
   can be encoded with "all" and "invalid" values, down to level of
   apv. also, need "conversion tool" that re-generates DetId key from
   this new key. this is only way...!!!  maybe can "safeguard" use of
   this key as a DetId by reserving bits 22-24 as a flag (eg, set all
   high), so that if an attempt to build DetId using SiStripDetId
   class, we can understand if key is real DetId or not... what about
   going to level of apv?... what about levels about module?...
*/
class SiStripDetKey : public SiStripKey {
  
 public:

  // ---------- Constructors ----------

  /** Constructor using partition. */
  SiStripDetKey( const uint16_t& partition );

  /** Constructor using DetId, APV pair and APV pos within pair. */
  SiStripDetKey( const DetId& det_id,
                 const uint16_t& apv_pair_number = 0,
                 const uint16_t& apv_within_pair = 0 );
  
  /** Constructor using SiStripDetId. */
  SiStripDetKey( const SiStripDetId& det_id );
  
  /** Constructor using 32-bit "DET key". */
  SiStripDetKey( const uint32_t& det_key );
  
  /** Constructor using directory path. */
  SiStripDetKey( const std::string& directory_path );

  /** Copy constructor. */
  SiStripDetKey( const SiStripDetKey& );

  /** Copy constructor using base class. */
  SiStripDetKey( const SiStripKey& );
  
  /** Copy to level specified by granularity. */
  SiStripDetKey( const SiStripKey&,
                 const sistrip::Granularity& );

  /** Default constructor */
  SiStripDetKey();
  
  // ---------- Public interface to member data ----------
  
  /** Returns partition. */
  inline const uint16_t& partition() const;
  
  /** Returns APV pair number. */
  inline const uint16_t& apvPairNumber() const;
  
  /** Returns APV position within pair. */
  inline const uint16_t& apvWithinPair() const;
  
  // ---------- Numbering schemes ---------- 
  
  //@@ nothing yet
  //@@ switch b/w det_id and det_key
  //@@ switch b/w strip, pair, apv, etc...

  // ---------- Utility methods ---------- 

  /** Identifies key objects with identical member data. */
  bool isEqual( const SiStripKey& ) const;
  
  /** "Consistent" means identical and/or null (ie, "all") data. */
  bool isConsistent( const SiStripKey& ) const;
  
  /** Identifies all member data as being "valid" or null ("all"). */
  bool isValid() const;
  
  /** All member data to level of "Granularity" are valid. If
      sistrip::Granularity is "undefined", returns false. */
  bool isValid( const sistrip::Granularity& ) const;
  
  /** Identifies all member data as being invalid. */
  bool isInvalid() const;
  
  /** All member data to level of "Granularity" are invalid. If
      sistrip::Granularity is "undefined", returns true.  */
  bool isInvalid( const sistrip::Granularity& ) const;

  // ---------- Print methods ----------
  
  /** Print member data of the key  */
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

  /** partition [0,1-4,invalid]. */
  uint16_t partition_;

  /** APV pair number [0,1-3,invalid]. */
  uint16_t apvPairNumber_;

  /** APV position within pair [0,1-2,invalid]. */
  uint16_t apvWithinPair_; 
  
  // Definition of bit field positions for 32-bit key 
  static const uint16_t partitionOffset_ = 29;

  // Definition of bit field masks for 32-bit key 
  static const uint16_t partitionMask_ = 0x07; // (3 bits)
  
};

// ---------- inline methods ----------

const uint16_t& SiStripDetKey::partition() const { return partition_; }
const uint16_t& SiStripDetKey::apvPairNumber() const { return apvPairNumber_; }
const uint16_t& SiStripDetKey::apvWithinPair() const { return apvWithinPair_; }

/** Debug info for SiStripDetKey class. */
std::ostream& operator<< ( std::ostream&, const SiStripDetKey& );

inline bool operator< ( const SiStripDetKey& a, const SiStripDetKey& b ) { return ( a.key() < b.key() ); }


#endif // DataFormats_SiStripCommon_SiStripDetKey_h
