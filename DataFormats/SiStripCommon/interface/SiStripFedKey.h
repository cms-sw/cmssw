// Last commit: $Id: SiStripFedKey.h,v 1.14 2008/02/22 09:53:14 bainbrid Exp $

#ifndef DataFormats_SiStripCommon_SiStripFedKey_H
#define DataFormats_SiStripCommon_SiStripFedKey_H

#include "DataFormats/SiStripCommon/interface/ConstantsForGranularity.h"
#include "DataFormats/SiStripCommon/interface/SiStripKey.h"

class SiStripFedKey;

/** Debug info for SiStripFedKey class. */
std::ostream& operator<< ( std::ostream&, const SiStripFedKey& );

/**
   @class SiStripFedKey
   @author R.Bainbridge

   @brief Utility class that identifies a position within the strip
   tracker readout structure, down to the level of an APV25 chip.
   
   The class allows to encode the position within a 32-bit "key" and,
   conversely, unpack a 32-bit key to provide the position.

   The class provides the following member data: 
   - FED key (32 bits),
   - FED id, 
   - Front-End unit ("external" numbering scheme),
   - channel within a Front-End unit ("external" numbering scheme),
   - APV number within a channel (or, equivalently, an APV pair).
   - directory path,
   - "granularity".
   
   Member data (integer in type only) with values of 0xFFFF signifies
   "invalid" (ie, FedId = 0xFFFF means "invalid FED id"). Data with
   null values signifies "all" (ie, FedId = 0 means "all FEDs").

   The class generates a "directory path" string according to the
   member data. This can be used to organise histograms / other data
   types when using DQM / root. Conversely, the member data can also
   be built using the directory path when provided as a constructor
   argument.

   The class also provides the "granularity" to which the FED key is
   unambiguous (ie, not "invalid" or "null") in defining a position
   within the readout system.

   In addition, the class provides static methods that allow to
   convert between the two "FED channel" numbering schema in
   place. The class member data hold values that respect the
   "external" numbering scheme used by the optical links
   group. Front-End units are numbered from 1 to 8, bottom to
   top. Channels with the FE units are numbered 1 to 12, bottom to
   top. The "internal" numbering scheme is used by the DAQ software,
   which numbers FED channels consecutively from 0 to 95, top to
   bottom.
*/
class SiStripFedKey : public SiStripKey {
  
 public:
  
  // ---------- Constructors ----------
  
  /** Constructor using FED id, FE unit, FE channel, and APV. */
  SiStripFedKey( const uint16_t& fed_id,
		 const uint16_t& fe_unit = 0,
		 const uint16_t& fe_chan = 0,
		 const uint16_t& fed_apv = 0 );
  
  /** Constructor using 32-bit "FED key". */
  SiStripFedKey( const uint32_t& fed_key );
  
  /** Constructor using directory path. */
  SiStripFedKey( const std::string& directory_path );

  /** Copy constructor. */
  SiStripFedKey( const SiStripFedKey& );

  /** Copy constructor using base class. */
  SiStripFedKey( const SiStripKey& );
  
  /** Default constructor */
  SiStripFedKey();
  
  // ---------- Public interface to member data ----------

  /** Returns FED id. */
  inline const uint16_t& fedId() const;

  /** Returns Front-End unit (according to "external" numbering). */
  inline const uint16_t& feUnit() const;

  /** Returns chan of FE unit (according to "external" numbering). */
  inline const uint16_t& feChan() const;

  /** Returns APV within FED channel. */
  inline const uint16_t& fedApv() const;

  /** Returns FED channel (according to "internal" numbering). */
  inline uint16_t fedChannel() const;

  // ---------- Numbering schemes ---------- 
  
  /** Returns FED channel ("internal" numbering scheme) for given
      Front-End unit and channel ("external" numbering scheme). */
  static uint16_t fedCh( const uint16_t& fe_unit,
			 const uint16_t& fe_chan );
  
  /** Returns Front-End unit ("external" numbering scheme) for given
      FED channel ("internal" numbering scheme). */
  static uint16_t feUnit( const uint16_t& fed_ch );
  
  /** Returns Front-End channel ("external" numbering scheme) for
      given FED channel ("internal" numbering scheme). */
  static uint16_t feChan( const uint16_t& fed_ch );
  
  /** Returns number that encodes FED id and FED channel, which can be
      used to index vectors containing event and non-event data. Users
      should check if returned value is valid for indexing vector! */
  static uint32_t fedIndex( const uint16_t& fed_id,
			    const uint16_t& fed_ch );
  
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

  /** FED id [0,50-489,invalid]. */
  uint16_t fedId_;  

  /** FE unit [0,1-8,invalid]. */
  uint16_t feUnit_; 

  /** FE channel [0,1-12,invalid]. */
  uint16_t feChan_; 

  /** APV [0,1-2,invalid]. */
  uint16_t fedApv_; 
  
  // Definition of bit field positions for 32-bit key 
  static const uint16_t fedCrateOffset_ = 24;
  static const uint16_t fedSlotOffset_  = 19;
  static const uint16_t fedIdOffset_    = 10;
  static const uint16_t feUnitOffset_   =  6;
  static const uint16_t feChanOffset_   =  2;
  static const uint16_t fedApvOffset_   =  0;

  // Definition of bit field masks for 32-bit key 
  static const uint16_t fedCrateMask_ = 0x03F; // (6 bits)
  static const uint16_t fedSlotMask_  = 0x01F; // (5 bits)
  static const uint16_t fedIdMask_    = 0x1FF; // (9 bits)
  static const uint16_t feUnitMask_   = 0x00F; // (4 bits)
  static const uint16_t feChanMask_   = 0x00F; // (4 bits)
  static const uint16_t fedApvMask_   = 0x003; // (2 bits)
  
};

// ---------- Inline methods ----------

const uint16_t& SiStripFedKey::fedId() const { return fedId_; }
const uint16_t& SiStripFedKey::feUnit() const { return feUnit_; }
const uint16_t& SiStripFedKey::feChan() const { return feChan_; }
const uint16_t& SiStripFedKey::fedApv() const { return fedApv_; }
uint16_t SiStripFedKey::fedChannel() const { return fedCh( feUnit_, feChan_ ); }

#endif // DataFormats_SiStripCommon_SiStripFedKey_H



