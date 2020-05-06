#ifndef DataFormats_CSCDigi_CSCCLCTDigi_h
#define DataFormats_CSCDigi_CSCCLCTDigi_h

/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 *
 * \author N. Terentiev, CMU
 */

#include <cstdint>
#include <iosfwd>
#include <vector>
#include <limits>

class CSCCLCTDigi {
public:
  typedef std::vector<std::vector<uint16_t>> ComparatorContainer;

  enum CLCTKeyStripMasks { kEightStripMask = 0x1, kQuartStripMask = 0x1, kHalfStripMask = 0x1f };
  enum CLCTKeyStripShifts { kEightStripShift = 6, kQuartStripShift = 5, kHalfStripShift = 0 };
  enum class Version { Legacy = 0, Run3 };

  /// Constructors
  CSCCLCTDigi(const uint16_t valid,
              const uint16_t quality,
              const uint16_t pattern,
              const uint16_t striptype,
              const uint16_t bend,
              const uint16_t strip,
              const uint16_t cfeb,
              const uint16_t bx,
              const uint16_t trknmb = 0,
              const uint16_t fullbx = 0,
              const int16_t compCode = -1,
              const Version version = Version::Legacy);
  /// default
  CSCCLCTDigi();

  /// clear this CLCT
  void clear();

  /// check CLCT validity (1 - valid CLCT)
  bool isValid() const { return valid_; }

  /// set valid
  void setValid(const uint16_t valid) { valid_ = valid; }

  /// return quality of a pattern (number of layers hit!)
  uint16_t getQuality() const { return quality_; }

  /// set quality
  void setQuality(const uint16_t quality) { quality_ = quality; }

  /// return pattern
  uint16_t getPattern() const { return pattern_; }

  /// set pattern
  void setPattern(const uint16_t pattern) { pattern_ = pattern; }

  /// return striptype
  uint16_t getStripType() const { return striptype_; }

  /// set stripType
  void setStripType(const uint16_t stripType) { striptype_ = stripType; }

  /// return bend
  uint16_t getBend() const { return bend_; }

  /// set bend
  void setBend(const uint16_t bend) { bend_ = bend; }

  /// return halfstrip that goes from 0 to 31 in a (D)CFEB
  uint16_t getStrip() const;

  /// set strip
  void setStrip(const uint16_t strip) { strip_ = strip; }

  /// set single quart strip bit
  void setQuartStrip(const bool quartStrip);

  /// get single quart strip bit
  bool getQuartStrip() const;

  /// set single eight strip bit
  void setEightStrip(const bool eightStrip);

  /// get single eight strip bit
  bool getEightStrip() const;

  /// return Key CFEB ID
  uint16_t getCFEB() const { return cfeb_; }

  /// set Key CFEB ID
  void setCFEB(const uint16_t cfeb) { cfeb_ = cfeb; }

  /// return BX
  uint16_t getBX() const { return bx_; }

  /// set bx
  void setBX(const uint16_t bx) { bx_ = bx; }

  /// return track number (1,2)
  uint16_t getTrknmb() const { return trknmb_; }

  /// Convert strip_ and cfeb_ to keyStrip. Each CFEB has up to 16 strips
  /// (32 halfstrips). There are 5 cfebs.  The "strip_" variable is one
  /// of 32 halfstrips on the keylayer of a single CFEB, so that
  /// Halfstrip = (cfeb*32 + strip).
  /// This function can also return the quartstrip or eightstrip
  /// when the comparator code has been set
  uint16_t getKeyStrip(uint16_t n = 2) const;

  /// Set track number (1,2) after sorting CLCTs.
  void setTrknmb(const uint16_t number) { trknmb_ = number; }

  /// return 12-bit full BX.
  uint16_t getFullBX() const { return fullbx_; }

  /// Set 12-bit full BX.
  void setFullBX(const uint16_t fullbx) { fullbx_ = fullbx; }

  // 12-bit comparator code
  int16_t getCompCode() const { return (isRun3() ? compCode_ : -1); }

  void setCompCode(const int16_t code) { compCode_ = code; }

  // comparator hits in this CLCT
  ComparatorContainer getHits() const { return hits_; }

  void setHits(const ComparatorContainer& hits) { hits_ = hits; }

  /// True if the left-hand side has a larger "quality".  Full definition
  /// of "quality" depends on quality word itself, pattern type, and strip
  /// number.
  bool operator>(const CSCCLCTDigi&) const;

  /// True if the two LCTs have exactly the same members (except the number).
  bool operator==(const CSCCLCTDigi&) const;

  /// True if the preceding one is false.
  bool operator!=(const CSCCLCTDigi&) const;

  /// Print content of digi.
  void print() const;

  /// Distinguish Run-1/2 from Run-3
  bool isRun3() const { return version_ == Version::Run3; }

  void setRun3(bool isRun3);

private:
  uint16_t valid_;
  uint16_t quality_;
  // In Run-3, the 4-bit pattern number is reinterpreted as the
  // 4-bit bending value. There will be 16 bending values * 2 (left/right)
  uint16_t pattern_;
  uint16_t striptype_;  // not used since mid-2008
  // Common definition for left/right bending in Run-1, Run-2 and Run-3.
  // 0: right; 1: left
  uint16_t bend_;
  // In Run-3, the strip number receives two additional bits
  // strip[4:0] -> 1/2 strip value
  // strip[5]   -> 1/4 strip bit
  // strip[6]   -> 1/8 strip bit
  uint16_t strip_;
  // There are up to 7 (D)CFEBs in a chamber
  uint16_t cfeb_;
  uint16_t bx_;
  uint16_t trknmb_;
  uint16_t fullbx_;

  // new in Run-3: 12-bit comparator code
  // set by default to -1 for Run-1 and Run-2 CLCTs
  int16_t compCode_;
  // which hits are in this CLCT?
  ComparatorContainer hits_;

  Version version_;
};

std::ostream& operator<<(std::ostream& o, const CSCCLCTDigi& digi);

#endif
