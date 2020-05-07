#ifndef CSCDigi_CSCALCTDigi_h
#define CSCDigi_CSCALCTDigi_h

/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives.
 *
 *
 * \author N. Terentiev, CMU
 */

#include <cstdint>
#include <iosfwd>
#include <limits>

class CSCALCTDigi {
public:
  enum class Version { Legacy = 0, Run3 };

  /// Constructors
  CSCALCTDigi(const uint16_t valid,
              const uint16_t quality,
              const uint16_t accel,
              const uint16_t patternb,
              const uint16_t keywire,
              const uint16_t bx,
              const uint16_t trknmb = 0,
              const uint16_t hmt = 0,
              const Version version = Version::Legacy);
  /// default
  CSCALCTDigi();

  /// clear this ALCT
  void clear();

  /// check ALCT validity (1 - valid ALCT)
  bool isValid() const { return valid_; }

  /// set valid
  void setValid(const uint16_t valid) { valid_ = valid; }

  /// return quality of a pattern
  uint16_t getQuality() const { return quality_; }

  /// set quality
  void setQuality(const uint16_t quality) { quality_ = quality; }

  /// return Accelerator bit
  /// 1-Accelerator pattern, 0-CollisionA or CollisionB pattern
  uint16_t getAccelerator() const { return accel_; }

  /// set accelerator bit
  void setAccelerator(const uint16_t accelerator) { accel_ = accelerator; }

  /// return Collision Pattern B bit
  /// 1-CollisionB pattern (accel_ = 0),
  /// 0-CollisionA pattern (accel_ = 0)
  uint16_t getCollisionB() const { return patternb_; }

  /// set Collision Pattern B bit
  void setCollisionB(const uint16_t collision) { patternb_ = collision; }

  /// return key wire group
  uint16_t getKeyWG() const { return keywire_; }

  /// set key wire group
  void setKeyWG(const uint16_t keyWG) { keywire_ = keyWG; }

  /// return BX - five low bits of BXN counter tagged by the ALCT
  uint16_t getBX() const { return bx_; }

  /// set BX
  void setBX(const uint16_t BX) { bx_ = BX; }

  /// return track number (1,2)
  uint16_t getTrknmb() const { return trknmb_; }

  /// Set track number (1,2) after sorting ALCTs.
  void setTrknmb(const uint16_t number) { trknmb_ = number; }

  /// return 12-bit full BX.
  uint16_t getFullBX() const { return fullbx_; }

  /// Set 12-bit full BX.
  void setFullBX(const uint16_t fullbx) { fullbx_ = fullbx; }

  /// return the high multiplicity bits
  uint16_t getHMT() const;

  /// set the high multiplicity bits
  void setHMT(const uint16_t hmt);

  /// True if the first ALCT has a larger quality, or if it has the same
  /// quality but a larger wire group.
  bool operator>(const CSCALCTDigi&) const;

  /// True if all members (except the number) of both ALCTs are equal.
  bool operator==(const CSCALCTDigi&) const;

  /// True if the preceding one is false.
  bool operator!=(const CSCALCTDigi&) const;

  /// Print content of digi.
  void print() const;

  /// set wiregroup number
  void setWireGroup(uint16_t wiregroup) { keywire_ = wiregroup; }

  /// Distinguish Run-1/2 from Run-3
  bool isRun3() const { return version_ == Version::Run3; }

  void setRun3(const bool isRun3);

private:
  uint16_t valid_;
  uint16_t quality_;
  uint16_t accel_;
  uint16_t patternb_;  // not used since 2007
  uint16_t keywire_;
  uint16_t bx_;
  uint16_t trknmb_;
  uint16_t fullbx_;
  /// Run-3 introduces high-multiplicity bits for CSCs.
  /// Note: In DN-20-016, 3 bits are allocated for HMT in the
  /// ALCT board. These bits are copied into the ALCT digi in CMSSW
  uint16_t hmt_;

  Version version_;
};

std::ostream& operator<<(std::ostream& o, const CSCALCTDigi& digi);
#endif
