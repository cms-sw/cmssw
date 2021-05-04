#ifndef CSCDigi_CSCALCTPreTriggerDigi_h
#define CSCDigi_CSCALCTPreTriggerDigi_h

/**\class CSCALCTPreTriggerDigi
 *
 * Pre-trigger Digi for ALCT trigger primitives.
 *
 */

#include <cstdint>
#include <iosfwd>

class CSCALCTPreTriggerDigi {
public:
  /// Constructors
  CSCALCTPreTriggerDigi(const int valid,
                        const int quality,
                        const int accel,
                        const int patternb,
                        const int keywire,
                        const int bx,
                        const int trknmb = 0);
  /// default
  CSCALCTPreTriggerDigi();

  /// clear this ALCT
  void clear();

  /// check ALCT validity (1 - valid ALCT)
  bool isValid() const { return valid_; }

  /// set valid
  void setValid(const int valid) { valid_ = valid; }

  /// return quality of a pattern
  int getQuality() const { return quality_; }

  /// set quality
  void setQuality(const int quality) { quality_ = quality; }

  /// return Accelerator bit
  /// 1-Accelerator pattern, 0-CollisionA or CollisionB pattern
  int getAccelerator() const { return accel_; }

  /// set accelerator bit
  void setAccelerator(const int accelerator) { accel_ = accelerator; }

  /// return Collision Pattern B bit
  /// 1-CollisionB pattern (accel_ = 0),
  /// 0-CollisionA pattern (accel_ = 0)
  int getCollisionB() const { return patternb_; }

  /// set Collision Pattern B bit
  void setCollisionB(const int collision) { patternb_ = collision; }

  /// return key wire group
  int getKeyWG() const { return keywire_; }

  /// set key wire group
  void setKeyWG(const int keyWG) { keywire_ = keyWG; }

  /// return BX - five low bits of BXN counter tagged by the ALCT
  int getBX() const { return bx_; }

  /// set BX
  void setBX(const int BX) { bx_ = BX; }

  /// return track number (1,2)
  int getTrknmb() const { return trknmb_; }

  /// Set track number (1,2) after sorting ALCTs.
  void setTrknmb(const uint16_t number) { trknmb_ = number; }

  /// return 12-bit full BX.
  int getFullBX() const { return fullbx_; }

  /// Set 12-bit full BX.
  void setFullBX(const uint16_t fullbx) { fullbx_ = fullbx; }

  /// True if the first ALCT has a larger quality, or if it has the same
  /// quality but a larger wire group.
  bool operator>(const CSCALCTPreTriggerDigi&) const;

  /// True if all members (except the number) of both ALCTs are equal.
  bool operator==(const CSCALCTPreTriggerDigi&) const;

  /// True if the preceding one is false.
  bool operator!=(const CSCALCTPreTriggerDigi&) const;

  /// Print content of digi.
  void print() const;

  /// set wiregroup number
  void setWireGroup(unsigned int wiregroup) { keywire_ = wiregroup; }

private:
  uint16_t valid_;
  uint16_t quality_;
  uint16_t accel_;
  uint16_t patternb_;  // not used since 2007
  uint16_t keywire_;
  uint16_t bx_;
  uint16_t trknmb_;
  uint16_t fullbx_;
};

std::ostream& operator<<(std::ostream& o, const CSCALCTPreTriggerDigi& digi);
#endif
