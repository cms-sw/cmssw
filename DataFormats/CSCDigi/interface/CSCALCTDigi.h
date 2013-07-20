#ifndef CSCDigi_CSCALCTDigi_h
#define CSCDigi_CSCALCTDigi_h

/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives. 
 *
 * $Date: 2009/03/25 13:13:11 $
 * $Revision: 1.13 $
 *
 * \author N. Terentiev, CMU
 */

#include <boost/cstdint.hpp>
#include <iosfwd>

class CSCALCTDigi {

 public:

  /// Constructors
  CSCALCTDigi(const int valid, const int quality, const int accel,
	      const int patternb, const int keywire, const int bx,
	      const int trknmb = 0);
  /// default
  CSCALCTDigi();

  /// clear this ALCT
  void clear();

  /// check ALCT validity (1 - valid ALCT)
  bool isValid() const {return valid_ ;}

  /// return quality of a pattern
  int getQuality() const {return quality_ ;}

  /// return Accelerator bit
  /// 1-Accelerator pattern, 0-CollisionA or CollisionB pattern
  int getAccelerator() const {return accel_ ;}

  /// return Collision Pattern B bit
  /// 1-CollisionB pattern (accel_ = 0),
  /// 0-CollisionA pattern (accel_ = 0)
  int getCollisionB() const {return patternb_ ;}

  /// return key wire group
  int getKeyWG() const {return keywire_ ;}

  /// return BX - five low bits of BXN counter tagged by the ALCT
  int getBX() const {return bx_ ;}

  /// return track number (1,2)
  int getTrknmb() const {return trknmb_ ;}

  /// Set track number (1,2) after sorting ALCTs.
  void setTrknmb(const uint16_t number) {trknmb_ = number;}

  /// return 12-bit full BX.
  int getFullBX() const {return fullbx_ ;}

  /// Set 12-bit full BX.
  void setFullBX(const uint16_t fullbx) {fullbx_ = fullbx;}

  /// True if the first ALCT has a larger quality, or if it has the same
  /// quality but a larger wire group.
  bool operator >  (const CSCALCTDigi&) const;

  /// True if all members (except the number) of both ALCTs are equal.
  bool operator == (const CSCALCTDigi&) const;

  /// True if the preceding one is false.
  bool operator != (const CSCALCTDigi&) const;

  /// Print content of digi.
  void print() const;

  /// set wiregroup number
  void setWireGroup(unsigned int wiregroup) {keywire_= wiregroup;}

 private:

  uint16_t valid_      ;
  uint16_t quality_    ;
  uint16_t accel_      ;
  uint16_t patternb_   ; // not used since 2007
  uint16_t keywire_    ;
  uint16_t bx_         ;
  uint16_t trknmb_     ;
  uint16_t fullbx_     ;
};

std::ostream & operator<<(std::ostream & o, const CSCALCTDigi& digi);
#endif
