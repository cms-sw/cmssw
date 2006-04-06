#ifndef CSCALCTDigi_CSCALCTDigi_h
#define CSCALCTDigi_CSCALCTDigi_h

/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives. 
 *
 * $Date: 2006/04/06 04:10:46 $
 * $Revision: 1.5 $
 *
 * \author N. Terentiev, CMU
 */

#include <boost/cstdint.hpp>

class CSCALCTDigi{

public:

  /// Constructors
  /// for consistency with DQM
  explicit CSCALCTDigi (int valid, int quality, int accel, int patternb, int keywire, int bx ); 
  explicit CSCALCTDigi (int valid, int quality, int accel, int patternb, int keywire, int bx, int trknmb); 
  /// default
  CSCALCTDigi ();

  /// return ALCT validity, 1 - valid ALCT
  int getValid() const {return valid_ ;}  
  /// check ALCT validity (for consistency with ORCA)    
  bool isValid() const {return valid_ ;}     
  /// return quality of a pattern
  int getQuality() const {return quality_ ;}
  /// return accel (obsolete, use getAccelerator() instead)
  int getAccel() const {return accel_ ;}
  /// return Accelerator bit (for consistency with ORCA)
  /// 1-Accelerator pattern, 0-CollisionA or CollisionB pattern
  int getAccelerator() const {return accel_ ;} 
  /// return pattern (obsolete, use getCollisionB() instead)
  int getPattern() const {return patternb_ ;}
  /// return Collision Pattern B bit (for consistency with ORCA),
  /// 1-CollisionB pattern if accel_ = 0,
  /// 0-CollisionA pattern if accel_ = 0 
  int getCollisionB() const {return patternb_ ;}
  /// return key wire group (obsolete, use getKeyWG() instead)
  int getKwire() const {return keywire_ ;}
  /// return key wire group (for consistency with ORCA)     
  int getKeyWG() const {return  keywire_ ;}
  /// return BX (obsolete, use getBX() instead)
  int getBx() const {return bx_ ;}
  /// return BX (for consistency with ORCA),
  /// five low bits of BXN counter tagged by the ALCT
  int getBX() const {return bx_ ;}
  /// return track number (1,2)
  int getTrknmb() const {return trknmb_ ;}

  /// Print content of digi
  void print() const;

private:
  friend class testCSCDigis;
  uint16_t valid_      ;
  uint16_t quality_    ;
  uint16_t accel_      ;
  uint16_t patternb_   ;
  uint16_t keywire_    ;
  uint16_t bx_         ;
  uint16_t trknmb_     ;
}; 


#include<iostream>
  /// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const CSCALCTDigi& digi) {
  return o << " CSC ALCT isValid "     << digi.isValid()
           << " CSC ALCT Quality "     << digi.getQuality()
           << " CSC ALCT Accelerator " << digi.getAccelerator()
           << " CSC ALCT PatternB "    << digi.getCollisionB()
           << " CSC ALCT key wire "    << digi.getKeyWG()
           << " CSC ALCT BX "          << digi.getBx()
           << " CSC ALCT track number" << digi.getTrknmb();
}
#endif
