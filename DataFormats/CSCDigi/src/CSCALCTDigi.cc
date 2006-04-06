/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives.
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>

#include <iostream>
#include <iomanip>

using namespace std;

  /// Constructors

CSCALCTDigi::CSCALCTDigi (int valid, int quality, int accel, int patternb,int keywire, int bx) {
  valid_    = valid;
  quality_  = quality;
  accel_    = accel;
  patternb_ = patternb;
  keywire_  = keywire;
  bx_       = bx;
  trknmb_   = 0;
}    

CSCALCTDigi::CSCALCTDigi (int valid, int quality, int accel, int patternb, int keywire, int bx, int trknmb) {
  valid_    = valid;
  quality_  = quality;
  accel_    = accel;
  patternb_ = patternb;
  keywire_  = keywire;
  bx_       = bx;
  trknmb_   = trknmb;
}
   /// Copy
CSCALCTDigi::CSCALCTDigi(const CSCALCTDigi& digi) {
  valid_    = digi.getValid(); 
  quality_  = digi.getQuality();
  accel_    = digi.getAccelerator();
  patternb_ = digi.getCollisionB();
  keywire_  = digi.getKeyWG();
  bx_       = digi.getBX();
  trknmb_   = digi.getTrknmb();
}
      /// Default
CSCALCTDigi::CSCALCTDigi (){
  valid_    = 0;
  quality_  = 0;
  accel_    = 0;
  patternb_ = 0;
  keywire_  = 0;
  bx_       = 0;
  trknmb_   = 0;
}


  /// Assignment
CSCALCTDigi& 
CSCALCTDigi::operator=(const CSCALCTDigi& digi){
  valid_    = digi.getValid();
  quality_  = digi.getQuality();
  accel_    = digi.getAccelerator();
  patternb_ = digi.getCollisionB();
  keywire_  = digi.getKeyWG();
  bx_       = digi.getBX();
  trknmb_   = digi.getTrknmb();

  return *this;
}

  /// Debug

void CSCALCTDigi::print() const { 
  std::cout << " CSC ALCT Valid: "     << setw(1)<<isValid() 
       << " CSC ALCT Quality: "        << setw(1)<<getQuality()
       << " CSC ALCT Accel.:  "        << setw(1)<<getAccelerator()
       << " CSC ALCT Collision B: "    << setw(1)<<getCollisionB()
       << " CSC ALCT key wire group: " << setw(3)<<getKeyWG()
       << " CSC ALCT BX: "             << setw(2)<<getBX()
       << " CSC ALCT track number: "   << setw(2)<<getTrknmb() << std::endl;
}
