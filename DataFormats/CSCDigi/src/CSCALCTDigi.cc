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
#include <bitset>

using namespace std;

  /// Constructors

/*CSCALCTDigi::CSCALCTDigi (int trknmb, int keywire,int bx, int quality, 
int pattern, int valid)
{
  set(trknmb, keywire, bx, quality, pattern, valid);
} */

CSCALCTDigi::CSCALCTDigi (int valid, int quality, int accel, int pattern,
int keywire, int bx) {
  set(valid, quality, accel, pattern, keywire, bx, 0);
}      // for DQM

CSCALCTDigi::CSCALCTDigi (int valid, int quality, int accel, int pattern, 
int keywire, int bx, int trknmb) {
  set(valid, quality, accel, pattern, keywire, bx, trknmb);
}
CSCALCTDigi::CSCALCTDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(
      ch->valid,
      ch->quality,
      ch->accel,
      ch->pattern,
      ch->keywire,
      ch->bx,
      ch->trknmb);
}

CSCALCTDigi::CSCALCTDigi (PackedDigiType packed_value){
  setData(packed_value);
}
      /// Copy
CSCALCTDigi::CSCALCTDigi(const CSCALCTDigi& digi) {
  persistentData = digi.persistentData;
}
      /// Default
CSCALCTDigi::CSCALCTDigi (){
  set(0,0,0,0,0,0,0);
}


  /// Assignment
CSCALCTDigi& 
CSCALCTDigi::operator=(const CSCALCTDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

  /// Getters

CSCALCTDigi::ChannelType
CSCALCTDigi::channel() const {
  const PackedDigiType* d = data();
  ChannelPacking result;
  result.valid    = d->valid;
  result.quality  = d->quality;
  result.accel    = d->accel;
  result.pattern  = d->pattern;
  result.keywire  = d->keywire;
  result.bx       = d->bx;
  result.trknmb   = d->trknmb;
  return *(reinterpret_cast<CSCALCTDigi::ChannelType*>(&result));
}

int CSCALCTDigi::getValid()       const { return data()->valid;  }
bool CSCALCTDigi::isValid()       const { return data()->valid; }

int CSCALCTDigi::getQuality()     const { return data()->quality;}

int CSCALCTDigi::getAccel()       const { return data()->accel;}
int CSCALCTDigi::getAccelerator() const { return data()->accel;}

int CSCALCTDigi::getPattern()     const { return data()->pattern;}
int CSCALCTDigi::getCollisionB()  const { return data()->pattern;}

int CSCALCTDigi::getKwire()       const { return data()->keywire;}
int CSCALCTDigi::getKeyWG()       const { return data()->keywire;}

int CSCALCTDigi::getBx()          const { return data()->bx;     }
int CSCALCTDigi::getBX()          const { return data()->bx;     }

int CSCALCTDigi::getTrknmb()      const { return data()->trknmb; }

  /// Debug

void CSCALCTDigi::print() const { 
  cout << " Valid: "          << setw(1)<<isValid() 
       << " Quality: "        << setw(1)<<getQuality()
       << " Accel.:  "        << setw(1)<<getAccelerator()
       << " Collision B: "    << setw(1)<<getCollisionB()
       << " Key Wire group: " << setw(3)<<getKeyWG()
       << " BX: "             << setw(2)<<getBX()
       << " Track number: "   << setw(2)<<getTrknmb() << endl;
}

void CSCALCTDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCALCTDigi::set(int valid, int quality, int accel, int pattern, int keywire, int bx, int trknmb) 
{
  PackedDigiType* d = data();
  d->valid   = valid;
  d->quality = quality;
  d->accel   = accel;
  d->pattern = pattern;
  d->keywire = keywire;
  d->bx      = bx;
  d->trknmb  = trknmb;
}

CSCALCTDigi::PackedDigiType* 
CSCALCTDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCALCTDigi::PackedDigiType* 
CSCALCTDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCALCTDigi::setData(PackedDigiType p){
  *(data()) = p;
}
