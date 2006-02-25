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
#include <bitset>

using namespace std;

  /// Constructors

/*CSCALCTDigi::CSCALCTDigi (int trknmb, int keywire,int bx, int quality, 
int pattern, int valid)
{
  set(trknmb, keywire, bx, quality, pattern, valid);
} */

CSCALCTDigi::CSCALCTDigi (int valid, int quality, int accel, int pattern, int keywire, int bx, int trknmb) {
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

int CSCALCTDigi::getValid()   const { return data()->valid;  }
int CSCALCTDigi::getQuality() const { return data()->quality;}
int CSCALCTDigi::getAccel()   const { return data()->accel;}
int CSCALCTDigi::getPattern() const { return data()->pattern;}
int CSCALCTDigi::getKwire()   const { return data()->keywire;}
int CSCALCTDigi::getBx()      const { return data()->bx;     }
int CSCALCTDigi::getTrknmb()  const { return data()->trknmb; }

  /// Debug

void CSCALCTDigi::print() const { 
  cout << "Validity    " << getValid() 
       << "Quality     " << getQuality()
       << "Accel       " << getAccel()
       << "Pattern     " << getPattern()
       << "Key wire    " << getKwire()
       << "Bx          " << getBx()
       << "Track number" << getTrknmb() << endl;
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
  d->trknmb   = trknmb;
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
