/**\class CSCALCTDigi
 *
 * Digi for ALCT trigger primitives.
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

  /// Constructors

CSCALCTDigi::CSCALCTDigi (int trknmb, int keywire,int bx, int quality, int pattern, int valid){
  set(trknmb, keywire, bx, quality, pattern, valid);
}

CSCALCTDigi::CSCALCTDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->trknmb,
      ch->keywire,
      ch->bx,
      ch->quality,
      ch->pattern,
      ch->valid);
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
  set(0,0,0,0,0,0);
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
  result.trknmb   = d->trknmb;
  result.keywire  = d->keywire;
  result.bx       = d->bx;
  result.quality  = d->quality;
  result.pattern  = d->pattern;
  result.valid    = d->valid;
  return *(reinterpret_cast<CSCALCTDigi::ChannelType*>(&result));
}

int CSCALCTDigi::getTrknmb()  const { return data()->trknmb; }
int CSCALCTDigi::getKwire()   const { return data()->keywire;}
int CSCALCTDigi::getBx()      const { return data()->bx;     }
int CSCALCTDigi::getQuality() const { return data()->quality;}
int CSCALCTDigi::getPattern() const { return data()->pattern;}
int CSCALCTDigi::getValid()   const { return data()->valid;  }

  /// Debug

void CSCALCTDigi::print() const {
  cout << "Track number" << getTrknmb()
       << "Key wire    " << getKwire()
       << "Bx          " << getBx()
       << "Quality     " << getQuality()
       << "Pattern     " << getPattern() 
       << "Validity    " << getValid() <<endl;
}

void CSCALCTDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCALCTDigi::set(int trknmb, int keywire,int bx, int quality, int pattern, int valid) {
  PackedDigiType* d = data();
  d->trknmb   = trknmb;
  d->keywire = keywire;
  d->bx      = bx;
  d->quality = quality;
  d->pattern = pattern;
  d->valid   = valid;

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
