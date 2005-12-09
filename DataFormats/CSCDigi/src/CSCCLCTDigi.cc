/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 * $Date$
 * $Revision$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

  /// Constructors

CSCCLCTDigi::CSCCLCTDigi (int trknmb, int pattern, int quality, int bend, int striptype, int strip, int bx){
  set(trknmb, pattern, quality, bend, striptype, strip, bx);
}

CSCCLCTDigi::CSCCLCTDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->trknmb,
      ch->pattern,
      ch->quality,
      ch->bend,
      ch->striptype,
      ch->strip,
      ch->bx);
}

CSCCLCTDigi::CSCCLCTDigi (PackedDigiType packed_value){
  setData(packed_value);
}
      /// Copy
CSCCLCTDigi::CSCCLCTDigi(const CSCCLCTDigi& digi) {
  persistentData = digi.persistentData;
}
      /// Default
CSCCLCTDigi::CSCCLCTDigi (){
  set(0,0,0,0,0,0,0);
}


  /// Assignment
CSCCLCTDigi& 
CSCCLCTDigi::operator=(const CSCCLCTDigi& digi){
  persistentData = digi.persistentData;
  return *this;
}

  /// Getters

CSCCLCTDigi::ChannelType
CSCCLCTDigi::channel() const {
  const PackedDigiType* d = data();
  ChannelPacking result;
  result.trknmb   = d->trknmb;
  result.pattern  = d->pattern;
  result.quality  = d->quality;
  result.bend     = d->bend;
  result.striptype= d->striptype;
  result.strip    = d->strip;
  result.bx       = d->bx;
  return *(reinterpret_cast<CSCCLCTDigi::ChannelType*>(&result));
}

int CSCCLCTDigi::getTrknmb()    const { return data()->trknmb;   }
int CSCCLCTDigi::getPattern()   const { return data()->pattern;  }
int CSCCLCTDigi::getQuality()   const { return data()->quality;  }
int CSCCLCTDigi::getBend()      const { return data()->bend;     }
int CSCCLCTDigi::getStriptype() const { return data()->striptype;}
int CSCCLCTDigi::getStrip()     const { return data()->strip;    }
int CSCCLCTDigi::getBx()        const { return data()->bx;       }

  /// Debug

void CSCCLCTDigi::print() const {
  cout << "Track number "   << getTrknmb()
       << "Pattern number " << getPattern()
       << "Quality "        << getQuality()
       << "Bend "           << getBend()
       << "Strip type "     << getStriptype()
       << "Strip "          << getStrip()
       << "Bx "             << getBx()<<endl;
}

void CSCCLCTDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCCLCTDigi::set(int trknmb, int pattern, int quality, int bend, int striptype, int strip, int bx) {
  PackedDigiType* d = data();
  d->trknmb    = trknmb;
  d->pattern   = pattern;
  d->quality   = quality;
  d->bend      = bend;
  d->striptype = striptype;
  d->strip     = strip;
  d->bx        = bx;
}

CSCCLCTDigi::PackedDigiType* 
CSCCLCTDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCCLCTDigi::PackedDigiType* 
CSCCLCTDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCCLCTDigi::setData(PackedDigiType p){
  *(data()) = p;
}
