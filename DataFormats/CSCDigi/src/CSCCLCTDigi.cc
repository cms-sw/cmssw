/**\class CSCCLCTDigi
 *
 * Digi for CLCT trigger primitives.
 *
 * $Date:$
 * $Revision:$
 *
 * \author N. Terentiev, CMU
 */

#include <DataFormats/CSCDigi/interface/CSCCLCTDigi.h>

#include <iostream>
#include <bitset>

using namespace std;

  /// Constructors

CSCCLCTDigi::CSCCLCTDigi (int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx, int trknmb) {
  set(valid, quality, patshape, striptype, bend, strip, cfeb, bx, trknmb);
}

CSCCLCTDigi::CSCCLCTDigi (ChannelType channel){
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->valid,
      ch->quality,
      ch->patshape,
      ch->striptype,
      ch->bend,
      ch->strip,
      ch->cfeb,
      ch->bx,
      ch->trknmb);
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
  set(0,0,0,0,0,0,0,0,0);
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
  result.valid    = d->valid;
  result.quality  = d->quality;
  result.patshape = d->patshape;
  result.striptype= d->striptype;
  result.bend     = d->bend;
  result.strip    = d->strip;
  result.cfeb     = d->cfeb;
  result.bx       = d->bx;
  result.trknmb   = d->trknmb;
  return *(reinterpret_cast<CSCCLCTDigi::ChannelType*>(&result));
}

int CSCCLCTDigi::getValid()     const { return data()->valid;    }
int CSCCLCTDigi::getQuality()   const { return data()->quality;  }
int CSCCLCTDigi::getPattern()   const { return data()->patshape; }
int CSCCLCTDigi::getStriptype() const { return data()->striptype;}
int CSCCLCTDigi::getBend()      const { return data()->bend;     }
int CSCCLCTDigi::getStrip()     const { return data()->strip;    }
int CSCCLCTDigi::getCfeb()      const { return data()->cfeb;     }
int CSCCLCTDigi::getBx()        const { return data()->bx;       }
int CSCCLCTDigi::getTrknmb()    const { return data()->trknmb;   }

  /// Debug

void CSCCLCTDigi::print() const {
  cout << "Valid "          << getValid()
       << "Quality "        << getQuality()
       << "Pattern shape "  << getPattern()
       << "Strip type "     << getStriptype()
       << "Bend "           << getBend()
       << "Strip "          << getStrip()
       << "Key CFEB ID "    << getCfeb()
       << "Bx "             << getBx()
       << "Track number "   << getTrknmb()<<endl;
}

void CSCCLCTDigi::dump() const {
  typedef bitset<8*sizeof(PackedDigiType)> bits;
  cout << *reinterpret_cast<const bits*>(data());  
}

  /// Private members

void CSCCLCTDigi::set(int valid, int quality, int patshape,int striptype, int bend,  int strip, int cfeb, int bx, int trknmb) {
  PackedDigiType* d = data();
  d->valid    = valid;
  d->quality  = quality;
  d->patshape = patshape;
  d->striptype= striptype;
  d->bend     = bend;
  d->strip    = strip;
  d->cfeb     = cfeb;
  d->bx       = bx;
  d->trknmb   = trknmb;
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
