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
#include <iomanip>
#include <bitset>

using namespace std;

  /// Constructors

CSCCLCTDigi::CSCCLCTDigi (int valid, int quality, int patshape, int striptype, int bend,  int strip, int cfeb, int bx) {
  set(valid, quality, patshape, striptype, bend, strip, cfeb, bx, 0);
}                 // for DQM

CSCCLCTDigi::CSCCLCTDigi (int valid, int quality, int patshape, int striptype,int bend,  int strip, int cfeb, int bx, int trknmb) {
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
bool CSCCLCTDigi::isValid()     const { return data()->valid;    }

int CSCCLCTDigi::getQuality()   const { return data()->quality;  }
int CSCCLCTDigi::getPattern()   const { return data()->patshape; }

int CSCCLCTDigi::getStriptype() const { return data()->striptype;}
int CSCCLCTDigi::getStripType() const { return data()->striptype;}

int CSCCLCTDigi::getBend()      const { return data()->bend;     }
int CSCCLCTDigi::getStrip()     const { return data()->strip;    }

int CSCCLCTDigi::getKeyStrip()     const {
    // Convert strip and CFEB to keyStrip. Each CFEB has up to 16 strips
    // (32 halfstrips). There are 5 cfebs.  The "strip" variable is one
    // of 32 halfstrips on the keylayer of a single CFEB, so that
    // Distrip   = (cfeb*32 + strip)/4.
    // Halfstrip = (cfeb*32 + strip).
 
                int keyStrip = 0;
                if (data()->striptype == 1)
                        keyStrip = data()->cfeb*32 +  data()->strip;
                else
                        keyStrip = data()->cfeb*8  + (data()->strip/4);
                return keyStrip;
}

int CSCCLCTDigi::getCfeb()      const { return data()->cfeb;     }
int CSCCLCTDigi::getCFEB()      const { return data()->cfeb;     }
 
int CSCCLCTDigi::getBx()        const { return data()->bx;       }
int CSCCLCTDigi::getBX()        const { return data()->bx;       }

int CSCCLCTDigi::getTrknmb()    const { return data()->trknmb;   }

  /// Debug

void CSCCLCTDigi::print() const {

  char stripType = (getStripType() == 0) ? 'D' : 'H';
  char bend      = (getBend()      == 0) ? 'L' : 'R';

  cout <<" Valid: "          <<setw(1)<<isValid()
       <<" Quality: "        <<setw(1)<<getQuality()
       <<" Pattern shape "   <<setw(1)<<getPattern()
       <<" Strip type: "     <<setw(1)<<stripType
       <<" Bend: "           <<setw(1)<<bend
       <<" Strip: "          <<setw(2)<<getStrip()
       <<" Key Strip: "      <<setw(3)<<getKeyStrip()
       <<" Key CFEB ID: "    <<setw(1)<<getCFEB()
       <<" BX: "             <<setw(1)<<getBX()
       <<" Track number: "   <<setw(1)<<getTrknmb()<<endl;
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
