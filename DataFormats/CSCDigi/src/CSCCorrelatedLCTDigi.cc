/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 * $Date: 2006/02/24 06:36:25 $
 * $Revision: 1.3 $
 *
 * \author L.Gray, UF
 */

#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <iostream>
#include <bitset>

CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(int trknmb, int valid, int quality,
					   int keywire, int strip, int clct_pattern,
					   int bend, int bx)
{
  set(trknmb, valid, quality, keywire, strip, clct_pattern, bend, bx);
}

CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(ChannelType channel)
{
  ChannelPacking* ch = reinterpret_cast<ChannelPacking*>(&channel);
  set(ch->trknmb, ch->valid, ch->quality, ch->keywire, 
      ch->strip, ch->pattern, ch->bend, ch->bx);  
}

CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(PackedDigiType packed_type)
{
  setData(packed_type);
}

/// Copy
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const CSCCorrelatedLCTDigi& digi)
{
  persistentData = digi.persistentData;
}

/// Default
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi()
{
  set(0, 0, 0, 0, 0, 0, 0, 0);
}

/// Assignment
CSCCorrelatedLCTDigi & 
CSCCorrelatedLCTDigi::operator=(const CSCCorrelatedLCTDigi& digi)
{
  if(&digi != this) persistentData = digi.persistentData;
  return *this;
}

/// Getters
CSCCorrelatedLCTDigi::ChannelType
CSCCorrelatedLCTDigi::channel() const
{
  const PackedDigiType* d = data();
  ChannelPacking result;
  
  result.trknmb  = d->trknmb;
  result.quality = d->quality;
  result.keywire = d->keywire;
  result.strip   = d->strip;
  result.pattern = d->pattern;
  result.bend    = d->bend;
  result.bx      = d->bx;
  result.valid   = d->valid;

  return *(reinterpret_cast<CSCCorrelatedLCTDigi::ChannelType*>(&result));
}

int CSCCorrelatedLCTDigi::getTrknmb()      const { return data()->trknmb; }
int CSCCorrelatedLCTDigi::getValid()       const { return data()->valid; }
int CSCCorrelatedLCTDigi::getQuality()     const { return data()->quality; }
int CSCCorrelatedLCTDigi::getKwire()       const { return data()->keywire; }
int CSCCorrelatedLCTDigi::getStrip()       const { return data()->strip;}
int CSCCorrelatedLCTDigi::getCLCTPattern() const { return ((data()->pattern) & 0x7); }
int CSCCorrelatedLCTDigi::getStriptype()   const { return (((data()->pattern) & 0x8) >> 3); }
int CSCCorrelatedLCTDigi::getBend()        const { return data()->bend; }
int CSCCorrelatedLCTDigi::getBx()          const { return data()->bx; }

/// Debug

void CSCCorrelatedLCTDigi::print() const
{
  std::cout<< "Track number: " << getTrknmb() 
	   << " Quality: " << getQuality()
	   << " Key Wire: " << getKwire()
	   << " Strip: " << getStrip()
	   << " CLCT Pattern: " << getCLCTPattern()
	   << " Strip Type: " << ( (getStriptype() == 0) ? 'D' : 'H' )
	   << " Bend: " << ( (getBend() == 0) ? 'L' : 'R' )
	   << " BX: " << getBx()
	   << " Valid: " << getValid() << std::endl;
}

void CSCCorrelatedLCTDigi::dump() const
{
  typedef std::bitset<8*sizeof(PackedDigiType)> bits;
  std::cout << *reinterpret_cast<const bits*>(data());
}

void CSCCorrelatedLCTDigi::set(int trknmb, int valid, int quality,
	 int keywire, int strip, int clct_pattern, 
	 int bend, int bx)
{
  PackedDigiType* d = data();
  d->trknmb  = trknmb;
  d->quality = quality;
  d->keywire = keywire;
  d->strip   = strip;
  d->pattern = clct_pattern;
  d->bend    = bend;
  d->bx      = bx;
  d->valid   = valid;
}

CSCCorrelatedLCTDigi::PackedDigiType* 
CSCCorrelatedLCTDigi::data() {
  return reinterpret_cast<PackedDigiType*>(&persistentData);
}

const CSCCorrelatedLCTDigi::PackedDigiType* 
CSCCorrelatedLCTDigi::data() const {
  return reinterpret_cast<const PackedDigiType*>(&persistentData);
}

void CSCCorrelatedLCTDigi::setData(PackedDigiType p){
  *(data()) = p;
}
