/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 * $Date: 2006/03/03 22:29:01 $
 * $Revision: 1.5 $
 *
 * \author L.Gray, UF
 */

#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <iostream>
#include <bitset>

CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(int itrknmb, int ivalid, int iquality,
					   int ikeywire, int istrip, int ipattern,
					   int ibend, int ibx)
{
  trknmb = itrknmb;
  valid  = ivalid;
  quality = iquality;
  keywire = ikeywire;
  strip = istrip;
  pattern = ipattern;
  bend = ibend;
  bx = ibx;
}

/// Default
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi()
{
  trknmb = 0;
  valid  = 0;
  quality = 0;
  keywire = 0;
  strip = 0;
  pattern = 0;
  bend = 0;
  bx = 0;
}

/// Getters
int CSCCorrelatedLCTDigi::getTrknmb()      const { return trknmb; }
int CSCCorrelatedLCTDigi::getValid()       const { return valid;  }
bool CSCCorrelatedLCTDigi::isValid()       const { return valid;  }
int CSCCorrelatedLCTDigi::getQuality()     const { return quality;}
int CSCCorrelatedLCTDigi::getKwire()       const { return keywire;}
int CSCCorrelatedLCTDigi::getKeyWG()       const { return keywire;}
int CSCCorrelatedLCTDigi::getStrip()       const { return strip;  }
int CSCCorrelatedLCTDigi::getPattern() const { return pattern; }
int CSCCorrelatedLCTDigi::getCLCTPattern() const { return (pattern & 0x7); }
int CSCCorrelatedLCTDigi::getStriptype()   const { return ((pattern & 0x8) >> 3); }
int CSCCorrelatedLCTDigi::getStripType()   const { return ((pattern & 0x8) >> 3); }
int CSCCorrelatedLCTDigi::getBend()        const { return bend;   }
int CSCCorrelatedLCTDigi::getBx()          const { return bx;     }
int CSCCorrelatedLCTDigi::getBX()          const { return bx;     }

/// Debug

void CSCCorrelatedLCTDigi::print() const
{
  std::cout<< " Track number: " << getTrknmb() 
	   << " Quality: " << getQuality()
	   << " Key Wire: " << getKeyWG()
	   << " Strip: " << getStrip()
	   << " CLCT Pattern: " << getCLCTPattern()
	   << " Strip Type: " << ( (getStripType() == 0) ? 'D' : 'H' )
	   << " Bend: " << ( (getBend() == 0) ? 'L' : 'R' )
	   << " BX: " << getBX()
	   << " Valid: " << isValid() << std::endl;
}

