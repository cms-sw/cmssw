/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 * $Date: 2006/05/18 19:59:33 $
 * $Revision: 1.7 $
 *
 * \author L.Gray, UF
 */

#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>
#include <iostream>

/// Constructors
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const int itrknmb, const int ivalid,
					   const int iquality,
					   const int ikeywire,
					   const int istrip,
					   const int ipattern, const int ibend,
					   const int ibx) {
  trknmb  = itrknmb;
  valid   = ivalid;
  quality = iquality;
  keywire = ikeywire;
  strip   = istrip;
  pattern = ipattern;
  bend    = ibend;
  bx      = ibx;
}

/// Default
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi() {
  clear(); // set contents to zero
}

/// Clears this LCT.
void CSCCorrelatedLCTDigi::clear() {
  trknmb  = 0;
  valid   = 0;
  quality = 0;
  keywire = 0;
  strip   = 0;
  pattern = 0;
  bend    = 0;
  bx      = 0;
}

/// Comparison
bool CSCCorrelatedLCTDigi::operator==(const CSCCorrelatedLCTDigi &rhs) const {
  return ((trknmb == rhs.trknmb)   && (quality == rhs.quality) &&
	  (keywire == rhs.keywire) && (strip == rhs.strip)     &&
	  (pattern == rhs.pattern) && (bend == rhs.bend)       &&
	  (bx == rhs.bx)           && (valid == rhs.valid)       );
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    std::cout << "CSC LCT #"        << getTrknmb() 
	      << ": Valid = "       << isValid()
	      << " Quality = "      << getQuality()
	      << " Key Wire = "     << getKeyWG()
	      << " Strip = "        << getStrip()
	      << " CLCT Pattern = " << getCLCTPattern()
	      << " Strip Type = "   << ( (getStripType() == 0) ? 'D' : 'H' )
	      << " Bend = "         << ( (getBend() == 0) ? 'L' : 'R' )
	      << " BX = "           << getBX() << std::endl;
  }
  else {
    std::cout << "Not a valid correlated LCT." << std::endl;
  }
}
