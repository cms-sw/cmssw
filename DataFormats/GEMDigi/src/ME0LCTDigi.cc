/**\class ME0LCTDigi
 *
 * Digi for ME0 LCT trigger primitives.
 *
 * Sven Dildick (TAMU)
 */

#include "DataFormats/GEMDigi/interface/ME0LCTDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

/// Constructors
ME0LCTDigi::ME0LCTDigi(const int itrknmb,
		       const int iquality,
		       const int istrip,
		       const int ipattern, 
		       const int ibend,
		       const int ibx):
  trknmb(itrknmb),
  quality(iquality),
  strip(istrip),
  pattern(ipattern),
  bend(ibend),
  bx(ibx)
{}

/// Default
ME0LCTDigi::ME0LCTDigi() {
  clear(); // set contents to zero
}

/// Clears this LCT.
void ME0LCTDigi::clear() {
  trknmb  = 0;
  quality = 0;
  strip   = 0;
  pattern = 0;
  bend    = 0;
  bx      = 0;
}

/// Comparison
bool ME0LCTDigi::operator==(const ME0LCTDigi &rhs) const {
  return ((trknmb == rhs.trknmb) && (quality == rhs.quality) &&
	  (strip == rhs.strip)   && (pattern == rhs.pattern) && 
	  (bend == rhs.bend)     && (bx == rhs.bx) );
}

/// Debug
void ME0LCTDigi::print() const {
  if (getPattern()==0) {
    edm::LogVerbatim("ME0LCTDigi")
              << "CSC LCT #"        << getTrknmb() 
	      << ": Quality = "      << getQuality()
	      << " Strip = "        << getStrip()
              << " Pattern = "      << getPattern()
	      << " Bend = "         << ( (getBend() == 0) ? 'L' : 'R' )
	      << " BX = "           << getBX();
  }
  else {
    edm::LogVerbatim("ME0LCTDigi") << "Not a valid ME0 LCT.";
  }
}

std::ostream & operator<<(std::ostream & o,
			  const ME0LCTDigi& digi) {
  return o << "CSC LCT #"   << digi.getTrknmb()
           << ": Quality = " << digi.getQuality()
           <<"  Strip = "    << digi.getStrip()
	   << " Pattern = " << digi.getPattern()
           << " Bend = "    << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << " BX = "      << digi.getBX()
	   << "\n";
}
