/**\class CSCCorrelatedLCTDigi
 *
 * Digi for Correlated LCT trigger primitives.
 *
 * $Date: 2010/07/07 10:53:22 $
 * $Revision: 1.14 $
 *
 * \author L.Gray, UF
 */

#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include <iostream>

/// Constructors
CSCCorrelatedLCTDigi::CSCCorrelatedLCTDigi(const int itrknmb, const int ivalid,
					   const int iquality,
					   const int ikeywire,
					   const int istrip,
					   const int ipattern, const int ibend,
					   const int ibx, const int impclink, 
					   const uint16_t ibx0,
					   const uint16_t isyncErr, 
					   const uint16_t icscID):
  trknmb(itrknmb),
  valid(ivalid),
  quality(iquality),
  keywire(ikeywire),
  strip(istrip),
  pattern(ipattern),
  bend(ibend),
  bx(ibx),
  mpclink(impclink),
  bx0(ibx0),
  syncErr(isyncErr),
  cscID(icscID)
{}

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
  mpclink = 0;
  bx0     = 0; 
  syncErr = 0;
  cscID   = 0;
}

/// Comparison
bool CSCCorrelatedLCTDigi::operator==(const CSCCorrelatedLCTDigi &rhs) const {
  return ((trknmb == rhs.trknmb)   && (quality == rhs.quality) &&
	  (keywire == rhs.keywire) && (strip == rhs.strip)     &&
	  (pattern == rhs.pattern) && (bend == rhs.bend)       &&
	  (bx == rhs.bx)           && (valid == rhs.valid) && (mpclink == rhs.mpclink) );
}

/// Debug
void CSCCorrelatedLCTDigi::print() const {
  if (isValid()) {
    std::cout << "CSC LCT #"        << getTrknmb() 
	      << ": Valid = "       << isValid()
	      << " Quality = "      << getQuality()
	      << " Key Wire = "     << getKeyWG()
	      << " Strip = "        << getStrip()
              << " Pattern = "      << getPattern()
	      << " Bend = "         << ( (getBend() == 0) ? 'L' : 'R' )
	      << " BX = "           << getBX() 
	      << " MPC Link = "     << getMPCLink() << std::endl;
  }
  else {
    std::cout << "Not a valid correlated LCT." << std::endl;
  }
}

std::ostream & operator<<(std::ostream & o,
			  const CSCCorrelatedLCTDigi& digi) {
  return o << "CSC LCT #"   << digi.getTrknmb()
           << ": Valid = "  << digi.isValid()
           << " Quality = " << digi.getQuality()
           << " MPC Link = " << digi.getMPCLink()
	   << " cscID = "   << digi.getCSCID() << "\n"
           <<"  cathode info: Strip = "    << digi.getStrip()
	   << " Pattern = " << digi.getPattern()
           << " Bend = "    << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           <<"    anode info: Key wire = " << digi.getKeyWG()
           << " BX = "      << digi.getBX()
           << " bx0 = "     << digi.getBX0()
           << " syncErr = " << digi.getSyncErr() << "\n";
}
