#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ME0TriggerDigi::ME0TriggerDigi(const int itrknmb,
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

ME0TriggerDigi::ME0TriggerDigi() {
  clear(); // set contents to zero
}

void ME0TriggerDigi::clear() {
  trknmb  = 0;
  quality = 0;
  strip   = 0;
  pattern = 0;
  bend    = 0;
  bx      = 0;
}

bool ME0TriggerDigi::operator==(const ME0TriggerDigi &rhs) const {
  return ((trknmb == rhs.trknmb) && (quality == rhs.quality) &&
	  (strip == rhs.strip)   && (pattern == rhs.pattern) && 
	  (bend == rhs.bend)     && (bx == rhs.bx) );
}

std::ostream & operator<<(std::ostream & o,
			  const ME0TriggerDigi& digi) {
  return o << "ME0 Trigger #" << digi.getTrknmb()
           << ": Quality = "  << digi.getQuality()
           << " Strip = "     << digi.getStrip()
	   << " Pattern = "   << digi.getPattern()
           << " Bend = "      << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << " BX = "        << digi.getBX()
	   << "\n";
}
