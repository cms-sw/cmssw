#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>

ME0TriggerDigi::ME0TriggerDigi(const int itrknmb,
			       const int iquality,
			       const int istrip,
			       const int ipartition,
			       const int ipattern, 
			       const int ibend,
			       const int ibx):
  trknmb_(itrknmb),
  quality_(iquality),
  strip_(istrip),
  partition_(ipartition),
  pattern_(ipattern),
  bend_(ibend),
  bx_(ibx)
{}

ME0TriggerDigi::ME0TriggerDigi() {
  clear(); // set contents to zero
}

void ME0TriggerDigi::clear() {
  trknmb_  = 0;
  quality_ = 0;
  strip_   = 0;
  partition_   = 0;
  pattern_ = 0;
  bend_    = 0;
  bx_      = 0;
}

bool ME0TriggerDigi::operator==(const ME0TriggerDigi &rhs) const {
  return ((trknmb_ == rhs.trknmb_) && (quality_ == rhs.quality_) &&
	  (strip_ == rhs.strip_)   && (partition_ == rhs.partition_)   && 
	  (pattern_ == rhs.pattern_) && 
	  (bend_ == rhs.bend_)     && (bx_ == rhs.bx_) );
}

std::ostream & operator<<(std::ostream & o,
			  const ME0TriggerDigi& digi) {
  return o << "ME0 Trigger #" << digi.getTrknmb()
           << ": Quality = "  << digi.getQuality()
           << " Strip = "     << digi.getStrip()
           << " Partition = "     << digi.getPartition()
	   << " Pattern = "   << digi.getPattern()
           << " Bend = "      << ((digi.getBend() == 0) ? 'L' : 'R') << "\n"
           << " BX = "        << digi.getBX()
	   << "\n";
}
