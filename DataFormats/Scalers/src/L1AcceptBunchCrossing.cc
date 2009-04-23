/*
 *   File: DataFormats/Scalers/src/L1AcceptBunchCrossing.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"

L1AcceptBunchCrossing::L1AcceptBunchCrossing() : 
  l1AcceptOffset_(0),
  orbitNumber_(0),
  bunchCrossing_(0),
  eventType_(0)
{ 
}

L1AcceptBunchCrossing::L1AcceptBunchCrossing(int l1AcceptOffset__,
					     unsigned int orbitNumber__,
					     unsigned int bunchCrossing__,
					     unsigned int eventType__) : 
  l1AcceptOffset_(l1AcceptOffset__),
  orbitNumber_(orbitNumber__),
  bunchCrossing_(bunchCrossing__),
  eventType_(eventType__)
{ 
}

L1AcceptBunchCrossing::L1AcceptBunchCrossing(int index, 
					     const unsigned long long data)
{ 
  l1AcceptOffset_ =  - index;
  orbitNumber_    = (unsigned int) (( data >> 32ULL ) & 0xFFFFFFFFULL);
  bunchCrossing_  = (unsigned int) (( data >> 4ULL ) & 0xFFFULL);
  eventType_      = (unsigned int) ( data & 0xFULL);
}

L1AcceptBunchCrossing::~L1AcceptBunchCrossing() { } 

/// Pretty-print operator for L1AcceptBunchCrossing
std::ostream& operator<<(std::ostream& s, const L1AcceptBunchCrossing& c) 
{
  char line[128];
  s << "L1AcceptBunchCrossing    L1AcceptOffset: " << c.l1AcceptOffset() 
    << std::endl;

  sprintf(line, "  OrbitNumber: %10d   BunchCrossing: %4d   EventType: %d", 
	  c.orbitNumber(), c.bunchCrossing(), c.eventType());
  s << line << std::endl;

  return s;
}
