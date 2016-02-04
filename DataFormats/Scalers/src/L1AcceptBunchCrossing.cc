/*
 *   File: DataFormats/Scalers/src/L1AcceptBunchCrossing.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>

L1AcceptBunchCrossing::L1AcceptBunchCrossing() : 
  l1AcceptOffset_(0),
  orbitNumber_(0),
  bunchCrossing_(0),
  eventType_(0)
{ 
}

L1AcceptBunchCrossing::L1AcceptBunchCrossing(const int l1AcceptOffset__,
					     const unsigned int orbitNumber__,
					     const unsigned int bunchCrossing__,
					     const unsigned int eventType__) : 
  l1AcceptOffset_(l1AcceptOffset__),
  orbitNumber_(orbitNumber__),
  bunchCrossing_(bunchCrossing__),
  eventType_(eventType__)
{ 
}

L1AcceptBunchCrossing::L1AcceptBunchCrossing(const int index, 
					     const unsigned long long data)
{ 
  l1AcceptOffset_ =  - index;
  orbitNumber_    = (unsigned int) (( data >> ORBIT_NUMBER_SHIFT ) 
				    & ORBIT_NUMBER_MASK);
  bunchCrossing_  = (unsigned int) (( data >> BUNCH_CROSSING_SHIFT ) 
				    & BUNCH_CROSSING_MASK );
  eventType_      = (unsigned int) (( data >> EVENT_TYPE_SHIFT )
				    & EVENT_TYPE_MASK);
}

L1AcceptBunchCrossing::~L1AcceptBunchCrossing() { } 

/// Pretty-print operator for L1AcceptBunchCrossing
std::ostream& operator<<(std::ostream& s, const L1AcceptBunchCrossing& c) 
{
  char line[128];

  sprintf(line, 
  "L1AcceptBC Offset:%2d  Orbit:%10d [0x%8.8X]  BC:%4d [0x%3.3X]  EvtTyp:%d", 
	  c.l1AcceptOffset(),
	  c.orbitNumber(), 
	  c.orbitNumber(), 
	  c.bunchCrossing(), 
	  c.bunchCrossing(), 
	  c.eventType());
  s << line << std::endl;

  return s;
}
