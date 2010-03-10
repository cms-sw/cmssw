/*
 *   File: DataFormats/Scalers/src/EventCounter0.cc   (W.Badgett)
 */

#include "DataFormats/Scalers/interface/EventCounter0.h"
#include "DataFormats/Scalers/interface/ScalersRaw.h"
#include <cstdio>

EventCounter0::EventCounter0() : 
  offset_(0),
  orbitNumber_(0)
{ for ( unsigned int i=0; i<N_SPARE; i++ ) { spare_.push_back(0ULL);}}

EventCounter0::EventCounter0(const int offset__,
			     const unsigned int orbitNumber__,
			     const unsigned long long * spare__) :
  offset_(offset__),
  orbitNumber_(orbitNumber__)
{ for ( unsigned int i=0; i<N_SPARE; i++ ) { spare_.push_back(spare__[i]);}}

EventCounter0::EventCounter0(const int index, 
			     const unsigned long long * data)
{ 
  offset_ =  - index;
  orbitNumber_    = (unsigned int) (( data[0] >> ORBIT_NUMBER_SHIFT ) 
				    & ORBIT_NUMBER_MASK);
  for ( unsigned int i=0; i<N_SPARE; i++) { spare_.push_back(data[i+1]); }
}

EventCounter0::~EventCounter0() { } 

/// Pretty-print operator for EventCounter0
std::ostream& operator<<(std::ostream& s, const EventCounter0& c) 
{
  char line[128];

  sprintf(line, 
	  "EventCounter0 Offset:%2d  Orbit:%10u [0x%8.8X]",
	  c.offset(),
	  c.orbitNumber(),
	  c.orbitNumber());
  s << line << std::endl;

  int length = c.spare().size();
  for ( int i=0; i<length; i++)
  {
    sprintf(line, 
	    "        Spare Offset:%2d   Data:%20llu [0x%16.16llX]",
	    i, c.spare(i), c.spare(i));
    s << line << std::endl;
  }

  return s;
}
