#ifndef DATAFORMATS_PHASE2TRACKERSTUB_H
#define DATAFORMATS_PHASE2TRACKERSTUB_H

#include <utility>
#include "boost/cstdint.hpp"
#include <assert.h> 

class Phase2TrackerStub {
public:
  
  Phase2TrackerStub():theChannel_(0),bend_(0),bxOffset_(0) {}

  Phase2TrackerStub( unsigned int halfstrip, unsigned int edge, unsigned int bend, unsigned int bx_offset):bend_(bend),bxOffset_(bx_offset) {
    assert(halfstrip<2032);
    assert(edge<2);
    theChannel_ = (halfstrip%2032)|((edge%2)<<12); // this is not like a digi!
  }

  unsigned int edge() const { return ((theChannel_&0x800)>>12); }

  float strip() const { return (theChannel_&0x7FF)/2.; }

  std::pair<unsigned int,float> barycenter() const { return std::make_pair(edge(),strip()); }

  unsigned int bx_offset() const { return bxOffset_; }

  unsigned int triggerBend() const { return bend_; }
  
private:

  uint16_t theChannel_;
  uint8_t  bend_;
  uint8_t  bxOffset_;

};

// Comparison operators
inline bool operator<( const Phase2TrackerStub& one, const Phase2TrackerStub& other) {
  if (one.edge()==other.edge())
    return one.strip() < other.strip();
  else 
    return one.edge()<other.edge();
}

#endif // DATAFORMATS_PHASE2TRACKERSTUB_H
