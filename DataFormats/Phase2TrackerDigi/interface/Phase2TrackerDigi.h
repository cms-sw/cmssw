#ifndef PHASE2TRACKERDIGI_H
#define PHASE2TRACKERDIGI_H

#include "boost/cstdint.hpp"
#include <assert.h> 

/**
 * Persistent digi for the Phase 2 tracker 
 */

class Phase2TrackerDigi {
public:

  typedef uint16_t PackedDigiType;

  Phase2TrackerDigi( unsigned int packed_channel) : theChannel(packed_channel) {}

  Phase2TrackerDigi( unsigned int row, unsigned int col) {
    assert(row<1016);
    assert(col<32);
    theChannel = (row%1016)|((col%32)<<10);
  }

  Phase2TrackerDigi() : theChannel(0)  {}

  // Access to digi information - pixel sensors
  unsigned int row()     const { return (theChannel & 0x03FF)>>0 ; }
  unsigned int column()  const { return (theChannel & 0xFC00)>>10; }
  // Access to digi information - strip sensors
  unsigned int strip()   const { return row(); }
  unsigned int edge()    const { return column(); } // CD: any better name for that? 
  // Access to the (raw) channel number
  unsigned int channel() const { return theChannel; }

  static std::pair<unsigned int,unsigned int> channelToPixel( unsigned int ch) {
    unsigned int row = (ch & 0x03FF)>>0 ;
    unsigned int col = (ch & 0xFC00)>>10;
    return std::pair<unsigned int, unsigned int>(row,col);
  }

  static int pixelToChannel( unsigned int row, unsigned int col) {
    assert(row<1016);
    assert(col<32);
    return (row%1016)|((col%32)<<10);
  }

 private:
  PackedDigiType theChannel;
};  

// Comparison operators
inline bool operator<( const Phase2TrackerDigi& one, const Phase2TrackerDigi& other) {
  return one.channel() < other.channel();
}

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const Phase2TrackerDigi& digi) {
  return o << " " << digi.channel();
}

#endif // PHASE2TRACKERDIGI_H
