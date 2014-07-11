#ifndef PHASE2TRACKERDIGI_H
#define PHASE2TRACKERDIGI_H

#include <stdint.h>
#include <utility>
#include <cassert> 

/**
 * Persistent digi for the Phase 2 tracker 
 */

class Phase2TrackerDigi {
public:

  typedef uint16_t PackedDigiType;

  Phase2TrackerDigi( unsigned int packed_channel) : theChannel(packed_channel) {}

  Phase2TrackerDigi( unsigned int row, unsigned int col) {
    theChannel = pixelToChannel(row,col);
  }

  Phase2TrackerDigi() : theChannel(0)  {}

  // Access to digi information - pixel sensors
  unsigned int row()     const { return channelToRow(theChannel); }
  unsigned int column()  const { return channelToColumn(theChannel); }
  // Access to digi information - strip sensors
  unsigned int strip()   const { return row(); }
  unsigned int edge()    const { return column(); } // CD: any better name for that? 
  // Access to the (raw) channel number
  unsigned int channel() const { return theChannel; }

  static std::pair<unsigned int,unsigned int> channelToPixel( unsigned int ch) {
    return std::pair<unsigned int, unsigned int>(channelToRow(ch),channelToColumn(ch));
  }

  static PackedDigiType pixelToChannel( unsigned int row, unsigned int col) {
    assert(row<1016);
    assert(col<32);
    return row|(col<<10);
  }

 private:
  PackedDigiType theChannel;
  static unsigned int channelToRow( unsigned int ch) { return ch & 0x03FF; } // (theChannel & 0x03FF)>>0 
  static unsigned int channelToColumn( unsigned int ch) { return ch >> 10; } // (theChannel & 0xFC00)>>10
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
