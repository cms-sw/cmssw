#ifndef DataFormats_Phase2TrackerDigi_Phase2TrackerDigi_H
#define DataFormats_Phase2TrackerDigi_Phase2TrackerDigi_H

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

  Phase2TrackerDigi( unsigned int row, unsigned int col, bool ot_flag) {    
    theChannel = pixelToChannel(row,col);
    if (ot_flag) theChannel |= (1<< 15);
  }
  
  Phase2TrackerDigi() : theChannel(0)  {}

  // Access to digi information - pixel sensors
  unsigned int row()     const { return channelToRow(theChannel); }
  unsigned int column()  const { return channelToColumn(theChannel); }
  uint16_t packedPosition() const { return 0x7FFF & theChannel; }
  // Access to digi information - strip sensors
  unsigned int strip()   const { return row(); }
  unsigned int edge()    const { return column(); } // CD: any better name for that? 
  // Access to the (raw) channel number
  unsigned int channel() const { return theChannel; }
  // Access Overthreshold bit
  bool overThreshold() const { return (otBit(theChannel) ? true : false); }

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
  static unsigned int channelToRow( unsigned int ch) { return ch & 0x03FF; } 
  static unsigned int channelToColumn( unsigned int ch) { return ((ch >> 10) & 0x1F); } 
  static unsigned int otBit( unsigned int ch) { return ((ch >> 15) & 0x1) ; } 
};  

// Comparison operators
inline bool operator<( const Phase2TrackerDigi& one, const Phase2TrackerDigi& other) {
  return one.packedPosition() < other.packedPosition();
}

// distance operators
inline int operator-( const Phase2TrackerDigi& one, const Phase2TrackerDigi& other) {
  return int(one.packedPosition()) - int(other.packedPosition());
}


#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const Phase2TrackerDigi& digi) {
  return o << " " << digi.channel();
}

#endif // DataFormats_Phase2TrackerDigi_Phase2TrackerDigi_H
