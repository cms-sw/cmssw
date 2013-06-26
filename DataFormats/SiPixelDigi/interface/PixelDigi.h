#ifndef TRACKINGOBJECTS_PIXELDIGI_H
#define TRACKINGOBJECTS_PIXELDIGI_H

// 25/06/06 - get rid of time(), change adc() from int to undigned short. d.k.

#include <utility>
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

/**
 * Persistent digi for the Pixels.
 */

class PixelDigi {
public:

  typedef unsigned int PackedDigiType;
  typedef unsigned int ChannelType;

  PixelDigi( int packed_value) : theData(packed_value) {}

  PixelDigi( int row, int col, int adc) {
    init( row, col, adc);
  }

  PixelDigi( int chan, int adc) {
    std::pair<int,int> rc = channelToPixel(chan);
    init( rc.first, rc.second, adc);
  }

  PixelDigi() : theData(0)  {}

  void init( int row, int col, int adc);

  // Access to digi information
  int row() const     {return (theData >> PixelChannelIdentifier::thePacking.row_shift) & PixelChannelIdentifier::thePacking.row_mask;}
  int column() const  {return (theData >> PixelChannelIdentifier::thePacking.column_shift) & PixelChannelIdentifier::thePacking.column_mask;}
  //int time() const    {return (theData >> PixelChannelIdentifier::thePacking.time_shift) & PixelChannelIdentifier::thePacking.time_mask;}
  unsigned short adc() const  {return (theData >> PixelChannelIdentifier::thePacking.adc_shift) & PixelChannelIdentifier::thePacking.adc_mask;}
  PackedDigiType packedData() const {return theData;}

  static std::pair<int,int> channelToPixel( int ch) {
    int row = ( ch >> PixelChannelIdentifier::thePacking.column_width) & PixelChannelIdentifier::thePacking.row_mask;
    int col = ch & PixelChannelIdentifier::thePacking.column_mask;
    return std::pair<int,int>(row,col);
  }

  static int pixelToChannel( int row, int col) {
    return (row << PixelChannelIdentifier::thePacking.column_width) | col;
  }

  int channel() const {return PixelChannelIdentifier::pixelToChannel( row(), column());}

 private:
  PackedDigiType theData;
};  

// Comparison operators
inline bool operator<( const PixelDigi& one, const PixelDigi& other) {
  return one.channel() < other.channel();
}

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const PixelDigi& digi) {
  return o << " " << digi.channel()
	   << " " << digi.adc();
}

#endif
