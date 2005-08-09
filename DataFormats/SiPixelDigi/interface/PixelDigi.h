#ifndef TRACKINGOBJECTS_PIXELDIGI_H
#define TRACKINGOBJECTS_PIXELDIGI_H

#include <utility>

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
  int row() const     {return (theData >> thePacking.row_shift) & thePacking.row_mask;}
  int column() const  {return (theData >> thePacking.column_shift) & thePacking.column_mask;}
  int time() const    {return (theData >> thePacking.time_shift) & thePacking.time_mask;}
  int adc() const     {return (theData >> thePacking.adc_shift) & thePacking.adc_mask;}
  PackedDigiType packedData() const {return theData;}

  static std::pair<int,int> channelToPixel( int ch) {
    int row = ( ch >> thePacking.column_width) & thePacking.row_mask;
    int col = ch & thePacking.column_mask;
    return std::pair<int,int>(row,col);
  }

  static int pixelToChannel( int row, int col) {
    return (row << thePacking.column_width) | col;
  }

  int channel() const {return pixelToChannel( row(), column());}

 private:
  PackedDigiType theData;
  /**
   * Pack the pixel information to use less memory
   */
  class Packing {
  public:

    // Constructor: pre-computes masks and shifts from field widths
    Packing(const int row_w, const int column_w, 
	    const int time_w, const int adc_w);

    // public data:
    int adc_shift;
    int time_shift;
    int row_shift;
    int column_shift;

    PackedDigiType adc_mask;
    PackedDigiType time_mask;
    PackedDigiType row_mask;
    PackedDigiType column_mask;

    int row_width;
    int column_width;
    int adc_width;

    int max_row;
    int max_column;
    int max_adc;
  };

  static Packing   thePacking;
};  

// Comparison operators
inline bool operator<( const PixelDigi& one, const PixelDigi& other) {
  return one.channel() < other.channel();
}

#include<iostream>
// needed by COBRA
inline std::ostream & operator<<(std::ostream & o, const PixelDigi & digi) {
  return o << digi.channel();
}

#endif
