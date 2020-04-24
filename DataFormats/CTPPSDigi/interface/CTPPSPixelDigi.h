#ifndef CTPPS_CTPPSDigi_CTPPSPixelDigi_h
#define CTPPS_CTPPSDigi_CTPPSPixelDigi_h
/**
 * Persistent digi for CTPPS Pixels.
 * Author: F.Ferro ferro@ge.infn.it
 */

#include "FWCore/Utilities/interface/Exception.h"

class CTPPSPixelDigi {
public:


  CTPPSPixelDigi( int packed_value) : theData(packed_value) {}

  CTPPSPixelDigi( int row, int col, int adc) {
    init( row, col, adc);
  }

  CTPPSPixelDigi( int chan, int adc) {
    std::pair<int,int> rc = channelToPixel(chan);
    init( rc.first, rc.second, adc);
  }

  CTPPSPixelDigi() : theData(0)  {}

  /// Access to digi information
  int row() const     {return (theData >> row_shift) & row_mask;}
  int column() const  {return (theData >> column_shift) & column_mask;} 
  unsigned short adc() const  {return (theData >> adc_shift) & adc_mask;}
  uint32_t packedData() const {return theData;}

  static std::pair<int,int> channelToPixel( int ch) {
    int row = ( ch >> column_width_ch) & row_mask_ch;
    int col = ch & column_mask_ch;
    return std::pair<int,int>(row,col);
  }

  static int pixelToChannel( int row, int col) {
    return (row << column_width_ch) | col;
  }

  int channel() const {return pixelToChannel( row(), column());}

/// const values for digi packing with bit structure: adc_bits+col_bits+row_bits
  static const uint32_t row_shift, column_shift, adc_shift;
  static const uint32_t row_mask, column_mask, adc_mask, rowcol_mask;
  static const uint32_t row_width, column_width, adc_width;
  static const uint32_t max_row, max_column, max_adc;

/// const values for channel definition with bit structure: row_bits+col_bits
  static const uint32_t column_width_ch; 
  static const uint32_t column_mask_ch;
  static const uint32_t row_mask_ch;

 private:

  void init( int row, int col, int adc) ;
  uint32_t theData;
};  

/// Comparison operator

inline bool operator<( const CTPPSPixelDigi& one, const CTPPSPixelDigi& other) {
  return (one.packedData()&CTPPSPixelDigi::rowcol_mask) < (other.packedData()&CTPPSPixelDigi::rowcol_mask);
}

#include<iostream>
inline std::ostream & operator<<(std::ostream & o, const CTPPSPixelDigi& digi) {
  return o << " " << digi.row() << " " << digi.column()
	   << " " << digi.adc();
}

#endif
