// Modify the pixel packing to make 100micron pixels possible. d.k. 2/02
//
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"

#include <algorithm>

void PixelDigi::init( int row, int col, int adc) {
  // This check is for the maximal row or col number that can be packed
  // in a PixelDigi. The actual number of rows or columns in a detector
  // may be smaller!
  if ( row < 0 || row > PixelChannelIdentifier::thePacking.max_row || 
       col < 0 || col > PixelChannelIdentifier::thePacking.max_column) {
    std::cout << "PixelDigi constructor: row or column out packing range" << std::endl;
  }

  // Set adc to max_adc in case of overflow
  adc = (adc > PixelChannelIdentifier::thePacking.max_adc) ? PixelChannelIdentifier::thePacking.max_adc : std::max(adc,0);

  theData = (row << PixelChannelIdentifier::thePacking.row_shift) | 
    (col << PixelChannelIdentifier::thePacking.column_shift) | 
    (adc << PixelChannelIdentifier::thePacking.adc_shift);
}

