#ifndef TRACKINGOBJECTS_PIXELDIGI_H
#define TRACKINGOBJECTS_PIXELDIGI_H

// 25/06/06 - get rid of time(), change adc() from int to undigned short. d.k.

#include <utility>
#include <algorithm>
#include "DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h"

/**
 * Persistent digi for the Pixels.
 */

class PixelDigi {
public:
  typedef unsigned int PackedDigiType;
  typedef unsigned int ChannelType;

  explicit PixelDigi(PackedDigiType packed_value) : theData(packed_value) {}

  PixelDigi(int row, int col, int adc) { init(row, col, adc); }
  PixelDigi(int row, int col, int adc, int flag) { init(row, col, adc, flag); }

  PixelDigi(int chan, int adc) {
    std::pair<int, int> rc = channelToPixel(chan);
    init(rc.first, rc.second, adc);
  }

  PixelDigi() : theData(0) {}

  void init(int row, int col, int adc, int flag = 0) {
#ifdef FIXME_DEBUG
    // This check is for the maximal row or col number that can be packed
    // in a PixelDigi. The actual number of rows or columns in a detector
    // may be smaller!
    // it is done much better in Raw2Digi...
    if (row < 0 || row > PixelChannelIdentifier::thePacking.max_row || col < 0 ||
        col > PixelChannelIdentifier::thePacking.max_column) {
      std::cout << "PixelDigi constructor: row or column out packing range " << row << ' ' << col << std::endl;
    }
#endif

    // Set adc to max_adc in case of overflow
    adc = (adc > PixelChannelIdentifier::thePacking.max_adc) ? PixelChannelIdentifier::thePacking.max_adc
                                                             : std::max(adc, 0);

    theData = (row << PixelChannelIdentifier::thePacking.row_shift) |
              (col << PixelChannelIdentifier::thePacking.column_shift) |
              (adc << PixelChannelIdentifier::thePacking.adc_shift) |
              (flag << PixelChannelIdentifier::thePacking.flag_shift);
  }

  // Access to digi information
  int row() const {
    return (theData >> PixelChannelIdentifier::thePacking.row_shift) & PixelChannelIdentifier::thePacking.row_mask;
  }
  int column() const {
    return (theData >> PixelChannelIdentifier::thePacking.column_shift) &
           PixelChannelIdentifier::thePacking.column_mask;
  }
  int flag() const {
    return (theData >> PixelChannelIdentifier::thePacking.flag_shift) & PixelChannelIdentifier::thePacking.flag_mask;
  }
  unsigned short adc() const {
    return (theData >> PixelChannelIdentifier::thePacking.adc_shift) & PixelChannelIdentifier::thePacking.adc_mask;
  }
  PackedDigiType packedData() const { return theData; }

  static std::pair<int, int> channelToPixel(int ch) {
    int row = (ch >> PixelChannelIdentifier::thePacking.column_width) & PixelChannelIdentifier::thePacking.row_mask;
    int col = ch & PixelChannelIdentifier::thePacking.column_mask;
    return std::pair<int, int>(row, col);
  }

  static int pixelToChannel(int row, int col) { return (row << PixelChannelIdentifier::thePacking.column_width) | col; }

  int channel() const { return PixelChannelIdentifier::pixelToChannel(row(), column()); }

private:
  PackedDigiType theData;
};

// Comparison operators

//inline bool operator<( const PixelDigi& one, const PixelDigi& other) {
//  return one.channel() < other.channel();
//}

inline bool operator<(const PixelDigi& one, const PixelDigi& other) {
  return (one.packedData() & PixelChannelIdentifier::thePacking.rowcol_mask) <
         (other.packedData() & PixelChannelIdentifier::thePacking.rowcol_mask);
}

#include <iostream>
inline std::ostream& operator<<(std::ostream& o, const PixelDigi& digi) {
  return o << " " << digi.channel() << " " << digi.adc();
}

#endif
