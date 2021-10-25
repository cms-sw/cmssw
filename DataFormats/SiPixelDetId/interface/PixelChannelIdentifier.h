#ifndef DataFormats_SiPixelDetId_interface_PixelChannelIdentifier_h
#define DataFormats_SiPixelDetId_interface_PixelChannelIdentifier_h

#include <utility>
#include <cstdint>

namespace pixelchannelidentifierimpl {
  /**
   * Pack the pixel information to use less memory
   */

  class Packing {
  public:
    using PackedDigiType = uint32_t;

    // Constructor: pre-computes masks and shifts from field widths
    constexpr Packing(unsigned int row_w, unsigned int column_w, unsigned int flag_w, unsigned int adc_w)
        : row_width(row_w),
          column_width(column_w),
          adc_width(adc_w),
          row_shift(0),
          column_shift(row_shift + row_w),
          flag_shift(column_shift + column_w),
          adc_shift(flag_shift + flag_w),
          row_mask(~(~0U << row_w)),
          column_mask(~(~0U << column_w)),
          flag_mask(~(~0U << flag_w)),
          adc_mask(~(~0U << adc_w)),
          rowcol_mask(~(~0U << (column_w + row_w))),
          max_row(row_mask),
          max_column(column_mask),
          max_adc(adc_mask) {}

    const uint32_t row_width;
    const uint32_t column_width;
    const uint32_t adc_width;

    const uint32_t row_shift;
    const uint32_t column_shift;
    const uint32_t flag_shift;
    const uint32_t adc_shift;

    const PackedDigiType row_mask;
    const PackedDigiType column_mask;
    const PackedDigiType flag_mask;
    const PackedDigiType adc_mask;
    const PackedDigiType rowcol_mask;

    const int max_row;
    const int max_column;
    const int max_adc;
  };
}  // namespace pixelchannelidentifierimpl

class PixelChannelIdentifier {
public:
  typedef unsigned int PackedDigiType;
  typedef unsigned int ChannelType;

  static std::pair<int, int> channelToPixel(int ch) {
    int row = (ch >> thePacking.column_width) & thePacking.row_mask;
    int col = ch & thePacking.column_mask;
    return std::pair<int, int>(row, col);
  }

  static int pixelToChannel(int row, int col) { return (row << thePacking.column_width) | col; }

  using Packing = pixelchannelidentifierimpl::Packing;

public:
  constexpr static Packing packing() { return Packing(8, 9, 4, 11); }

  constexpr static Packing thePacking = {11, 10, 1, 10};
};

#endif  // DataFormats_SiPixelDetId_interface_PixelChannelIdentifier_h
