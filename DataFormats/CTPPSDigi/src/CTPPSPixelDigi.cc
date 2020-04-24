#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"

/// row_w 11, column_w 11, adc_w 10 bits
const uint32_t CTPPSPixelDigi::row_shift = 0; 
const uint32_t CTPPSPixelDigi::column_shift = 11; 
const uint32_t CTPPSPixelDigi::adc_shift = 22;
const uint32_t CTPPSPixelDigi::row_width = 11; 
const uint32_t CTPPSPixelDigi::column_width = 11; 
const uint32_t CTPPSPixelDigi::adc_width = 10;
const uint32_t CTPPSPixelDigi::row_mask = 0x7FF; 
const uint32_t CTPPSPixelDigi::column_mask = 0x7FF;
const uint32_t CTPPSPixelDigi::adc_mask = 0x3FF; 
const uint32_t CTPPSPixelDigi::rowcol_mask = 0x3FFFFF;
const uint32_t CTPPSPixelDigi::max_row = 0x7FF; 
const uint32_t CTPPSPixelDigi::max_column = 0x7FF; 
const uint32_t CTPPSPixelDigi::max_adc = 0x3FF;

const uint32_t CTPPSPixelDigi::column_width_ch = 11; 
const uint32_t CTPPSPixelDigi::column_mask_ch = 0x7FF;
const uint32_t CTPPSPixelDigi::row_mask_ch = 0x7FF;

void CTPPSPixelDigi::init(int row, int col, int adc) {

/// Set adc to max_adc in case of overflow
  adc = (uint32_t(adc) > max_adc) ? max_adc : std::max(adc,0);

  theData = (row << row_shift) |
    (col << column_shift) |
    (adc << adc_shift);

}
