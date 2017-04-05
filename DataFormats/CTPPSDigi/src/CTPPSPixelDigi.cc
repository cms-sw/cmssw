#include "DataFormats/CTPPSDigi/interface/CTPPSPixelDigi.h"

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
/// row_w 11, column_w 11, adc_w 10 bits

void CTPPSPixelDigi::init(int row, int col, int adc) {

  if(row_width+column_width+adc_width != 32) 
    throw cms::Exception("Invalid CTPPS pixel packing widths") 
      << " row_width = " << row_width 
      << "  column_width = " << column_width 
      << "   adc_width = " << adc_width << ".";
  if(~((adc_mask << adc_shift) | (column_mask << column_shift) | row_mask) != 0) 
    throw cms::Exception("Invalid CTPPS pixel packing masks") 
      << " row_mask = " << row_mask 
      << "  column_mask = " << column_mask 
      << "   adc_mask = " << adc_mask << ".";
  if( ((adc_mask << adc_shift) & (column_mask << column_shift) & row_mask) != 0) 
    throw cms::Exception("Invalid CTPPS pixel packing masks 2") 
      << " row_mask = " << row_mask 
      << "  column_mask = " << column_mask 
      << "   adc_mask = " << adc_mask << ".";

/// Set adc to max_adc in case of overflow
  adc = (uint32_t(adc) > max_adc) ? max_adc : std::max(adc,0);

  theData = (row << row_shift) |
    (col << column_shift) |
    (adc << adc_shift);

}
