#ifndef DATAFORMATS_MTDCHANNELIDENTIFIER_H
#define DATAFORMATS_MTDCHANNELIDENTIFIER_H

#include <utility>
#include "FWCore/Utilities/interface/GCC11Compatibility.h"


class MTDChannelIdentifier {

  public:

    static const MTDChannelIdentifier& GetInstance() {
      static const MTDChannelIdentifier instance(6, 5, 12, 9);
      return instance;
    }

    typedef unsigned int PackedDigiType;
    typedef unsigned int ChannelType;
  
    static std::pair<int,int> channelToPixel( int ch) {
      int row = ( ch >>  GetInstance().column_width) & GetInstance().row_mask;
      int col = ch & GetInstance().column_mask;
      return std::pair<int,int>(row,col);
    }
  
    static int pixelToChannel( int row, int col) {
      return (row << GetInstance().column_width) | col;
    }


  private:

    MTDChannelIdentifier(unsigned int row_w, unsigned int column_w,
			 unsigned int time_w, unsigned int adc_w) :
       row_width(row_w), column_width(column_w), adc_width(adc_w)
      ,row_shift(0)
      ,column_shift(row_shift + row_w)
      ,time_shift(column_shift + column_w)
      ,adc_shift(time_shift + time_w)
      ,row_mask(~(~0U << row_w))
      ,column_mask( ~(~0U << column_w))
      ,time_mask(~(~0U << time_w))
      ,adc_mask(~(~0U << adc_w))
      ,rowcol_mask(~(~0U << (column_w+row_w)))
      ,max_row(row_mask)
      ,max_column(column_mask)
      ,max_adc(adc_mask){}

    ~MTDChannelIdentifier() = default;

    MTDChannelIdentifier(const MTDChannelIdentifier&) = delete;
    MTDChannelIdentifier& operator=(const MTDChannelIdentifier&) = delete;
    MTDChannelIdentifier(MTDChannelIdentifier&&) = delete;
    MTDChannelIdentifier& operator=(MTDChannelIdentifier&&) = delete;

    const int row_width;
    const int column_width;
    const int adc_width;
    
    const int row_shift;
    const int column_shift;
    const int time_shift;
    const int adc_shift;
   
    const PackedDigiType row_mask;
    const PackedDigiType column_mask;
    const PackedDigiType time_mask;
    const PackedDigiType adc_mask;
    const PackedDigiType rowcol_mask;
    
    const int max_row;
    const int max_column;
    const int max_adc;

};

#endif
