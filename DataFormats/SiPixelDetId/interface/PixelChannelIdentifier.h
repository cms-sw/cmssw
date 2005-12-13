#ifndef DATAFORMATS_PIXELCHANMNELIDENTIFIER_H
#define DATAFORMATS_PIXELCHANMNELIDENTIFIER_H

#include <utility>

class PixelChannelIdentifier{
 public:
  
  typedef unsigned int PackedDigiType;
  typedef unsigned int ChannelType;
  
  static std::pair<int,int> channelToPixel( int ch) {
    int row = ( ch >> thePacking.column_width) & thePacking.row_mask;
    int col = ch & thePacking.column_mask;
    return std::pair<int,int>(row,col);
  }
  
  static int pixelToChannel( int row, int col) {
    return (row << thePacking.column_width) | col;
  }
  
 private:
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
  
 public:
  static Packing   thePacking;
};  


#endif
