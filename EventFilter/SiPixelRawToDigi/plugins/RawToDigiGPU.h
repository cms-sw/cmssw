/*Sushil Dubey, Shashi Dugad, TIFR
 *
 */

#ifndef RAWTODIGIGPU_H
#define RAWTODIGIGPU_H

#include <cuda_runtime.h>

#include "SiPixelFedCablingMapGPU.h"
#include<algorithm>

const uint32_t layerStartBit_   = 20;
const uint32_t ladderStartBit_  = 12;
const uint32_t moduleStartBit_  = 2;

const uint32_t panelStartBit_   = 10;
const uint32_t diskStartBit_    = 18;
const uint32_t bladeStartBit_   = 12;

const uint32_t layerMask_       = 0xF;
const uint32_t ladderMask_      = 0xFF;
const uint32_t moduleMask_      = 0x3FF;
const uint32_t panelMask_       = 0x3;
const uint32_t diskMask_        = 0xF;
const uint32_t bladeMask_       = 0x3F;

const uint32_t LINK_bits        = 6;
const uint32_t ROC_bits         = 5;
const uint32_t DCOL_bits        = 5;
const uint32_t PXID_bits        = 8;
const uint32_t ADC_bits         = 8;
// special for layer 1
const uint32_t LINK_bits1       = 6;
const uint32_t ROC_bits1        = 5;
const uint32_t COL_bits1_l1     = 6;
const uint32_t ROW_bits1_l1     = 7;
const uint32_t OMIT_ERR_bits    = 1;

const uint32_t maxROCIndex      = 8;
const uint32_t numRowsInRoc     = 80;
const uint32_t numColsInRoc     = 52;

const uint32_t MAX_WORD = 2000;

const uint32_t ADC_shift  = 0;
const uint32_t PXID_shift = ADC_shift + ADC_bits;
const uint32_t DCOL_shift = PXID_shift + PXID_bits;
const uint32_t ROC_shift  = DCOL_shift + DCOL_bits;
const uint32_t LINK_shift = ROC_shift + ROC_bits1;
// special for layer 1 ROC
const uint32_t ROW_shift = ADC_shift + ADC_bits;
const uint32_t COL_shift = ROW_shift + ROW_bits1_l1;
const uint32_t OMIT_ERR_shift = 20;

const uint32_t LINK_mask = ~(~uint32_t(0) << LINK_bits1);
const uint32_t ROC_mask  = ~(~uint32_t(0) << ROC_bits1);
const uint32_t COL_mask  = ~(~uint32_t(0) << COL_bits1_l1);
const uint32_t ROW_mask  = ~(~uint32_t(0) << ROW_bits1_l1);
const uint32_t DCOL_mask = ~(~uint32_t(0) << DCOL_bits);
const uint32_t PXID_mask = ~(~uint32_t(0) << PXID_bits);
const uint32_t ADC_mask  = ~(~uint32_t(0) << ADC_bits);
const uint32_t ERROR_mask = ~(~uint32_t(0) << ROC_bits1);
const uint32_t OMIT_ERR_mask = ~(~uint32_t(0) << OMIT_ERR_bits);

struct DetIdGPU {
  uint32_t RawId;
  uint32_t rocInDet;
  uint32_t moduleId;
};

struct Pixel {
 uint32_t row;
 uint32_t col;
};


namespace gpudetails{

class Packing {
  public:
  using PackedDigiType = uint32_t;
    
    // Constructor: pre-computes masks and shifts from field widths
__host__ __device__
inline
    constexpr Packing(unsigned int row_w, unsigned int column_w,
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

							   
    uint32_t  row_width;
    uint32_t  column_width;
    uint32_t  adc_width;
    
    uint32_t  row_shift;
    uint32_t  column_shift;
    uint32_t  time_shift;
    uint32_t  adc_shift;
   
    PackedDigiType row_mask;
    PackedDigiType column_mask;
    PackedDigiType time_mask;
    PackedDigiType adc_mask;
    PackedDigiType rowcol_mask;
    
    
    uint32_t  max_row;
    uint32_t  max_column;
    uint32_t  max_adc;
  };


// const PixelChannelIdentifier::Packing PixelChannelIdentifier::thePacking( 11, 11, 0, 10); // row, col, time, adc


__host__ __device__
inline
constexpr gpudetails::Packing packing() { return gpudetails::Packing(11, 11, 0, 10);}

}

// constexpr Packing thePacking = packing();

__host__ __device__
inline uint32_t pack(uint32_t row, uint32_t col, uint32_t adc) {
  constexpr gpudetails::Packing thePacking = gpudetails::packing();
  adc = std::min(adc, thePacking.max_adc);

  return (row << thePacking.row_shift) |
    (col << thePacking.column_shift) |
    (adc << thePacking.adc_shift);

}


// configuration and memory buffers alocated on the GPU
struct context {
  cudaStream_t stream;

  uint32_t * word_d;
  uint8_t * fedId_d;
  uint32_t * pdigi_d;
  uint16_t * xx_d;
  uint16_t * yy_d;
  uint16_t * xx_adc;
  uint16_t * yy_adc;
  uint32_t * moduleId_d;
  uint16_t * adc_d;
  uint16_t * layer_d;
  uint32_t * rawIdArr_d;
  uint32_t * errType_d;
  uint32_t * errWord_d;
  uint32_t * errFedID_d;
  uint32_t * errRawID_d;

  // store the start and end index for each module (total 1856 modules-phase 1)
  int *mIndexStart_d;
  int *mIndexEnd_d;
};


// wrapper function to call RawToDigi on the GPU from host side
void RawToDigi_wrapper(context &, const SiPixelFedCablingMapGPU* cablingMapDevice, const uint32_t wordCounter, uint32_t *word, 
                        const uint32_t fedCounter,  uint8_t *fedId_h,
                        bool convertADCtoElectrons, uint32_t * pdigi_h, int *mIndexStart_h, int *mIndexEnd_h, 
                        uint32_t *rawIdArr_h, uint32_t *errType_h, uint32_t *errWord_h, uint32_t *errFedID_h, uint32_t *errRawID_h,
                        bool useQualityInfo, bool includeErrors, bool debug = false);

// void initCablingMap();
context initDeviceMemory();
void freeMemory(context &);

// reference cmssw/RecoLocalTracker/SiPixelClusterizer
// all are runtime const, should be specified in python _cfg.py
struct ADCThreshold {
  const int thePixelThreshold = 1000; // default Pixel threshold in electrons
  const int theSeedThreshold = 1000; //seed thershold in electrons not used in our algo
  const float theClusterThreshold = 4000; // Cluster threshold in electron
  const int ConversionFactor = 65;  // adc to electron conversion factor

  // following are the default value
  // it should be i python script
  const int theStackADC_  = 255; // the maximum adc count for stack layer
  const int theFirstStack_ = 5; // the index of the fits stack layer
  const double theElectronPerADCGain_ = 600; //ADC to electron conversion
};

#endif
