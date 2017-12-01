/*Sushil Dubey, Shashi Dugad, TIFR
 *
 */

#ifndef RAWTODIGIGPU_H
#define RAWTODIGIGPU_H

#include "CablingMapGPU.h"
//typedef unsigned long long Word64;
typedef unsigned int uint; 

const uint layerStartBit_  = 20;
const uint ladderStartBit_ = 12;
const uint moduleStartBit_ = 2;

const uint panelStartBit_  = 10;
const uint diskStartBit_   = 18;
const uint bladeStartBit_  =  12;

const uint layerMask_      = 0xF;
const uint ladderMask_     = 0xFF;
const uint moduleMask_     = 0x3FF;
const uint panelMask_      = 0x3;
const uint diskMask_       = 0xF;
const uint bladeMask_      = 0x3F;

const uint LINK_bits = 6;
const uint ROC_bits  = 5;
const uint DCOL_bits = 5;
const uint PXID_bits = 8;
const uint ADC_bits  = 8;
// special for layer 1
const uint LINK_bits1   = 6;
const uint ROC_bits1    = 5;
const uint COL_bits1_l1 = 6;
const uint ROW_bits1_l1 = 7;

const uint maxROCIndex  = 8;
const uint numRowsInRoc = 80;
const uint numColsInRoc = 52;

// Maximum fed for phase1 is 150 but not all of them are filled
// Update the number FED based on maximum fed found in the cabling map
 const uint MAX_FED  = 150;  
 const uint MAX_LINK = 48; //maximum links/channels for phase1 
 const uint MAX_ROC  = 8;
 const uint MAX_WORD = 2000;//500; 
 const int NSTREAM = 1;
                 
 const uint ADC_shift  = 0;
 const uint PXID_shift = ADC_shift + ADC_bits;
 const uint DCOL_shift = PXID_shift + PXID_bits;
 const uint ROC_shift  = DCOL_shift + DCOL_bits;
 const uint LINK_shift = ROC_shift + ROC_bits1;
// special for layer 1 ROC
 const uint ROW_shift = ADC_shift + ADC_bits;
 const uint COL_shift = ROW_shift + ROW_bits1_l1;

 const uint LINK_mask = ~(~uint(0) << LINK_bits1);
 const uint ROC_mask  = ~(~uint(0) << ROC_bits1);
 const uint COL_mask  = ~(~uint(0) << COL_bits1_l1);
 const uint ROW_mask  = ~(~uint(0) << ROW_bits1_l1);
 const uint DCOL_mask = ~(~uint(0) << DCOL_bits);
 const uint PXID_mask = ~(~uint(0) << PXID_bits);
 const uint ADC_mask  = ~(~uint(0) << ADC_bits); 

struct DetIdGPU {
  uint RawId;
  uint rocInDet;
  uint moduleId;
};

struct Pixel {
 uint row;
 uint col;
};

 //CablingMap *Map;
 //GPU specific
 uint *word_d, *fedIndex_d, *eventIndex_d;       // Device copy of input data
 uint *xx_d, *yy_d,*xx_adc, *yy_adc, *moduleId_d, *adc_d, *layer_d, *rawIdArr_d;  // Device copy
 // store the start and end index for each module (total 1856 modules-phase 1)
 cudaStream_t stream[NSTREAM];
 int *mIndexStart_d, *mIndexEnd_d; 
 // CablingMap *Map;
#endif
