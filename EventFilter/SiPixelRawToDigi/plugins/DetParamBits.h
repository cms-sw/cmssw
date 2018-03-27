// Sushil Dubey, Shashi Dugad, TIFR
#ifndef DETPARAMBITS_H
#define DETPARAMBITS_H
typedef unsigned int uint;
//reference
//http://cmsdoxygen.web.cern.ch/cmsdoxygen/CMSSW_9_2_0/doc/html/d3/db2/PixelROC_8cc_source.html#l00197 
const uint layerStartBit_  = 20;
const uint ladderStartBit_ = 12;
const uint moduleStartBit_ = 2;

const uint panelStartBit_  = 10;
const uint diskStartBit_   = 18;
const uint bladeStartBit_  = 12;

const uint layerMask_      = 0xF;
const uint ladderMask_     = 0xFF;
const uint moduleMask_     = 0x3FF;
const uint panelMask_      = 0x3;
const uint diskMask_       = 0xF;
const uint bladeMask_      = 0x3F;

// __host__ __device__ bool isBarrel(uint rawId) {
//   return (1==((rawId>>25)&0x7));
// }

__host__ __device__ int getLayer(uint rawId) {
  int layer  = (rawId >> layerStartBit_)  & layerMask_;
  return layer;
}

__host__ __device__ int getDisk(uint rawId) {
  // int side =1;
  // unsigned int panel = ((rawId>>panelStartBit_) & panelMask_);
  // if(panel==1) side = -1;
   unsigned int disk = int((rawId>>diskStartBit_) & diskMask_);
  // return disk*side;
  return disk;
}
#endif
