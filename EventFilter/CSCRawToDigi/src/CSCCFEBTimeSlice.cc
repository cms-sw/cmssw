#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include <cassert>
#include <iomanip>


// a Gray code is An ordering of 2n binary numbers such that
// only one bit changes from one entry to the next
unsigned layerGrayCode[] = {3,1,5,6,4,2};
unsigned layerInverseGrayCode[] = {1,5,0,4,2,3};
unsigned channelGrayCode[] = {0,1,3,2, 6,7,5,4, 12,13,15,14, 10,11,9,8};
unsigned channelInverseGrayCode[] = {0,1,3,2, 7,6,4,5, 15,14,12,13, 8,9,11,10};


CSCCFEBDataWord * CSCCFEBTimeSlice::timeSample(int layer, int channel) const {
  assert(layer >= 1 && layer <= 6);
  assert(channel >=1 && channel <= 16);
  int layerIndex = layerInverseGrayCode[layer-1];
  unsigned channelIndex = channelInverseGrayCode[channel-1];
  unsigned scaBin = channelIndex*6 + layerIndex;
  assert(scaBin >= 0 && scaBin < 96);
  return timeSample(scaBin);
}


CSCCFEBSCAControllerWord CSCCFEBTimeSlice::scaControllerWord(int layer) const {
  unsigned int result=0;
  for(unsigned i = 0; i < 16; ++i) {
     result |= timeSample(i*6+layer-1)->controllerData << i;
  }
  return *(CSCCFEBSCAControllerWord *)(&result);
}


void CSCCFEBTimeSlice::setControllerWord(const CSCCFEBSCAControllerWord & controllerWord) {
  for(int layer = 1; layer <= 6; ++layer) {
    for(int channel = 1; channel <= 16; ++channel) {
       unsigned short * shortWord = (unsigned short *) &controllerWord;
       timeSample(layer, channel)->controllerData
         = ( *shortWord >> (channel-1)) & 1;
    }
  }
}


std::ostream & operator<<(std::ostream & os, const CSCCFEBTimeSlice & slice) {
  for(int ichannel = 1; ichannel <= 16; ++ichannel) {
    for(int ilayer = 1; ilayer <= 6; ++ilayer) {
      //unsigned index = (ilayer-1) + (ichannel-1)*6;
      //int value = (slice.timeSample(index))->adcCounts - 560;
      int value = (slice.timeSample(ilayer, ichannel))->adcCounts - 560; 
      os << " " << std::setw(5) << std::dec << value;
    }
    os << std::endl;
  }
  return os;
}

