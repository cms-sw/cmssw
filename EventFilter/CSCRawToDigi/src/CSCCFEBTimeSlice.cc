#include "EventFilter/CSCRawToDigi/interface/CSCCFEBTimeSlice.h"
#include <cassert>
#include <iomanip>
#include <stdint.h>

// a Gray code is An ordering of 2n binary numbers such that
// only one bit changes from one entry to the next
// const unsigned layerGrayCode[] = {3,1,5,6,4,2};
const unsigned layerInverseGrayCode[] = {1,5,0,4,2,3};
// const unsigned channelGrayCode[] = {0,1,3,2, 6,7,5,4, 12,13,15,14, 10,11,9,8};
const unsigned channelInverseGrayCode[] = {0,1,3,2, 7,6,4,5, 15,14,12,13, 8,9,11,10};

CSCCFEBTimeSlice::CSCCFEBTimeSlice() 
{
    bzero(this, 99*2);
  dummy = 0x7FFF;
  blank_space_1 = 0x7;
  blank_space_3 = 0x7;
}


CSCCFEBSCAControllerWord::CSCCFEBSCAControllerWord(unsigned short frame)
// trig_time( frame & 0xFF ),
// sca_blk( (frame>>8) & 0xF ),
// l1a_phase((frame>>12( & 0x1),
// lct_phase((frame>>13( & 0x1),
// sca_full((frame>>14) & 0x1),
// ts_flag(((frame>>15) & 0x1)
{
  memcpy(this, &frame, 2);
}


CSCCFEBDataWord * CSCCFEBTimeSlice::timeSample(int layer, int channel, bool isDCFEB) const 
{
  assert(layer >= 1 && layer <= 6);
  assert(channel >=1 && channel <= 16);
  int layerIndex = layerInverseGrayCode[layer-1];

  unsigned channelIndex = channelInverseGrayCode[channel-1];
  if (isDCFEB) channelIndex = channel-1; //!!! New DCFEBs don't use gray coding for channels
  unsigned scaBin = channelIndex*6 + layerIndex;
  assert(scaBin < 96U); // scaBin >= 0, since scaBin is unsigned
  return timeSample(scaBin);
}


CSCCFEBSCAControllerWord CSCCFEBTimeSlice::scaControllerWord(int layer) const 
{
  unsigned int result=0;
  for(unsigned i = 0; i < 16; ++i) {
     result |= timeSample(i*6+layer-1)->controllerData << i;
  }
  return CSCCFEBSCAControllerWord(result);
}


void CSCCFEBTimeSlice::setControllerWord(const CSCCFEBSCAControllerWord & controllerWord) 
{
  for(int layer = 1; layer <= 6; ++layer)
    {
      for(int channel = 1; channel <= 16; ++channel)
	{
	  const unsigned short * shortWord = reinterpret_cast<const unsigned short *>(&controllerWord);
	  timeSample(layer, channel)->controllerData
	    = ( *shortWord >> (channel-1)) & 1;
	}
    }
}


unsigned CSCCFEBTimeSlice::calcCRC() const
{
        unsigned CRC=0;
        for(uint16_t pos=0; pos<96; ++pos)
        CRC=(theSamples[pos]&0x1fff)^((theSamples[pos]&0x1fff)<<1)^(((CRC&0x7ffc)>>2)|((0x0003&CRC)<<13))^((CRC&0x7ffc)>>1);
        return CRC;
}


std::ostream & operator<<(std::ostream & os, const CSCCFEBTimeSlice & slice) 
{
  for(int ichannel = 1; ichannel <= 16; ++ichannel) 
    {
      for(int ilayer = 1; ilayer <= 6; ++ilayer)
	{
	  //unsigned index = (ilayer-1) + (ichannel-1)*6;
	  //int value = (slice.timeSample(index))->adcCounts - 560;
	  int value = (slice.timeSample(ilayer, ichannel))->adcCounts - 560; 
	  os << " " << std::setw(5) << std::dec << value;
	}
      os << std::endl;
    }
  return os;
}

