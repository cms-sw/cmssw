/**********************************************************
*
* Seyed Mohsen Etesami (setesami@cern.ch)    
*
**********************************************************/

#ifndef EventFilter_CTPPSRawToDigi_DiamondVFATFrame
#define EventFilter_CTPPSRawToDigi_DiamondVFATFrame

#include <vector>
#include <cstddef>
#include <stdint.h>

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h" 


/** 
 * This class intended to handle the timing infromation of diamond VFAT frame
**/
class DiamondVFATFrame : public VFATFrame
{
  
  public:
    DiamondVFATFrame(const word* inputData = NULL)
    {}
    virtual ~DiamondVFATFrame() {}

    /// get timing infromation
    uint32_t getLeadingEdgeTime() const
    {
      uint32_t time = ((data[7]&0x1f)<<16)+data[8];
      time = (time & 0xFFE7FFFF) << 2 | (time & 0x00180000) >> 19;    //HPTDC inperpolation bits are MSB but should be LSB.... ask HPTDC designers...
      return time;     
    }

    uint32_t getTrailingEdgeTime() const
    {
      uint32_t time = ((data[5]&0x1f)<<16)+data[6];
      time = (time & 0xFFE7FFFF) << 2 | (time & 0x00180000) >> 19;   //HPTDC inperpolation bits are MSB but should be LSB.... ask HPTDC designers...
      return time;
    }
    
    uint32_t getThresholdVoltage() const
    {
      return ((data[3]&0x7ff)<<16)+data[4];
    }

    VFATFrame::word getMultihit() const
    {
      return data[2] & 0x01;
    }

    VFATFrame::word getHptdcErrorFlag() const
    {
      return data[1] & 0xFFFF;
    }

};                                                                     

#endif
