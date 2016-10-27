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
    DiamondVFATFrame(const word* _data = NULL): timepresenceFlags(15)// by default LEDT, TEDT, TheVoltage Multihit are present
    {}
    virtual ~DiamondVFATFrame() {}

    /// get timing infromation
    uint32_t getLeadingEdgeTime() const
    {
      uint32_t time = ((data[6]&0x1f)<<16)+data[5];
      time = (time & 0xFFE7FFFF) << 2 | (time & 0x00180000) >> 19;    //HPTDC inperpolation bits are MSB but should be LSB.... ask HPTDC designers...
      return time;     
    }

    uint32_t getTrailingEdgeTime() const
    {
      uint32_t time = ((data[8]&0x1f)<<16)+data[7];
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


    /// Sets timing information presence flags.

    void setTimingPresenceFlags(uint8_t v)
    {
      timepresenceFlags = v;
    }
    /// Returns true if the leading edge time  word is present in the frame.
    bool isLeadingEdgeTimePresent() const
    {
      return timepresenceFlags & 0x1;
    }

    /// Returns true if the trainling edge time  word is present in the frame.
    bool isTrailingEdgeTimePresent() const
    {
      return timepresenceFlags & 0x2;
    } 
  
    /// Returns true if the threshold voltage  word is present in the frame.
    bool isThresholdVoltagePresent() const
    {
      return timepresenceFlags & 0x4;
    }
 
    /// Returns true if the multi hit  word is present in the frame.
    bool isMultiHitPresent() const
    {
      return timepresenceFlags & 0x8;
    }



    /// Returns false if any of the groups (in LEDTime, TEDTime, Threshold Voltage and Multi hit  words) is present but wrong. 
    bool checkTimeinfo() const
    {
      if (isLeadingEdgeTimePresent() && (data[7] & 0xF800) != 0x6000)
        return false;

      if (isTrailingEdgeTimePresent() && (data[5] & 0xF800) != 0x6800)
        return false;

      if (isThresholdVoltagePresent() && (data[3] & 0xF800) != 0x7000)
        return false;

      if (isMultiHitPresent() && (data[2] & 0xF800) != 0x7800)
        return false;

      return true;
    }

  private:
    /** Organization of the Raw Data for Diamond VFAT in buffer index 1->8 is as follwoing. 
    * The rest is the same as standard VFAT structure  
    * \verbatim
    * buffer index   content       size
    * ---------------------------------------------------------------
    *   0            CRC                    16 bits

    *   1            HPTDC error            16 bits
    *   2            Multi hit              5 constant bits (01111) 10 empty bits + 1 bit
    *   3,4          Threshold Voltage      5 constant bits (01110) +27 bits
    *   5,6          Trailing Etime         5 constant bits (01101) +27 bits
    *   7,8          Leading Etime          5 constant bits (01100) +27 bits

    *   9            ChipID                 4 constant bits (1110) + 12 bits
    *   10           EC, Flags              4 constant bits (1100) + 8, 4 bits
    *   11           BC                     4 constant bits (1010) + 12 bits
    * \endverbatim
    **/


    /// Flag indicating the presence of timing components.
    ///   bit 1: "Leading edge time  word" (buffer index 7)
    ///   bit 2: "Trailing edge time word" (buffer index 5)
    ///   bit 3: "threshould voltage word" (buffer index 3)
    ///   bit 4: "multihit           word" (buffer index 2)
    uint8_t timepresenceFlags;




};                                                                     

#endif
