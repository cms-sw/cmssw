/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*   Seyed Mohsen Etesami (setesami@cern.ch)
*   Nicola Minafra
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_DiamondVFATFrame
#define EventFilter_CTPPSRawToDigi_DiamondVFATFrame

#include <vector>
#include <cstddef>
#include <cstdint>

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h" 

#pragma pack(push,1)
struct TotemSampicData{
  uint8_t      sample[24];
   
  TotemSampicData() {};
}
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicInfo{
  uint16_t      reserved[3];
  uint16_t      CellInfo;
  uint16_t      TimestampA;
  uint16_t      TimestampB;
  uint8_t       FPGATime[5];
  uint16_t      ADC_EOC;
  uint8_t       controlBits[4];
  uint8_t       hwId;
   
  TotemSampicInfo() {};
}
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicEventInfo{
  uint8_t       reserved[2];
  uint8_t       offsetOfSamples;
  uint8_t       numberOfSamples;
  uint16_t      L1ALatency;
  uint16_t      channelMap;
  uint32_t      eventNumber;
  uint32_t      orbitNumber;
  uint16_t      bunchNumber;
  uint64_t      L1ATimeStamp;
  uint8_t       hwId;
   
  TotemSampicEventInfo() {};
}
#pragma pack(pop)


/** 
 * This class is intended to handle the timing infromation of SAMPIC in the TOTEM implementation
**/
class TotemSampicFrame
{
  public:
    TotemSampicFrame(const uint8_t* inputData = nullptr)
    {
      
    }
    ~TotemSampicFrame() {}

    
    
    
    
  protected:
    /** Raw data frame as sent by electronics.
    * The container is organized as follows:
    * Even IndexinFiber: Ch Data
    * \verbatim
    * buffer index      content         size
    * ---------------------------------------------------------------
    *   0->23           Channel data    sampic 8bit samples
    * \endverbatim
    *    
    * Odd IndexinFiber: Ch Info
    * \verbatim
    * buffer index      content         size
    * ---------------------------------------------------------------
    *   0->5            Empty           48 bit
    *   6->7            Cell Info       16 bit 
    *   8->9            TimestampA      16 bit 
    *   10->11          TimestampB      16 bit 
    *   12->16          FPGATime        40 bit 
    *   17->19          ADC EOC         16 bit 
    *   20->22          controlBits     32 bit 
    *   23              hwId            8 bit 
    * \endverbatim
    **/
    word data[12];

};                                                                     

#endif
