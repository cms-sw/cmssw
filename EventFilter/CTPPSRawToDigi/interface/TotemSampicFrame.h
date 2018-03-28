/****************************************************************************
*
* This is a part of the TOTEM offline software.
* Authors:
*   Nicola Minafra
*
****************************************************************************/

#ifndef EventFilter_CTPPSRawToDigi_TotemSampicFrame
#define EventFilter_CTPPSRawToDigi_TotemSampicFrame

#include <vector>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <bitset>

#include "EventFilter/CTPPSRawToDigi/interface/VFATFrame.h" 

enum TotemSampicConstant {
  NumberOfSamples = 24,
};

#pragma pack(push,1)
struct TotemSampicData{
  uint8_t      samples[TotemSampicConstant::NumberOfSamples];
   
  TotemSampicData() {};
};
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicInfo{
  uint8_t       hwId;
  uint8_t       controlBits[6];
  uint8_t       FPGATime[5];
  uint16_t      TimeStampA;
  uint16_t      TimeStampB;
  uint16_t      CellInfo;
  uint8_t       PlaneChannelId;
  uint8_t       reserved[5];
   
  TotemSampicInfo() {};
};
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicEventInfo{
  uint8_t       hwId;
  uint8_t       L1ATimeStamp[5];
  uint16_t      bunchNumber;
  uint32_t      orbitNumber;
  uint32_t      eventNumber;
  uint16_t      channelMap;
  uint16_t      L1ALatency;
  uint8_t       numberOfSamples;
  uint8_t       offsetOfSamples;
  uint8_t       FWVersion;
  uint8_t       PLLInfo;
   
  TotemSampicEventInfo() {};
};
#pragma pack(pop)

uint8_t GrayToBinary_8bit(const uint8_t &gcode_data)
{
  //b[0] = g[0]   
  uint8_t binary_byte=(gcode_data & 0x80); //MSB is the same

  //b[i] = g[i] xor b[i-1]
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x40;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x20;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x10;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x08;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x04;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x02;
  binary_byte |= (gcode_data ^ (binary_byte >> 1)) & 0x01;

  return binary_byte;
}



/** 
 * This class is intended to handle the timing infromation of SAMPIC in the TOTEM implementation
**/
class TotemSampicFrame
{
  public:
    TotemSampicFrame(const uint8_t* chInfoPtr, const uint8_t* chDataPtr, const uint8_t* eventInfoPtr) : 
    TotemSampicInfo_(nullptr), TotemSampicData_(nullptr), TotemSampicEventInfo_(nullptr), status_(0)
    {
      if ( chInfoPtr!=nullptr && chDataPtr!=nullptr && eventInfoPtr!=nullptr) {
        TotemSampicInfo_ = (TotemSampicInfo*) chInfoPtr;
        TotemSampicData_ = (TotemSampicData*) chDataPtr;
        TotemSampicEventInfo_ = (TotemSampicEventInfo*) eventInfoPtr;
      }
      if ( TotemSampicEventInfo_->numberOfSamples == TotemSampicConstant::NumberOfSamples || TotemSampicInfo_->controlBits[3] == 0x69 )
        status_ = 1;
    }
    ~TotemSampicFrame() {}

    
    /// Prints the frame.
    /// If binary is true, binary format is used.
    void PrintRaw(bool binary = false) const
    {
      std::cout << "Event Info: " << std::endl;
      PrintRawBuffer( (uint16_t*) TotemSampicEventInfo_ );
      
      std::cout << "Channel Info: " << std::endl;
      PrintRawBuffer( (uint16_t*) TotemSampicInfo_ );
      
      std::cout << "Channel Data: " << std::endl;
      PrintRawBuffer( (uint16_t*) TotemSampicData_ );
    }
    
    void Print() const
    {
      std::bitset<16> bitsChannelMap( getChannelMap() );
      std::bitset<16> bitsPLLInfo( getPLLInfo() );
      std::cout << "TotemSampicFrame:\nEvent:"
          << "\nHardwareId (Event):\t" << std::hex << (unsigned int) getEventHardwareId()
          << "\nL1A Time Stamp:\t" << std::dec << getL1ATimeStamp()
          << "\nL1A Latency:\t" << std::dec << getL1ALatency()
          << "\nBunch Number:\t" << std::dec << getBunchNumber()
          << "\nOrbit Number:\t" << std::dec << getOrbitNumber()
          << "\nEvent Number:\t" << std::dec << getEventNumber()
          << "\nChannels fired:\t" << std::hex << bitsChannelMap.to_string()
          << "\nNumber of Samples:\t" << std::dec << getNumberOfSentSamples()
          << "\nOffset of Samples:\t" << std::dec << (int) getOffsetOfSamples()
          << "\nFW Version:\t" << std::hex << (int) getFWVersion()
          << "\nChannel:\nHardwareId:\t" << std::hex << (unsigned int) getHardwareId()
          << "\nFPGATimeStamp:\t" << std::dec << getFPGATimeStamp()
          << "\nTimeStampA:\t" << std::dec << getTimeStampA()
          << "\nTimeStampA:\t" << std::dec << getTimeStampA()
          << "\nCellInfo:\t" << std::dec << getCellInfo()
          << "\nPlane:\t" << std::dec << getDetPlane()
          << "\nChannel:\t" << std::dec << getDetChannel()
          << "\\nPLL Info:\t" << bitsPLLInfo.to_string()
          << std::endl << std::endl; 
    }
              
    // All getters
    inline uint8_t getHardwareId() const 
    { 
      return status_ * TotemSampicInfo_->hwId; 
    }
  
    inline uint64_t getFPGATimeStamp() const
    {
      uint64_t time = 0;
      for (int i=0; i<5; ++i) 
        time += TotemSampicInfo_->FPGATime[i] << 8*1;
      return status_ * time;
    }
    
    inline uint16_t getTimeStampA() const
    {
      return status_ * GrayToBinary_8bit(TotemSampicInfo_->TimeStampA);
    }
    
    inline uint16_t getTimeStampB() const
    {
      return status_ * GrayToBinary_8bit(TotemSampicInfo_->TimeStampB);
    }
    
    inline uint16_t getCellInfo() const
    {
      return status_ * (TotemSampicInfo_->CellInfo & 0x3F);
    }
    
    inline int getDetPlane() const
    {
      return status_ * ((TotemSampicInfo_->PlaneChannelId & 0xF0)>>4);
    }
    
    inline int getDetChannel() const
    {
      return status_ * (TotemSampicInfo_->PlaneChannelId & 0xF0);
    }
    
    inline int getPLLInfo() const
    {
      return status_ * TotemSampicEventInfo_->PLLInfo;
    }
    
    inline int getFWVersion() const
    {
      return status_ * TotemSampicEventInfo_->FWVersion;
    }
    
    const std::vector< uint8_t > getSamples() const
    {
      std::vector< uint8_t > samples;
      if (status_) {
        samples.assign( TotemSampicData_->samples, TotemSampicData_->samples + TotemSampicConstant::NumberOfSamples);
        std::for_each(samples.begin(), samples.end(), &GrayToBinary_8bit);
      }
      return samples;
    }
    
    inline unsigned int getNumberOfSamples() const
    {
      return status_ * TotemSampicConstant::NumberOfSamples;
    }
    
    
    
    // Event Info
    inline uint8_t getEventHardwareId() const 
    { 
      return status_ * TotemSampicEventInfo_->hwId; 
    }
        
    inline uint64_t getL1ATimeStamp() const
    {
      uint64_t time = 0;
      for (int i=0; i<5; ++i) 
        time += TotemSampicEventInfo_->L1ATimeStamp[i] << 8*1;
      return status_ * time;
    }
    
    inline uint16_t getBunchNumber() const
    {
      return status_ * TotemSampicEventInfo_->bunchNumber;
    }
    
    inline uint32_t getOrbitNumber() const
    {
      return status_ * TotemSampicEventInfo_->orbitNumber;
    }
    
    inline uint32_t getEventNumber() const
    {
      return status_ * TotemSampicEventInfo_->eventNumber;
    }
    
    inline uint16_t getChannelMap() const
    {
      return status_ * TotemSampicEventInfo_->channelMap;
    }
    
    inline uint16_t getL1ALatency() const
    {
      return status_ *TotemSampicEventInfo_->L1ALatency;
    }
    
    inline uint8_t getNumberOfSentSamples() const
    {
      return status_ * TotemSampicEventInfo_->numberOfSamples;
    }
    
    inline uint8_t getOffsetOfSamples() const
    {
      return status_ * TotemSampicEventInfo_->offsetOfSamples;
    }
        
    
    
    inline bool isOK() const
    {
      return status_;
    }
    
    
    
    
  protected:
    const TotemSampicInfo* TotemSampicInfo_;
    const TotemSampicData* TotemSampicData_;
    const TotemSampicEventInfo* TotemSampicEventInfo_;
    
    int status_;
    
    
    inline void PrintRawBuffer(uint16_t const* buffer, const bool binary=false, const unsigned int size=12) const {
      for (unsigned int i = 0; i < size; i++) { 
        if (binary) {
          std::bitset<16> bits( *(buffer++) );
          std::cout << bits.to_string() << std::endl;
        }
        else
          std::cout << std::setfill('0') << std::setw(4) << std::hex << *(buffer++) << std::endl;
      }
      std::cout << std::endl;
    }
    
};   


#endif
