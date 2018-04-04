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

enum TotemSampicConstant
{
  numberOfSamples = 24,
};

#pragma pack(push,1)
struct TotemSampicData
{
  uint8_t      samples[TotemSampicConstant::numberOfSamples];

  TotemSampicData() {}
};
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicInfo
{
  uint8_t       hwId;
  uint8_t       controlBits[6];
  uint8_t       fpgaTime[5];
  uint16_t      timestampA;
  uint16_t      timestampB;
  uint16_t      cellInfo;
  uint8_t       planeChannelId;
  uint8_t       reserved[5];

  TotemSampicInfo() {}
};
#pragma pack(pop)

#pragma pack(push,1)
struct TotemSampicEventInfo
{
  uint8_t       hwId;
  uint8_t       l1ATimestamp[5];
  uint16_t      bunchNumber;
  uint32_t      orbitNumber;
  uint32_t      eventNumber;
  uint16_t      channelMap;
  uint16_t      l1ALatency;
  uint8_t       numberOfSamples;
  uint8_t       offsetOfSamples;
  uint8_t       fwVersion;
  uint8_t       pllInfo;

  TotemSampicEventInfo() {}
};
#pragma pack(pop)

uint8_t grayToBinary_8bit( const uint8_t &gcode_data )
{
  //b[0] = g[0]
  uint8_t binary_byte = gcode_data & 0x80; // MSB is the same

  //b[i] = g[i] xor b[i-1]
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x40;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x20;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x10;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x08;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x04;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x02;
  binary_byte |= ( gcode_data ^ ( binary_byte >> 1 ) ) & 0x01;

  return binary_byte;
}

/**
 * This class is intended to handle the timing infromation of SAMPIC in the TOTEM implementation
**/
class TotemSampicFrame
{
  public:
    TotemSampicFrame( const uint8_t* chInfoPtr, const uint8_t* chDataPtr, const uint8_t* eventInfoPtr ) :
      totemSampicInfo_( nullptr ), totemSampicData_( nullptr ), totemSampicEventInfo_( nullptr ),
      status_( 0 ) {
      if ( chInfoPtr != nullptr && chDataPtr != nullptr && eventInfoPtr != nullptr) {
        totemSampicInfo_ = (TotemSampicInfo*) chInfoPtr;
        totemSampicData_ = (TotemSampicData*) chDataPtr;
        totemSampicEventInfo_ = (TotemSampicEventInfo*) eventInfoPtr;
      }
      if ( totemSampicEventInfo_->numberOfSamples == TotemSampicConstant::numberOfSamples
        || totemSampicInfo_->controlBits[3] == 0x69 )
        status_ = 1;
    }
    ~TotemSampicFrame() {}

    /// Prints the frame.
    /// If binary is true, binary format is used.
    void printRaw( bool binary = false ) const {
      std::cout << "Event Info: " << std::endl;
      printRawBuffer( (uint16_t*) totemSampicEventInfo_ );

      std::cout << "Channel Info: " << std::endl;
      printRawBuffer( (uint16_t*) totemSampicInfo_ );

      std::cout << "Channel Data: " << std::endl;
      printRawBuffer( (uint16_t*) totemSampicData_ );
    }

    void print() const {
      std::bitset<16> bitsChannelMap( getChannelMap() );
      std::bitset<16> bitsPLLInfo( getPLLInfo() );
      std::cout << "TotemSampicFrame:\nEvent:"
          << "\nHardwareId (Event):\t" << std::hex << (unsigned int) getEventHardwareId()
          << "\nL1A Timestamp:\t" << std::dec << getL1ATimestamp()
          << "\nL1A Latency:\t" << std::dec << getL1ALatency()
          << "\nBunch Number:\t" << std::dec << getBunchNumber()
          << "\nOrbit Number:\t" << std::dec << getOrbitNumber()
          << "\nEvent Number:\t" << std::dec << getEventNumber()
          << "\nChannels fired:\t" << std::hex << bitsChannelMap.to_string()
          << "\nNumber of Samples:\t" << std::dec << getNumberOfSentSamples()
          << "\nOffset of Samples:\t" << std::dec << (int) getOffsetOfSamples()
          << "\nFW Version:\t" << std::hex << (int) getFWVersion()
          << "\nChannel:\nHardwareId:\t" << std::hex << (unsigned int) getHardwareId()
          << "\nFPGATimestamp:\t" << std::dec << getFPGATimestamp()
          << "\nTimestampA:\t" << std::dec << getTimestampA()
          << "\nTimestampA:\t" << std::dec << getTimestampA()
          << "\nCellInfo:\t" << std::dec << getCellInfo()
          << "\nPlane:\t" << std::dec << getDetPlane()
          << "\nChannel:\t" << std::dec << getDetChannel()
          << "\\nPLL Info:\t" << bitsPLLInfo.to_string()
          << std::endl << std::endl;
    }

    // All getters
    inline uint8_t getHardwareId() const {
      return status_ * totemSampicInfo_->hwId;
    }

    inline uint64_t getFPGATimestamp() const {
      uint64_t time = 0;
      for ( unsigned short i = 0; i < 5; ++i )
        time += totemSampicInfo_->fpgaTime[i] << 8*1;
      return status_ * time;
    }

    inline uint16_t getTimestampA() const {
      return status_ * grayToBinary_8bit( totemSampicInfo_->timestampA );
    }

    inline uint16_t getTimestampB() const {
      return status_ * grayToBinary_8bit( totemSampicInfo_->timestampB );
    }

    inline uint16_t getCellInfo() const {
      return status_ * ( totemSampicInfo_->cellInfo & 0x3F );
    }

    inline int getDetPlane() const {
      return status_ * ( ( totemSampicInfo_->planeChannelId & 0xF0 ) >> 4 );
    }

    inline int getDetChannel() const {
      return status_ * ( totemSampicInfo_->planeChannelId & 0xF0 );
    }

    inline int getPLLInfo() const {
      return status_ * totemSampicEventInfo_->pllInfo;
    }

    inline int getFWVersion() const {
      return status_ * totemSampicEventInfo_->fwVersion;
    }

    const std::vector<uint8_t> getSamples() const {
      std::vector<uint8_t> samples;
      if ( status_ ) {
        samples.assign( totemSampicData_->samples, totemSampicData_->samples + TotemSampicConstant::numberOfSamples );
        std::for_each( samples.begin(), samples.end(), &grayToBinary_8bit );
      }
      return samples;
    }

    inline unsigned int getNumberOfSamples() const {
      return status_ * TotemSampicConstant::numberOfSamples;
    }

    // Event Info
    inline uint8_t getEventHardwareId() const {
      return status_ * totemSampicEventInfo_->hwId;
    }

    inline uint64_t getL1ATimestamp() const {
      uint64_t time = 0;
      for ( unsigned short i = 0; i < 5; ++i )
        time += totemSampicEventInfo_->l1ATimestamp[i] << 8*1;
      return status_ * time;
    }

    inline uint16_t getBunchNumber() const
    {
      return status_ * totemSampicEventInfo_->bunchNumber;
    }

    inline uint32_t getOrbitNumber() const
    {
      return status_ * totemSampicEventInfo_->orbitNumber;
    }

    inline uint32_t getEventNumber() const
    {
      return status_ * totemSampicEventInfo_->eventNumber;
    }

    inline uint16_t getChannelMap() const
    {
      return status_ * totemSampicEventInfo_->channelMap;
    }

    inline uint16_t getL1ALatency() const
    {
      return status_ * totemSampicEventInfo_->l1ALatency;
    }

    inline uint8_t getNumberOfSentSamples() const
    {
      return status_ * totemSampicEventInfo_->numberOfSamples;
    }

    inline uint8_t getOffsetOfSamples() const
    {
      return status_ * totemSampicEventInfo_->offsetOfSamples;
    }

    inline bool valid() const {
      return status_ != 0;
    }

  protected:
    const TotemSampicInfo* totemSampicInfo_;
    const TotemSampicData* totemSampicData_;
    const TotemSampicEventInfo* totemSampicEventInfo_;

    int status_;

    inline void printRawBuffer( const uint16_t* buffer, const bool binary = false, const unsigned int size = 12 ) const {
      for ( unsigned int i = 0; i < size; i++ ) {
        if ( binary ) {
          std::bitset<16> bits( *( buffer++ ) );
          std::cout << bits.to_string() << std::endl;
        }
        else
          std::cout << std::setfill( '0' ) << std::setw( 4 ) << std::hex << *( buffer++ ) << std::endl;
      }
      std::cout << std::endl;
    }
};


#endif
