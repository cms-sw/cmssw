#ifndef CTPPSDigi_TotemTimingDigi_h
#define CTPPSDigi_TotemTimingDigi_h

/** \class TotemTimingDigi
 *
 * Digi Class for CTPPS Timing Detector
 *  
 * \author Mirko Berretti
 * \author Nicola Minafra
 * March 2018
 */

#include <cstdint>
#include <vector>

#include <DataFormats/CTPPSDigi/interface/TotemTimingEventInfo.h>

class TotemTimingDigi{

  public:  
    TotemTimingDigi(const uint8_t hwId, const uint64_t FPGATimeStamp, const uint16_t TimeStampA, const uint16_t TimeStampB, const uint16_t CellInfo, const std::vector< uint8_t>& Samples, const TotemTimingEventInfo& totemTimingEventInfo);
    TotemTimingDigi(const TotemTimingDigi& digi);
    TotemTimingDigi();
    ~TotemTimingDigi() {};
  
    /// Digis are equal if they have all the same values, NOT checking the samples!
    bool operator==(const TotemTimingDigi& digi) const;

    /// Return digi values number
  
    /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
    inline unsigned int getHardwareId() const 
    { 
      return hwId_; 
    }
    
    inline unsigned int getHardwareBoardId() const
    {
      return (hwId_&0xE0)>>5;
    }
    
    inline unsigned int getHardwareSampicId() const
    {
      return (hwId_&0x10)>>4;
    }
    
    inline unsigned int getHardwareChannelId() const
    {
      return (hwId_&0x0F);
    }
    
    inline unsigned int getFPGATimeStamp() const
    {
      return FPGATimeStamp_;
    }
    
    inline unsigned int getTimeStampA() const
    {
      return TimeStampA_;
    }
    
    inline unsigned int getTimeStampB() const
    {
      return TimeStampB_;
    }
    
    inline unsigned int getCellInfo() const
    {
      return CellInfo_;
    }
    
    inline std::vector< uint8_t > getSamples() const
    {
      return samples_;
    }
    
    inline std::vector< uint8_t >::const_iterator getSamplesBegin() const
    {
      return samples_.cbegin();
    }
    
    inline std::vector< uint8_t >::const_iterator getSamplesEnd() const
    {
      return samples_.cend();
    }
    
    inline unsigned int getNumberOfSamples() const
    {
      return samples_.size();
    }
    
    inline int getSampleAt( const unsigned int i ) const
    {
      int sampleValue = -1;
      if ( i < samples_.size() ) sampleValue = (int) samples_.at(i);
      return sampleValue;
    }
    
    inline TotemTimingEventInfo getEventInfo() const
    {
      return totemTimingEventInfo_;
    }
       

    /// Set digi values
    /// Hardware Id formatted as: bits 0-3 Channel Id, bit 4 Sampic Id, bits 5-7 Digitizer Board Id
    inline void setHardwareId(const uint8_t hwId) 
    { 
      hwId_ = hwId; 
    }
    
    inline void setHardwareBoardId(const unsigned int BoardId)
    {
      hwId_ &= 0x1F;      // Set board bits to 0
      hwId_ |= ((BoardId&0x07)<<5) & 0xE0;
    }
    
    inline void setHardwareSampicId(const unsigned int SampicId)
    {
      hwId_ &= 0xEF;      // Set sampic bit to 0
      hwId_ |= ((SampicId&0x01)<<4) & 0x10;
    }
    
    inline void setHardwareChannelId(const unsigned int ChannelId)
    {
      hwId_ &= 0xF0;      // Set sampic bit to 0
      hwId_ |= (ChannelId&0x0F) & 0x0F;
    }
    
    inline void setFPGATimeStamp(const uint64_t FPGATimeStamp)
    {
      FPGATimeStamp_ = FPGATimeStamp;
    }
    
    inline void setTimeStampA(const uint16_t TimeStampA)
    {
      TimeStampA_ = TimeStampA;
    }
    
    inline void setTimeStampB(const uint16_t TimeStampB)
    {
      TimeStampB_ = TimeStampB;
    }
    
    inline void setCellInfo(const uint16_t CellInfo)
    {
      CellInfo_ = CellInfo & 0x3F;
    }
    
    inline void setSamples(const std::vector< uint8_t >& samples)
    {
      samples_ = samples;
    }
    
    inline void addSample(const uint8_t sampleValue)
    {
      samples_.emplace_back(sampleValue);
    }
    
    inline void setSampleAt( const unsigned int i, const uint8_t sampleValue )
    {
      if ( i < samples_.size() ) samples_.at(i) = sampleValue;
    }
    
    inline void setEventInfo( const TotemTimingEventInfo& totemTimingEventInfo )
    {
      totemTimingEventInfo_ = totemTimingEventInfo;
    }
    



  private:
    uint8_t hwId_;
    uint64_t FPGATimeStamp_;
    uint16_t TimeStampA_;
    uint16_t TimeStampB_;
    uint16_t CellInfo_;
    
    std::vector< uint8_t > samples_;
    
    TotemTimingEventInfo totemTimingEventInfo_;
    
};

#include <iostream>


inline bool operator< (const TotemTimingDigi& one, const TotemTimingDigi& other)
{
  if ( one.getEventInfo() < other.getEventInfo() )
    return true;
  if ( one.getHardwareId() < other.getHardwareId() )                                     
    return true; 
  return false;
}  


inline std::ostream & operator<<(std::ostream & o, const TotemTimingDigi& digi)
{
  return o << "TotemTimingDigi:"
	   << "\nHardwareId:\t" << std::hex << digi.getHardwareId()
           << "\nDB: " << std::dec << digi.getHardwareBoardId() << "\tSampic: " << digi.getHardwareSampicId() << "\tChannel: " << digi.getHardwareChannelId() 
           << "\nFPGATimeStamp:\t" << std::hex << digi.getFPGATimeStamp()
           << "\nTimeStampA:\t" << std::hex << digi.getTimeStampA()
           << "\nTimeStampA:\t" << std::hex << digi.getTimeStampA()
           << "\nCellInfo:\t" << std::hex << digi.getCellInfo()
           << "\nNumberOfSamples:\t" << std::dec << digi.getNumberOfSamples()
           << std::endl << digi.getEventInfo() << std::endl;
           
}

#endif

