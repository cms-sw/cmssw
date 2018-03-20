/** \file
 * 
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 */

#include <DataFormats/CTPPSDigi/interface/TotemTimingEventInfo.h>

using namespace std;

TotemTimingEventInfo::TotemTimingEventInfo(const uint8_t hwId, const uint64_t L1ATimeStamp, const uint16_t bunchNumber, const uint32_t orbitNumber, const uint32_t eventNumber, const uint16_t channelMap, const uint16_t L1ALatency, const uint8_t numberOfSamples, const uint8_t offsetOfSamples ) :
  hwId_(hwId), L1ATimeStamp_(L1ATimeStamp), bunchNumber_(bunchNumber), orbitNumber_(orbitNumber), eventNumber_(eventNumber), channelMap_(channelMap), L1ALatency_(L1ALatency), numberOfSamples_(numberOfSamples), offsetOfSamples_(offsetOfSamples) 
{}

TotemTimingEventInfo::TotemTimingEventInfo(const TotemTimingEventInfo& eventInfo) :
  hwId_(eventInfo.getHardwareId()), L1ATimeStamp_(eventInfo.getL1ATimeStamp()), bunchNumber_(eventInfo.getBunchNumber()), orbitNumber_(eventInfo.getOrbitNumber()), eventNumber_(eventInfo.getEventNumber()), channelMap_(eventInfo.getChannelMap()), L1ALatency_(eventInfo.getL1ALatency()), numberOfSamples_(eventInfo.getNumberOfSamples()), offsetOfSamples_(eventInfo.getOffsetOfSamples())
{}

TotemTimingEventInfo::TotemTimingEventInfo() :
  hwId_(0), L1ATimeStamp_(0), bunchNumber_(0), orbitNumber_(0), eventNumber_(0), channelMap_(0), L1ALatency_(0), numberOfSamples_(0), offsetOfSamples_(0)
{}


// Comparison
bool
TotemTimingEventInfo::operator==(const TotemTimingEventInfo& eventInfo) const
{
  if ( hwId_            !=      eventInfo.getHardwareId()
    || L1ATimeStamp_    !=      eventInfo.getL1ATimeStamp()
    || bunchNumber_     !=      eventInfo.getBunchNumber()
    || orbitNumber_     !=      eventInfo.getOrbitNumber()
    || eventNumber_     !=      eventInfo.getEventNumber()
    || channelMap_      !=      eventInfo.getChannelMap()
    || L1ALatency_      !=      eventInfo.getL1ALatency()
    || numberOfSamples_ !=      eventInfo.getNumberOfSamples()
    || offsetOfSamples_ !=      eventInfo.getOffsetOfSamples()
  ) return false;
  else  
    return true; 
} 

