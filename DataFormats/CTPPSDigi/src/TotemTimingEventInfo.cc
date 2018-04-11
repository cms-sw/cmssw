/** \file
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 */

#include <DataFormats/CTPPSDigi/interface/TotemTimingEventInfo.h>

TotemTimingEventInfo::TotemTimingEventInfo( const uint8_t hwId, const uint64_t l1ATimestamp,
                                            const uint16_t bunchNumber, const uint32_t orbitNumber, const uint32_t eventNumber,
                                            const uint16_t channelMap, const uint16_t l1ALatency,
                                            const uint8_t numberOfSamples, const uint8_t offsetOfSamples, const uint8_t pllInfo ) :
  hwId_( hwId ), l1ATimestamp_( l1ATimestamp ),
  bunchNumber_( bunchNumber ), orbitNumber_( orbitNumber ), eventNumber_( eventNumber ),
  channelMap_( channelMap ), l1ALatency_( l1ALatency ),
  numberOfSamples_( numberOfSamples ), offsetOfSamples_( offsetOfSamples ), pllInfo_( pllInfo )
{}

TotemTimingEventInfo::TotemTimingEventInfo( const TotemTimingEventInfo& eventInfo ) :
  hwId_( eventInfo.hwId_ ), l1ATimestamp_( eventInfo.l1ATimestamp_ ),
  bunchNumber_( eventInfo.bunchNumber_ ), orbitNumber_( eventInfo.orbitNumber_ ), eventNumber_( eventInfo.eventNumber_ ),
  channelMap_( eventInfo.channelMap_ ), l1ALatency_( eventInfo.l1ALatency_ ),
  numberOfSamples_( eventInfo.numberOfSamples_ ), offsetOfSamples_( eventInfo.offsetOfSamples_ ), pllInfo_ ( eventInfo.pllInfo_ )
{}

TotemTimingEventInfo::TotemTimingEventInfo() :
  hwId_( 0 ), l1ATimestamp_( 0 ),
  bunchNumber_( 0 ), orbitNumber_( 0 ), eventNumber_( 0 ),
  channelMap_( 0 ), l1ALatency_( 0 ),
  numberOfSamples_( 0 ), offsetOfSamples_( 0 ), pllInfo_( 0 )
{}

// Comparison
bool
TotemTimingEventInfo::operator==(const TotemTimingEventInfo& eventInfo) const
{
  if ( hwId_            != eventInfo.hwId_
    || l1ATimestamp_    != eventInfo.l1ATimestamp_
    || bunchNumber_     != eventInfo.bunchNumber_
    || orbitNumber_     != eventInfo.orbitNumber_
    || eventNumber_     != eventInfo.eventNumber_
    || channelMap_      != eventInfo.channelMap_
    || l1ALatency_      != eventInfo.l1ALatency_
    || numberOfSamples_ != eventInfo.numberOfSamples_
    || offsetOfSamples_ != eventInfo.offsetOfSamples_
    || pllInfo_         != eventInfo.pllInfo_
  ) return false;
  return true;
}

