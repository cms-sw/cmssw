/** \file
 *
 *
 * \author Mirko Berretti
 * \author Nicola Minafra
 */

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"

TotemTimingDigi::TotemTimingDigi( const uint8_t hwId,
                                  const uint64_t fpgaTimestamp, const uint16_t timestampA, const uint16_t timestampB,
                                  const uint16_t cellInfo, const std::vector<uint8_t>& samples,
                                  const TotemTimingEventInfo& totemTimingEventInfo ) :
  hwId_( hwId ), fpgaTimestamp_( fpgaTimestamp ), timestampA_( timestampA ), timestampB_( timestampB ),
  cellInfo_( cellInfo ), samples_( samples ), totemTimingEventInfo_( totemTimingEventInfo )
{}

TotemTimingDigi::TotemTimingDigi( const TotemTimingDigi& digi ) :
  hwId_( digi.hwId_ ), fpgaTimestamp_( digi.fpgaTimestamp_ ), timestampA_( digi.timestampA_ ), timestampB_( digi.timestampB_ ),
  cellInfo_( digi.cellInfo_ ), samples_( digi.samples_ ), totemTimingEventInfo_( digi.totemTimingEventInfo_ )
{}

TotemTimingDigi::TotemTimingDigi() :
  hwId_( 0 ), fpgaTimestamp_( 0 ), timestampA_( 0 ), timestampB_( 0 ), cellInfo_( 0 )
{}

// Comparison
bool
TotemTimingDigi::operator==( const TotemTimingDigi& digi ) const
{
  if ( hwId_          != digi.hwId_
    || fpgaTimestamp_ != digi.fpgaTimestamp_
    || timestampA_    != digi.timestampA_
    || timestampB_    != digi.timestampB_
    || cellInfo_      != digi.cellInfo_
    || samples_       != digi.samples_
  ) return false;
  return true;
}

