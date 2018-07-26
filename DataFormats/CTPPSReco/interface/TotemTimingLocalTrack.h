/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Mateusz Szpyrka (mateusz.szpyrka@cern.ch)
 *
 ****************************************************************************/


#ifndef DataFormats_CTPPSReco_TotemTimingLocalTrack
#define DataFormats_CTPPSReco_TotemTimingLocalTrack

#include "DataFormats/CTPPSReco/interface/CTPPSTimingLocalTrack.h"

//----------------------------------------------------------------------------------------------------

class TotemTimingLocalTrack : public CTPPSTimingLocalTrack
{
  public:
    TotemTimingLocalTrack() {}
    TotemTimingLocalTrack( const math::XYZPoint& pos0, const math::XYZPoint& pos0_sigma,
                           float t, float t_sigma ) :
      CTPPSTimingLocalTrack( pos0, pos0_sigma, t, t_sigma ) {}

    // no specific class members yet
};


#endif
