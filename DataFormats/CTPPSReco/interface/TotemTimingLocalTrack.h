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

namespace io_v1 {
  class TotemTimingLocalTrack : public CTPPSTimingLocalTrack {
  public:
    using CTPPSTimingLocalTrack::CTPPSTimingLocalTrack;
    // no specific class members yet
  };
}  // namespace io_v1
using TotemTimingLocalTrack = io_v1::TotemTimingLocalTrack;
#endif
