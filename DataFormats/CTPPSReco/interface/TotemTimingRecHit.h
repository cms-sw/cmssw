/****************************************************************************
*
* This is a part of CTPPS offline software.
* Authors:
*   Laurent Forthomme (laurent.forthomme@cern.ch)
*   Nicola Minafra (nicola.minafra@cern.ch)
*
****************************************************************************/

#ifndef DataFormats_CTPPSReco_TotemTimingRecHit
#define DataFormats_CTPPSReco_TotemTimingRecHit

#include "DataFormats/CTPPSReco/interface/CTPPSTimingRecHit.h"

/// Reconstructed hit in totem ufsd detectors.
class TotemTimingRecHit : public CTPPSTimingRecHit
{
  public:
    TotemTimingRecHit() :
      CTPPSTimingRecHit(),
      amplitude_(0)
    {}
    TotemTimingRecHit( float x, float x_width, float y, float y_width, float z, float z_width, float t, float tot, float t_precision, float amplitude ) :
      CTPPSTimingRecHit( x, x_width, y, y_width, z, z_width, t, tot, t_precision ),
      amplitude_(amplitude)
    {}

    inline void setAmplitude( const float& amplitude ) { amplitude_ = amplitude; }
    inline float getAmplitude() const { return amplitude_; }

  private:
    float amplitude_;

};

#endif
