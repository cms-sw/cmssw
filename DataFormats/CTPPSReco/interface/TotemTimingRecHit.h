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
class TotemTimingRecHit : public CTPPSTimingRecHit {
public:
  enum TimingAlgorithm
  {NOT_SET, CFD, SMART, SIMPLE};

  TotemTimingRecHit()
      : CTPPSTimingRecHit(), amplitude_(0), baseline_rms_(0), mode_(NOT_SET)
      {}

  TotemTimingRecHit(float x, float x_width, float y, float y_width, float z,
                    float z_width, float t, float tot, float t_precision,
                    float amplitude, float baseline_rms, TimingAlgorithm mode)
      : CTPPSTimingRecHit(x, x_width, y, y_width, z, z_width, t, tot,
                          t_precision),
        amplitude_(amplitude), baseline_rms_(baseline_rms), mode_(mode)
      {}


  inline void setAmplitude(const float &amplitude) { amplitude_ = amplitude; }
  inline float getAmplitude() const { return amplitude_; }

  inline void setBaselineRMS(const float &baseline_rms) { baseline_rms_ = baseline_rms; }
  inline float getBaselineRMS() const { return baseline_rms_; }

  inline TimingAlgorithm getTimingAlgorithm() const { return mode_; }

private:
  float amplitude_;
  float baseline_rms_;
  TimingAlgorithm mode_;
};

#endif
