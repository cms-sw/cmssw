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

/// Reconstructed hit in Totem UFSDetectors.
/// \note t: time computed using algorithm timingAlgorithm()
class TotemTimingRecHit : public CTPPSTimingRecHit {
public:
  enum TimingAlgorithm { NOT_SET, CFD, SMART, SIMPLE };
  enum { NO_T_AVAILABLE = -100 };

  TotemTimingRecHit()
      : CTPPSTimingRecHit(), sampicThresholdTime_(0), tPrecision_(0), amplitude_(0), baselineRMS_(0), mode_(NOT_SET) {}

  TotemTimingRecHit(float x,
                    float xWidth,
                    float y,
                    float yWidth,
                    float z,
                    float zWidth,
                    float t,
                    float sampicThresholdTime,
                    float tPrecision,
                    float amplitude,
                    float baselineRMS,
                    TimingAlgorithm mode)
      : CTPPSTimingRecHit(x, xWidth, y, yWidth, z, zWidth, t),
        sampicThresholdTime_(sampicThresholdTime),
        tPrecision_(tPrecision),
        amplitude_(amplitude),
        baselineRMS_(baselineRMS),
        mode_(mode) {}

  inline void setSampicThresholdTime(float stt) { sampicThresholdTime_ = stt; }
  inline float sampicThresholdTime() const { return sampicThresholdTime_; }

  inline void setTPrecision(float tPrecision) { tPrecision_ = tPrecision; }
  inline float tPrecision() const { return tPrecision_; }

  inline void setAmplitude(float amplitude) { amplitude_ = amplitude; }
  inline float amplitude() const { return amplitude_; }

  inline void setBaselineRMS(float brms) { baselineRMS_ = brms; }
  inline float baselineRMS() const { return baselineRMS_; }

  inline TimingAlgorithm timingAlgorithm() const { return mode_; }

private:
  float sampicThresholdTime_, tPrecision_;
  float amplitude_;
  float baselineRMS_;
  TimingAlgorithm mode_;
};

#endif
