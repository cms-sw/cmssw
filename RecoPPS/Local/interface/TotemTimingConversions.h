/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra (nicola.minafra@cern.ch)
 *   Filip Dej
 *
 ****************************************************************************/

#ifndef RecoPPS_Local_TotemTimingConversions
#define RecoPPS_Local_TotemTimingConversions

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"

#include <string>
#include <vector>

class TotemTimingConversions {
public:
  TotemTimingConversions(bool mergeTimePeaks, const PPSTimingCalibration& calibration);

  float timeOfFirstSample(const TotemTimingDigi& digi) const;
  float triggerTime(const TotemTimingDigi& digi) const;
  float timePrecision(const TotemTimingDigi& digi) const;
  std::vector<float> timeSamples(const TotemTimingDigi& digi) const;
  std::vector<float> voltSamples(const TotemTimingDigi& digi) const;

private:
  static constexpr float SAMPIC_SAMPLING_PERIOD_NS = 1. / 7.695;
  static constexpr float SAMPIC_ADC_V = 1. / 256;
  static constexpr int SAMPIC_MAX_NUMBER_OF_SAMPLES = 64;
  static constexpr int SAMPIC_DEFAULT_OFFSET = 30;
  static constexpr int ACCEPTED_TIME_RADIUS = 4;
  static constexpr unsigned long CELL0_MASK = 0xfffffff000;

  PPSTimingCalibration calibration_;
  bool mergeTimePeaks_;
  reco::FormulaEvaluator calibrationFunction_;
};

#endif
