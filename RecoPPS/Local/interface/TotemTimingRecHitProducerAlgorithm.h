/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra
 *   Christopher Misan (krzysztof.misan@cern.ch)
 *
 ****************************************************************************/

#ifndef RecoPPS_Local_TotemTimingRecHitProducerAlgorithm
#define RecoPPS_Local_TotemTimingRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "CondFormats/PPSObjects/interface/PPSTimingCalibration.h"

#include "RecoPPS/Local/interface/TotemTimingConversions.h"

#include <memory>

class TotemTimingRecHitProducerAlgorithm {
public:
  TotemTimingRecHitProducerAlgorithm(const edm::ParameterSet& conf);

  void setCalibration(const PPSTimingCalibration&);
  void build(const CTPPSGeometry&, const edm::DetSetVector<TotemTimingDigi>&, edm::DetSetVector<TotemTimingRecHit>&);

private:
  struct RegressionResults {
    float m, q, rms;
    RegressionResults() : m(0.), q(0.), rms(0.) {}
  };

  RegressionResults simplifiedLinearRegression(const std::vector<float>& time,
                                               const std::vector<float>& data,
                                               const unsigned int start_at,
                                               const unsigned int points) const;

  int fastDiscriminator(const std::vector<float>& data, float threshold) const;

  float constantFractionDiscriminator(const std::vector<float>& time, const std::vector<float>& data);

  static constexpr float SINC_COEFFICIENT = M_PI * 2 / 7.8;

  std::unique_ptr<TotemTimingConversions> sampicConversions_;

  bool mergeTimePeaks_;
  int baselinePoints_;
  double saturationLimit_;
  double cfdFraction_;
  int smoothingPoints_;
  double lowPassFrequency_;
  double hysteresis_;
  double sampicOffset_;
  double sampicSamplingPeriodNs_;
  TotemTimingRecHit::TimingAlgorithm mode_;
};

#endif
