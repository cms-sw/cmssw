/****************************************************************************
 *
 * This is a part of CTPPS offline software.
 * Authors:
 *   Laurent Forthomme (laurent.forthomme@cern.ch)
 *   Nicola Minafra
 *
 ****************************************************************************/

#ifndef RecoCTPPS_TotemRPLocal_TotemTimingRecHitProducerAlgorithm
#define RecoCTPPS_TotemRPLocal_TotemTimingRecHitProducerAlgorithm

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"

#include "DataFormats/CTPPSDetId/interface/TotemTimingDetId.h"
#include "DataFormats/CTPPSDigi/interface/TotemTimingDigi.h"
#include "DataFormats/CTPPSReco/interface/TotemTimingRecHit.h"
#include "RecoCTPPS/TotemRPLocal/interface/TotemTimingConversions.h"

#include "TGraph.h"
#include <algorithm>

#include "Geometry/VeryForwardGeometryBuilder/interface/CTPPSGeometry.h"
#include "Geometry/VeryForwardRPTopology/interface/RPTopology.h"

class TotemTimingRecHitProducerAlgorithm {
public:
  TotemTimingRecHitProducerAlgorithm(const edm::ParameterSet &conf);

  void build(const CTPPSGeometry *, const edm::DetSetVector<TotemTimingDigi> &,
             edm::DetSetVector<TotemTimingRecHit> &);

  struct RegressionResults {
    float m;
    float q;
    float rms;
    RegressionResults() : m(0), q(0), rms(0){};
  };

private:
  static const float SINC_COEFFICIENT;

  TotemTimingConversions sampicConversions_;
  int baselinePoints_;
  double saturationLimit_;
  double cfdFraction_;
  int smoothingPoints_;
  double lowPassFrequency_;
  double hysteresis_;
  TotemTimingRecHit::TimingAlgorithm mode_;

  RegressionResults simplifiedLinearRegression(const std::vector<float> &time,
                                               const std::vector<float> &data,
                                               const unsigned int start_at,
                                               const unsigned int points) const;

  int fastDiscriminator(const std::vector<float> &data,
                        const float &threshold) const;

  float constantFractionDiscriminator(const std::vector<float> &time,
                                      const std::vector<float> &data);
};

#endif
