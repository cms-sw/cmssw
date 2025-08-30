#ifndef DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_
#define DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace torchportable {

  GENERATE_SOA_LAYOUT(ParticleLayout, SOA_COLUMN(float, pt), SOA_COLUMN(float, eta), SOA_COLUMN(float, phi))
  using ParticleSoA = ParticleLayout<>;

  GENERATE_SOA_LAYOUT(ClassificationLayout, SOA_COLUMN(float, c1), SOA_COLUMN(float, c2))
  using ClassificationSoA = ClassificationLayout<>;

  GENERATE_SOA_LAYOUT(RegressionLayout, SOA_COLUMN(float, reco_pt))
  using RegressionSoA = RegressionLayout<>;

}  // namespace torchportable

#endif  // DATA_FORMATS__PYTORCH_TEST__INTERFACE__LAYOUT_H_
