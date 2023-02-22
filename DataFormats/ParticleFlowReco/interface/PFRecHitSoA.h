#ifndef ParticleFlowReco_PFRecHitSoA_h
#define ParticleFlowReco_PFRecHitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

using Vec8 = Eigen::Matrix<uint32_t, 8, 1>;
GENERATE_SOA_LAYOUT(PFRecHitSoALayout,
  SOA_COLUMN(uint32_t, detId),
  SOA_COLUMN(float, energy),
  SOA_COLUMN(float, time),
  SOA_COLUMN(int, depth),
  SOA_EIGEN_COLUMN(Vec8, neighbours)
)

using PFRecHitSoA = PFRecHitSoALayout<>;

#endif
