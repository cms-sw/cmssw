#ifndef DataFormats_ParticleFlowReco_interface_PFRecHitSoA_h
#define DataFormats_ParticleFlowReco_interface_PFRecHitSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

namespace reco {

  using PFRecHitsNeighbours = Eigen::Matrix<int32_t, 8, 1>;
  GENERATE_SOA_LAYOUT(PFRecHitSoALayout,
                      SOA_COLUMN(uint32_t, detId),
                      SOA_COLUMN(float, energy),
                      SOA_COLUMN(float, time),
                      SOA_COLUMN(int, depth),
                      SOA_COLUMN(PFLayer::Layer, layer),
                      SOA_EIGEN_COLUMN(PFRecHitsNeighbours,
                                       neighbours),  // Neighbour indices (or -1); order: N, S, E, W, NE, SW, SE, NW
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z),
                      SOA_SCALAR(uint32_t, size)  // Number of PFRecHits in SoA
  )

  using PFRecHitSoA = PFRecHitSoALayout<>;

}  // namespace reco

#endif  // DataFormats_ParticleFlowReco_interface_PFRecHitSoA_h
