#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologySoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologySoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

// This data structure is an implementation detail of the RecoParticleFlow/PFRecHitProducer subpackage. Due to Alpaka build rules, it has to be located in the interface+src directories.
namespace reco {
  using PFRecHitsTopologyNeighbours = Eigen::Matrix<uint32_t, 8, 1>;
  GENERATE_SOA_LAYOUT(PFRecHitHCALTopologySoALayout,
                      SOA_COLUMN(float, positionX),
                      SOA_COLUMN(float, positionY),
                      SOA_COLUMN(float, positionZ),
                      SOA_EIGEN_COLUMN(PFRecHitsTopologyNeighbours, neighbours))
  GENERATE_SOA_LAYOUT(PFRecHitECALTopologySoALayout,
                      SOA_COLUMN(float, positionX),
                      SOA_COLUMN(float, positionY),
                      SOA_COLUMN(float, positionZ),
                      SOA_EIGEN_COLUMN(PFRecHitsTopologyNeighbours, neighbours))

  using PFRecHitHCALTopologySoA = PFRecHitHCALTopologySoALayout<>;
  using PFRecHitECALTopologySoA = PFRecHitECALTopologySoALayout<>;
}  // namespace reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitTopologySoA_h
