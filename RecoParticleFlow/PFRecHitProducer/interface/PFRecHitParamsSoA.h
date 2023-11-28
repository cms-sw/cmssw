#ifndef RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsSoA_h
#define RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

// This data structure is an implementation detail of the
// RecoParticleFlow/PFRecHitProducer subpackage. Due to Alpaka build rules,
// it has to be located in the interface+src directories.
namespace reco {
  GENERATE_SOA_LAYOUT(PFRecHitHCALParamsSoALayout, SOA_COLUMN(float, energyThresholds))
  GENERATE_SOA_LAYOUT(PFRecHitECALParamsSoALayout,
                      SOA_COLUMN(float, energyThresholds),
                      SOA_SCALAR(float, cleaningThreshold))

  using PFRecHitHCALParamsSoA = PFRecHitHCALParamsSoALayout<>;
  using PFRecHitECALParamsSoA = PFRecHitECALParamsSoALayout<>;
}  // namespace reco

#endif  // RecoParticleFlow_PFRecHitProducer_interface_PFRecHitParamsSoA_h
