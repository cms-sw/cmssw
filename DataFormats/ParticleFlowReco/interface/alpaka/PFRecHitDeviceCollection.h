#ifndef DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h
#define DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h

#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::PFRecHitHostCollection;
  using PFRecHitDeviceCollection = PortableCollection<::reco::PFRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // DataFormats_ParticleFlowReco_interface_alpaka_PFRecHitDeviceCollection_h
