#ifndef DataFormats_ParticleFlowReco_interface_alpaka_CaloRecHitDeviceCollection_h
#define DataFormats_ParticleFlowReco_interface_alpaka_CaloRecHitDeviceCollection_h

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::reco {

  using ::reco::CaloRecHitHostCollection;
  using CaloRecHitDeviceCollection = PortableCollection<::reco::CaloRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::reco

#endif  // DataFormats_ParticleFlowReco_interface_alpaka_CaloRecHitDeviceCollection_h
