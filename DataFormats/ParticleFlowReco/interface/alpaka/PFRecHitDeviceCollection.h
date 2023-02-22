#ifndef ParticleFlowReco_PFRecHitDeviceCollection_h
#define ParticleFlowReco_PFRecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using PFRecHitDeviceCollection = PortableCollection<PFRecHitSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
