#ifndef ParticleFlowReco_CaloRecHitDeviceCollection_h
#define ParticleFlowReco_CaloRecHitDeviceCollection_h

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitSoA.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  using CaloRecHitDeviceCollection = PortableCollection<CaloRecHitSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif