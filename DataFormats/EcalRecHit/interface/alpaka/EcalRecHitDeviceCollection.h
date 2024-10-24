#ifndef DataFormats_EcalRecHit_alpaka_EcalRecHitDeviceCollection_h
#define DataFormats_EcalRecHit_alpaka_EcalRecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // EcalRecHitSoA in device global memory
  using EcalRecHitDeviceCollection = PortableCollection<EcalRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
