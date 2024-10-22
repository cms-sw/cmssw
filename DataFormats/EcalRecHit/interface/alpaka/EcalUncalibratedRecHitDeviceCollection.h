#ifndef DataFormats_EcalRecHit_alpaka_EcalUncalibratedRecHitDeviceCollection_h
#define DataFormats_EcalRecHit_alpaka_EcalUncalibratedRecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // EcalUncalibratedRecHitSoA in device global memory
  using EcalUncalibratedRecHitDeviceCollection = PortableCollection<EcalUncalibratedRecHitSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
