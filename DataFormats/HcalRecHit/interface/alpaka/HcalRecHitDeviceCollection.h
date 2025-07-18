#ifndef DataFormats_HcalRecHit_alpaka_HcalRecHitDeviceCollection_h
#define DataFormats_HcalRecHit_alpaka_HcalRecHitDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hcal {

    // make the names from the top-level hcal namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hcal namespace
    using namespace ::hcal;

    // HcalRecHitSoA in device global memory
    using RecHitDeviceCollection = PortableCollection<HcalRecHitSoA>;
  }  // namespace hcal

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
