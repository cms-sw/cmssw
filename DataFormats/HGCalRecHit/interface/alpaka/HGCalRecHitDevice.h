#ifndef DataFormats_HGCalRecHit_interface_alpaka_HGCalRecHitDevice_h
#define DataFormats_HGCalRecHit_interface_alpaka_HGCalRecHitDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalRecHit/interface/HGCalRecHitSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcalrechit {

    // make the names from the top-level hgcalrechit namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hgcalrechit namespace
    using namespace ::hgcalrechit;

    // SoA in device global memory
    using HGCalRecHitDevice = PortableCollection<HGCalRecHitSoA>;

  }  // namespace hgcalrechit

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalRecHit_interface_alpaka_HGCalRecHitDevice_h
