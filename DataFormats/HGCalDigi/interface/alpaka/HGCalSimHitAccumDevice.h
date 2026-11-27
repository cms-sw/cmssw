#ifndef DataFormats_HGCalDigi_interface_alpaka_HGCalSimHitAccumDevice_h
#define DataFormats_HGCalDigi_interface_alpaka_HGCalSimHitAccumDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalSimHitAccumSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcaldigi {

    // make the names from the top-level hgcaldigi namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi namespace
    using namespace ::hgcaldigi;

    // SoA in device global memory for sim hit accumulators
    using HGCalSimHitAccumDevice = PortableCollection<HGCalSimHitAccumSoA>;

    // SoA with realistic Geant4 hits in HGCAL
    using HGCalRealisticSimHitDevice = PortableCollection<HGCalRealisticSimHitSoA>;

    // SoA in device global memory for ROC words accumulators
    using HGCalROCDigisAccumDevice = PortableCollection<HGCalROCDigisAccumSoA>;

  }  // namespace hgcaldigi

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalSimHitAccum_interface_alpaka_HGCalSimHitAccumDevice_h
