#ifndef DataFormats_HGCalDigi_interface_alpaka_HGCalDigiDevice_h
#define DataFormats_HGCalDigi_interface_alpaka_HGCalDigiDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalDigi/interface/HGCalDigiSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcaldigi {

    // make the names from the top-level hgcaldigi namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi namespace
    using namespace ::hgcaldigi;

    // SoA in device global memory
    using HGCalDigiDevice = PortableCollection<HGCalDigiSoA>;

  }  // namespace hgcaldigi

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // DataFormats_HGCalDigi_interface_alpaka_HGCalDigiDevice_h
