#ifndef HGCalCommissioning_HGCalDigiTrigger_interface_alpaka_HGCalDigiTriggerDevice_h
#define HGCalCommissioning_HGCalDigiTrigger_interface_alpaka_HGCalDigiTriggerDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HGCalCommissioning/HGCalDigiTrigger/interface/HGCalDigiTriggerSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcaldigi {

    // make the names from the top-level hgcaldigi namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi namespace
    using namespace ::hgcaldigi;

    // SoA in device global memory
    using HGCalDigiTriggerDevice = PortableCollection<HGCalDigiTriggerSoA>;

  }  // namespace hgcaldigi

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // HGCalCommissioning_HGCalDigiTrigger_interface_alpaka_HGCalDigiTriggerDevice_h
