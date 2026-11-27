#ifndef DataFormats_HGCalDigi_interface_alpaka_HGCalECONTPacketInfoDevice_h
#define DataFormats_HGCalDigi_interface_alpaka_HGCalECONTPacketInfoDevice_h

#include "DataFormats/HGCalDigi/interface/HGCalECONTPacketInfoSoA.h"
#include "DataFormats/Portable/interface/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcaldigi {

    // make the names from the top-level hgcaldigi namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hgcaldigi namespace
    using namespace ::hgcaldigi;

    // SoA in device global memory
    using HGCalECONTPacketInfoDevice = PortableCollection<HGCalECONTPacketInfoSoA>;

  }  // namespace hgcaldigi

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
