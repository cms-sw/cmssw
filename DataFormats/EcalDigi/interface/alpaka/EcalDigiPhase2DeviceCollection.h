#ifndef DataFormats_EcalDigi_interface_alpaka_EcalDigiPhase2DeviceCollection_h
#define DataFormats_EcalDigi_interface_alpaka_EcalDigiPhase2DeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2SoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  // EcalDigiPhase2SoA in device global memory
  using EcalDigiPhase2DeviceCollection = PortableCollection<EcalDigiPhase2SoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
