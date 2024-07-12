#ifndef DataFormats_HcalDigi_interface_alpaka_HcalDigiDeviceCollection_h
#define DataFormats_HcalDigi_interface_alpaka_HcalDigiDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HcalDigi/interface/HcalDigiSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hcal {

    // make the names from the top-level hcal namespace visible for unqualified lookup
    // inside the ALPAKA_ACCELERATOR_NAMESPACE::hcal namespace
    using namespace ::hcal;

    // HcalDigiSoA in device global memory
    using Phase1DigiDeviceCollection = PortableCollection<HcalPhase1DigiSoA>;
    using Phase0DigiDeviceCollection = PortableCollection<HcalPhase0DigiSoA>;

  }  // namespace hcal
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
