#ifndef DataFormats_SiStripDigiSoA_interface_alpaka_SiStripDigiDevice_h
#define DataFormats_SiStripDigiSoA_interface_alpaka_SiStripDigiDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiHost.h"
#include "DataFormats/SiStripDigiSoA/interface/SiStripDigiSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip {
  // make the names from the top-level sistrip namespace visible for unqualified lookup
  // inside the ALPAKA_ACCELERATOR_NAMESPACE::sistrip namespace
  using namespace ::sistrip;
  using SiStripDigiDevice = PortableCollection<SiStripDigiSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::sistrip

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(sistrip::SiStripDigiDevice, sistrip::SiStripDigiHost);

#endif
