#ifndef DataFormats_PortableTestObjects_interface_alpaka_MaskDeviceCollection_h
#define DataFormats_PortableTestObjects_interface_alpaka_MaskDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "DataFormats/PortableTestObjects/interface/MaskHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MaskSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::portabletest {

  using namespace ::portabletest;

  using MaskDeviceCollection = PortableCollection<MaskSoA>;
  using ScalarMaskDeviceCollection = PortableCollection<ScalarMaskSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::portabletest

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::MaskDeviceCollection, portabletest::MaskHostCollection);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(portabletest::ScalarMaskDeviceCollection, portabletest::ScalarMaskHostCollection);

#endif  // DataFormats_PortableTestObjects_interface_alpaka_MaskDeviceCollection_h
