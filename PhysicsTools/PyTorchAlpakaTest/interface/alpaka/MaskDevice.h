#ifndef PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h
#define PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskHost.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchportabletest {

  using namespace ::torchportabletest;

  using MaskDevice = PortableCollection<Mask>;
  using ScalarMaskDevice = PortableCollection<ScalarMask>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchportabletest

ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::MaskDevice, torchportabletest::MaskHost);
ASSERT_DEVICE_MATCHES_HOST_COLLECTION(torchportabletest::ScalarMaskDevice, torchportabletest::ScalarMaskHost);

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_alpaka_MaskDevice_h