#ifndef PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h
#define PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/MaskSoA.h"

namespace torchportabletest {

  using MaskHost = PortableHostCollection<Mask>;
  using ScalarMaskHost = PortableHostCollection<ScalarMask>;

}  // namespace torchportabletest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_MaskHost_h