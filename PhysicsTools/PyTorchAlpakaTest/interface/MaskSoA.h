#ifndef PhysicsTools_PyTorchAlpakaTest_interface_MaskSoA_h
#define PhysicsTools_PyTorchAlpakaTest_interface_MaskSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace torchportabletest {

  using PartMask = Eigen::Vector<uint8_t, 3>;
  GENERATE_SOA_LAYOUT(MaskLayout, SOA_EIGEN_COLUMN(PartMask, mask));
  GENERATE_SOA_LAYOUT(ScalarMaskLayout, SOA_SCALAR(uint8_t, scalar_mask))

  using Mask = MaskLayout<>;
  using ScalarMask = ScalarMaskLayout<>;

}  // namespace torchportabletest

#endif  // PhysicsTools_PyTorchAlpakaTest_interface_MaskSoA_h