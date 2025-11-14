#ifndef DataFormats_PortableTestObjects_interface_ImageSoA_h
#define DataFormats_PortableTestObjects_interface_ImageSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  using ColorChannel = Eigen::Matrix<float, 9, 9>;
  GENERATE_SOA_LAYOUT(ImageLayout,
                      SOA_EIGEN_COLUMN(ColorChannel, r),
                      SOA_EIGEN_COLUMN(ColorChannel, g),
                      SOA_EIGEN_COLUMN(ColorChannel, b))
  using ImageSoA = ImageLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_ImageSoA_h
