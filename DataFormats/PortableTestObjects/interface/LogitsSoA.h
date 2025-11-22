#ifndef DataFormats_PortableTestObjects_interface_LogitsSoA_h
#define DataFormats_PortableTestObjects_interface_LogitsSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  using LogitsType = Eigen::Vector<float, 10>;
  GENERATE_SOA_LAYOUT(LogitsLayout, SOA_EIGEN_COLUMN(LogitsType, logits))
  using LogitsSoA = LogitsLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_LogitsSoA_h
