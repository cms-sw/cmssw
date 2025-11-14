#ifndef DataFormats_PortableTestObjects_interface_MultiHeadNetSoA_h
#define DataFormats_PortableTestObjects_interface_MultiHeadNetSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  using ClassificationHead = Eigen::Vector<float, 3>;
  GENERATE_SOA_LAYOUT(MultiHeadNetLayout,
                      SOA_COLUMN(float, regression_head),
                      SOA_EIGEN_COLUMN(ClassificationHead, classification_head))
  using MultiHeadNetSoA = MultiHeadNetLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_MultiHeadNetSoA_h
