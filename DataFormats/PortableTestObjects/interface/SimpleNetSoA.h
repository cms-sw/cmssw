#ifndef DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
#define DataFormats_PortableTestObjects_interface_SimpleNetSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(SimpleNetLayout, SOA_COLUMN(float, reco_pt))
  using SimpleNetSoA = SimpleNetLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
