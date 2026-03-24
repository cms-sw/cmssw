#ifndef DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
#define DataFormats_PortableTestObjects_interface_SimpleNetSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/Common/interface/StdArray.h"
#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Common/interface/FPX.h"

namespace portabletest {

  GENERATE_SOA_LAYOUT(SimpleNetLayout, SOA_COLUMN(float, reco_pt))
  GENERATE_SOA_LAYOUT(SimpleNetLayoutFPX, SOA_COLUMN(FPX, reco_pt))
  using SimpleNetSoAFPX = SimpleNetLayoutFPX<>;
  using SimpleNetSoA = SimpleNetLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SimpleNetSoA_h
