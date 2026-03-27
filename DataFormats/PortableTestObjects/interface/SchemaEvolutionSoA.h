#ifndef DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
#define DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Common/interface/StdArray.h"

namespace portabletest {

  using SEArray = edm::StdArray<int, 3>;
  using SEEigenVector = Eigen::Matrix<int64_t, 3, 3>;
  /*struct SiPixelErrorCompact {
  uint32_t rawId;
  uint32_t word;
  uint8_t errorType;
  double fedId;
};*/

  GENERATE_SOA_LAYOUT(SchemaEvolutionSoATemplate,
                      SOA_COLUMN(float, cOneFloat),
                      SOA_COLUMN(int, cTwoInt),
                      SOA_COLUMN(double, cThreeDouble),
                      // SOA_COLUMN(SiPixelErrorCompact, pixelErrors),
                      SOA_COLUMN(SEArray, cFourArray),
                      SOA_EIGEN_COLUMN(SEEigenVector, eOneVector3d),
                      SOA_SCALAR(int, sOneInt),
                      SOA_SCALAR(float, sTwoFloat),
                      SOA_SCALAR(double, sThreeDouble));

  using SchemaEvolutionSoA = SchemaEvolutionSoATemplate<>;
  using SchemaEvolutionSoAConstView = SchemaEvolutionSoA::ConstView;
  using SchemaEvolutionSoAView = SchemaEvolutionSoA::View;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
