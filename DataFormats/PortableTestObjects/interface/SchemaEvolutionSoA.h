#ifndef DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
#define DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Common/interface/StdArray.h"

namespace portabletest {

  using SEArray = edm::StdArray<int, 3>;
  using SEEigenVector = Eigen::Matrix<float, 4, 1>;
  enum class PixelType : int64_t { s1 = 0, s2 = 1, s3 = 2 };
  template<typename T>
  struct SiPixelErrorCompact {
  T rawId;
  T word;
  uint8_t errorType;
  double fedId;
};

using SiPixelErrorCompact64 = SiPixelErrorCompact<uint32_t>;

  GENERATE_SOA_LAYOUT(SchemaEvolutionSoATemplate,
                      SOA_COLUMN(float, cOneFloat),
                      SOA_COLUMN(int, cTwoInt),
                      SOA_COLUMN(double, cThreeDouble),
                      SOA_COLUMN(SEArray, cFourArray),
                      SOA_COLUMN(PixelType, cpixelType),
                      SOA_COLUMN(SiPixelErrorCompact64, customStruct),
                      SOA_EIGEN_COLUMN(SEEigenVector, eOneVector3d),
                      SOA_SCALAR(int, sOneInt),
                      SOA_SCALAR(float, sTwoFloat),
                      SOA_SCALAR(double, sThreeDouble),
                      SOA_SCALAR(PixelType, sPixelType));

  using SchemaEvolutionSoA = SchemaEvolutionSoATemplate<>;
  using SchemaEvolutionSoAConstView = SchemaEvolutionSoA::ConstView;
  using SchemaEvolutionSoAView = SchemaEvolutionSoA::View;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
