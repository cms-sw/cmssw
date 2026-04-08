#ifndef DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
#define DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/Common/interface/StdArray.h"

// These layouts from the basis of the schema evolution test.
// ROOT files have been written with SoAEvolutionZeroLayout and will be read with SoAEvolutionOneLayout to SoAEvolutionFiveLayout
namespace portabletest {

  using SEEigenObject = Eigen::Matrix<float, 4, 2>;
  using SEEigenObjectTwo = Eigen::Matrix<double, 4, 2>;
  enum class SEEnumType : uint16_t { s1 = 0, s2 = 1, s3 = 2 };
  enum class SEEnumTypeTwo : uint32_t { s1 = 0, s2 = 1, s3 = 2 };

  GENERATE_SOA_LAYOUT(SoAEvolutionZeroLayout,
                      SOA_COLUMN(float, cFloat),
                      SOA_COLUMN(int, cInt),
                      SOA_COLUMN(double, cDouble),
                      SOA_COLUMN(SEEnumType, cEnum),
                      SOA_EIGEN_COLUMN(SEEigenObject, eEigenObject),
                      SOA_SCALAR(int, sInt),
                      SOA_SCALAR(float, sFloat),
                      SOA_SCALAR(double, sDouble),
                      SOA_SCALAR(SEEnumType, sEnum));

  GENERATE_SOA_LAYOUT(SoAEvolutionOneLayout,
                      SOA_COLUMN(double, cFloat),
                      SOA_COLUMN(float, cInt),
                      SOA_COLUMN(int, cDouble),
                      SOA_COLUMN(SEEnumTypeTwo, cEnum),
                      SOA_EIGEN_COLUMN(SEEigenObjectTwo, eEigenObject),
                      SOA_SCALAR(double, sInt),
                      SOA_SCALAR(int8_t, sFloat),
                      SOA_SCALAR(float, sDouble),
                      SOA_SCALAR(SEEnumTypeTwo, sEnum));

  GENERATE_SOA_LAYOUT(SoAEvolutionTwoLayout,
                      SOA_COLUMN(float, cFloat),
                      SOA_COLUMN(int, newColumn),
                      SOA_COLUMN(SEEnumType, cEnum),
                      SOA_EIGEN_COLUMN(SEEigenObject, eEigenObject),
                      SOA_EIGEN_COLUMN(SEEigenObjectTwo, newEigenObject),
                      SOA_SCALAR(float, sFloatNewName),
                      SOA_SCALAR(int8_t, newScalar),
                      SOA_SCALAR(SEEnumType, sEnum));
					  

  using SoAEvolutionZero = SoAEvolutionZeroLayout<>;
  using SoAEvolutionOne = SoAEvolutionOneLayout<>;
  using SoAEvolutionTwo = SoAEvolutionTwoLayout<>;

}  // namespace portabletest

#endif  // DataFormats_PortableTestObjects_interface_SchemaEvolutionSoA_h
