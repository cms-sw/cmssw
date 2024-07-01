#include <cassert>
#include <Eigen/Core>
#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

int main() {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::MagneticFieldParabolicPortable;
  using Vector3f = Eigen::Matrix<float, 3, 1>;

  Vector3f position_in{1, 1, 1};
  assert(MagneticFieldAtPoint(position_in) == B0Z(position_in) * Kr(position_in));

  Vector3f position_out{1, 1, 300};
  assert(MagneticFieldAtPoint(position_out) == 0);

  return 0;
}