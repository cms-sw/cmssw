#include <cassert>
#include <Eigen/Core>
#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

int main() {
  using namespace ALPAKA_ACCELERATOR_NAMESPACE::MagneticFieldParabolicPortable;
  using Vector3f = Eigen::Matrix<float, 3, 1>;

  Vector3f position{1, 1, 1};
  assert(MagneticFieldAtPoint(position) == B0Z(position) * Kr(position));

  return 0;
}