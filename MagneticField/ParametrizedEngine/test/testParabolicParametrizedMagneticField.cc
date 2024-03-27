#include <cassert>
#include <Eigen/Core>
#include "MagneticField/ParametrizedEngine/interface/alpaka/ParabolicParametrizedMagneticField.h"

using Vector3f = Eigen::Matrix<float, 3, 1>;
using namespace MagneticFieldParabolicPortable;

Vector3f position{1, 1, 1};

assert(MagneticFieldAtPoint(position) == B0Z(position) * Kr(position));
