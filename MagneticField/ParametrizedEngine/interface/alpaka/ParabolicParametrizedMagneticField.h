/**
 Description: Utility function to calculate the Magnetic Field on the GPU
*/

#ifndef DataFormats_EgammaReco_plugins_alpaka_MagneticFieldParabolicPortable_h
#define DataFormats_EgammaReco_plugins_alpaka_MagneticFieldParabolicPortable_h

#include <Eigen/Core>
#include "MagneticField/ParametrizedEngine/interface/ParabolicParametrizedMagneticField.h"

using Vector3f = Eigen::Matrix<float, 3, 1>;

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace MagneticFieldParabolicPortable {

    struct Parameters {
      float c1 = parabolicparametrizedmagneticfield::c1;
      float b0 = parabolicparametrizedmagneticfield::b0;
      float b1 = parabolicparametrizedmagneticfield::b1;
      float a = parabolicparametrizedmagneticfield::a;
    };

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float Kr(V3 vec) {
      Parameters p;
      return p.a * (vec(0) * vec(0) + vec(1) * vec(1)) + 1.;
    }

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float B0Z(V3 vec) {
      Parameters p;
      return p.b0 * vec(2) * vec(2) + p.b1 * vec(2) + p.c1;
    }

    template <typename V3>
    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float MagneticFieldAtPoint(V3 vec) {
      return B0Z(vec) * Kr(vec);
    }

  }  // namespace MagneticFieldParabolicPortable

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
