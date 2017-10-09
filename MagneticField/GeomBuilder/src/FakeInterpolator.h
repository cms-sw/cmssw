#ifndef FakeInterpolator_h
#define FakeInterpolator_h

/** \class magneticfield::FakeInterpolator
 *
 *  Fake field interpolator, that always return B=0.
 *
 *  \author N. Amapane - CERN
 */

  #include "MagneticField/Interpolation/interface/MagProviderInterpol.h"

namespace magneticfield {
class FakeInterpolator : public MagProviderInterpol {
 public:
  /// Constructor
  FakeInterpolator() {};
  
  // Operations
  virtual LocalVectorType valueInTesla( const LocalPointType& p) const {
    return LocalVectorType(0.,0.,0.);
  }
};
}
#endif
