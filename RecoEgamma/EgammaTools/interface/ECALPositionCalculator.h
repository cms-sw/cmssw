#ifndef ECALPositionCalculator_h
#define ECALPositionCalculator_h

#include "DataFormats/Math/interface/Vector3D.h"

class ECALPositionCalculator
{
   public:
      ECALPositionCalculator() { };
      double ecalPhi(math::XYZVector &momentum, math::XYZVector &vertex, int charge);
      double ecalEta(math::XYZVector &momentum, math::XYZVector &vertex);
   private:

};

#endif

