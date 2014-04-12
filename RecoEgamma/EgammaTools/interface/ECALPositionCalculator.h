#ifndef ECALPositionCalculator_h
#define ECALPositionCalculator_h

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"

class MagneticField;

class ECALPositionCalculator
{
   public:
      ECALPositionCalculator() { };
      double ecalPhi(const MagneticField *magField, const math::XYZVector &momentum, const math::XYZPoint &vertex, const int charge);
      double ecalEta(const math::XYZVector &momentum, const math::XYZPoint &vertex);
   private:

};

#endif

