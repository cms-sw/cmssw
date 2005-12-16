#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector.h"
#include "DataFormats/Math/interface/Error.h"

namespace {
  namespace {
    ROOT::Math::PtEtaPhiE4D<Double32_t> p41;
    ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> > lv1;
    ROOT::Math::PxPyPzE4D<Double32_t> p42;
    ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<Double32_t> > lv2;
    ROOT::Math::Cartesian3D<Double32_t> c31;
    ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<Double32_t> > v31;
    ROOT::Math::CylindricalEta3D<Double32_t> c32;
    ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<Double32_t> > v32;
    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > p3;

    math::Vector<1> v1;
    math::Vector<2> v2;
    math::Vector<3> v3;
    math::Vector<4> v4;
    math::Vector<5> v5;
    math::Vector<6> v6;

    math::Error<1> e1;
    math::Error<2> e2;
    math::Error<3> e3;
    math::Error<4> e4;
    math::Error<5> e5;
    math::Error<6> e6;
  }
}
