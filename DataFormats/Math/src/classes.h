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
    ROOT::Math::Polar3D<Double32_t> c33;
    ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<Double32_t> > v33;
    ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<Double32_t> > p3;

    math::Vector<1>::type v1;
    math::Vector<2>::type v2;
    math::Vector<3>::type v3;
    math::Vector<4>::type v4;
    math::Vector<5>::type v5;
    math::Vector<6>::type v6;

    math::Error<1>::type e1;
    math::Error<2>::type e2;
    math::Error<3>::type e3;
    math::Error<4>::type e4;
    math::Error<5>::type e5;
    math::Error<6>::type e6;
  }
}
