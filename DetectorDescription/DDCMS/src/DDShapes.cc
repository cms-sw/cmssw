#include "DetectorDescription/DDCMS/interface/DDShapes.h"

#include "DD4hep/Shapes.h"
#include <TGeoBBox.h>

using namespace cms;
using namespace cms::dd;

template <class T>
bool convFpToBool(T val) {
  return (static_cast<int>(val + 0.5) != 0);
}

DDSolidShape cms::dd::getCurrentShape(const DDFilteredView &fview) {
  if (fview.isABox())
    return (DDSolidShape::ddbox);

  if (fview.isAConeSeg())
    return (DDSolidShape::ddcons);

  if (fview.isATrapezoid())
    return (DDSolidShape::ddtrap);

  if (fview.isATubeSeg())
    return (DDSolidShape::ddtubs);

  if (fview.isATruncTube())
    return (DDSolidShape::ddtrunctubs);

  if (fview.isAPseudoTrap())  // Rarely used -- put it last
    return (DDSolidShape::ddpseudotrap);

  return (DDSolidShape::dd_not_init);
}

// ** DDBox
DDBox::DDBox(const DDFilteredView &fv) : valid{fv.isABox()} {
  if (valid) {
    const TGeoBBox *box = fv.getShapePtr<TGeoBBox>();
    dx_ = box->GetDX();
    dy_ = box->GetDY();
    dz_ = box->GetDZ();
  }
}

// ** end DDBox

// ** DDCons

DDCons::DDCons(const DDFilteredView &fv) : valid{fv.isAConeSeg()} {
  if (valid) {
    const TGeoConeSeg *coneSeg = fv.getShapePtr<TGeoConeSeg>();
    dz_ = coneSeg->GetDZ();
    phi1_ = coneSeg->GetPhi1();
    phi2_ = coneSeg->GetPhi2();
    rmin1_ = coneSeg->GetRmin1();
    rmin2_ = coneSeg->GetRmin2();
    rmax1_ = coneSeg->GetRmax1();
    rmax2_ = coneSeg->GetRmax2();
  }
}

// ** end of DDCons

// ** DDPseudoTrap

DDPseudoTrap::DDPseudoTrap(const DDFilteredView &fv) : valid{fv.isAPseudoTrap()} {
  if (valid) {
    auto trap = fv.solid();
    std::vector<double> params = trap.dimensions();
    minusX_ = params[0];
    plusX_ = params[1];
    minusY_ = params[2];
    plusY_ = params[3];
    dz_ = params[4];
    rmax_ = params[5];
    minusZSide_ = convFpToBool(params[6]);
  }
}

// ** end of DDPseudoTrap

// *** DDTrap

DDTrap::DDTrap(const DDFilteredView &fv) : valid{fv.isATrapezoid()} {
  if (valid) {
    const TGeoTrap *trap = fv.getShapePtr<TGeoTrap>();
    halfZ_ = trap->GetDz();
    theta_ = trap->GetTheta();
    phi_ = trap->GetPhi();
    x1_ = trap->GetBl1();  // Along x, low y, low z
    x2_ = trap->GetTl1();  // Along x, high y, low z
    y1_ = trap->GetH1();   // Along y, low z
    y2_ = trap->GetH2();   // Along y, high z
    x3_ = trap->GetBl2();  // Along x, low y, high z
    x4_ = trap->GetTl2();  // Along x, high y, high z
    alpha1_ = trap->GetAlpha1();
    alpha2_ = trap->GetAlpha2();
  }
}

// *** end of DDTrap

// ** DDTubs

DDTubs::DDTubs(const DDFilteredView &fv) : valid{fv.isATubeSeg()} {
  if (valid) {
    const TGeoTubeSeg *tube = fv.getShapePtr<TGeoTubeSeg>();
    zHalf_ = tube->GetDz();
    rIn_ = tube->GetRmin();
    rOut_ = tube->GetRmax();
    startPhi_ = tube->GetPhi1();
    deltaPhi_ = tube->GetPhi2();
  }
}

// *** end of DDTubs

// ** DDTruncTubs

DDTruncTubs::DDTruncTubs(const DDFilteredView &fv) : valid{fv.isATruncTube()} {
  if (valid) {
    auto tube = fv.solid();
    std::vector<double> params = tube.dimensions();
    zHalf_ = params[0];
    rIn_ = params[1];
    rOut_ = params[2];
    startPhi_ = params[3];
    deltaPhi_ = params[4];
    cutAtStart_ = params[5];
    cutAtDelta_ = params[6];
    cutInside_ = convFpToBool(params[7]);
  }
}

// *** end of DDTruncTubs
