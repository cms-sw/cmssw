#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DataFormats/GeometryVector/interface/Phi.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DD4hep/Shapes.h"
#include <TGeoBBox.h>

using namespace cms;
using namespace cms::dd;
using namespace angle_units::operators;

template <class T>
bool convFpToBool(T val) {
  return (static_cast<int>(val + 0.5) != 0);
}

cms::DDSolidShape cms::dd::getCurrentShape(const DDFilteredView &fview) {
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
    phi1_ = convertDegToRad(coneSeg->GetPhi1());
    phi2_ = convertDegToRad(coneSeg->GetPhi2()) - phi1_;

    // Limit to range -pi to pi
    Geom::NormalizeWrapper<double, Geom::MinusPiToPi>::normalize(phi1_);
    rmin1_ = coneSeg->GetRmin1();
    rmin2_ = coneSeg->GetRmin2();
    rmax1_ = coneSeg->GetRmax1();
    rmax2_ = coneSeg->GetRmax2();
  }
}

// ** end of DDCons

// ** DDPseudoTrap
// No longer used -- this code does not work right
/*
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
*/
// ** end of DDPseudoTrap

// *** DDTrap

DDTrap::DDTrap(const DDFilteredView &fv) : valid{fv.isATrapezoid()} {
  if (valid) {
    const TGeoTrap *trap = fv.getShapePtr<TGeoTrap>();
    halfZ_ = trap->GetDz();
    theta_ = convertDegToRad(trap->GetTheta());
    phi_ = convertDegToRad(trap->GetPhi());
    x1_ = trap->GetBl1();  // Along x, low y, low z
    x2_ = trap->GetTl1();  // Along x, high y, low z
    y1_ = trap->GetH1();   // Along y, low z
    y2_ = trap->GetH2();   // Along y, high z
    x3_ = trap->GetBl2();  // Along x, low y, high z
    x4_ = trap->GetTl2();  // Along x, high y, high z
    alpha1_ = convertDegToRad(trap->GetAlpha1());
    alpha2_ = convertDegToRad(trap->GetAlpha2());
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
    startPhi_ = convertDegToRad(tube->GetPhi1());
    deltaPhi_ = convertDegToRad(tube->GetPhi2()) - startPhi_;

    // Limit to range -pi to pi
    Geom::NormalizeWrapper<double, Geom::MinusPiToPi>::normalize(startPhi_);
  }
}

// *** end of DDTubs

// ** DDTruncTubs

DDTruncTubs::DDTruncTubs(const DDFilteredView &fv) : valid{fv.isATruncTube()} {
  if (valid) {
    auto tube = fv.solid();
    std::vector<double> params = tube.dimensions();
    if (params.size() < 8) {
      edm::LogError("DDShapes DDTruncTubs") << "Truncated tube parameters list too small: " << params.size();
      return;
    }
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs zHalf = " << params[0];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs rIn = " << params[1];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs rOut = " << params[2];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs startPhi = " << params[3];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs deltaPhi = " << params[4];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs cutAtStart = " << params[5];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs cutAtDelta = " << params[6];
    LogTrace("DDShapes DDTruncTubs") << "DDTruncTubs cutInside = " << params[7];

    zHalf_ = params[0];  // This order determined by reading DD4hep source code
    rIn_ = params[1];
    rOut_ = params[2];
    startPhi_ = params[3];
    deltaPhi_ = params[4];
    cutAtStart_ = params[5];
    cutAtDelta_ = params[6];
    cutInside_ = (params[7] != 0);

    /* Previous versions of DD4hep output parameters that required more complex conversion
     * to produce the values CMS needs. Now the desired values are returned directly by the
     * "dimensions" function.  If the more complex conversion is ever needed again, the git history
     * of this file from before 2019-11-25 has code for converting from the internal DD4hep parameters
     * for a TruncatedTube to the eight parameters used by CMS.
     * There is also example code for checking the parameters of the TGeoCompositeShape.
    */
  }
}

// *** end of DDTruncTubs
