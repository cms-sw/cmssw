#include "DetectorDescription/DDCMS/interface/DDAlgoArguments.h"
#include "DetectorDescription/DDCMS/interface/DDShapes.h"
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
    if (params.size() < 16) {
      edm::LogError("DDShapes DDTruncTubs") << "Truncated tube parameters list too small";
      return;
    }
    for (unsigned int index = 0; index < params.size(); ++index) {
      edm::LogVerbatim("DDShapes DDTruncTubs") << "DDTruncTubs param " << index << " = " << params[index];
    }
    rIn_ = params[1];
    rOut_ = params[2];
    zHalf_ = params[3];
    startPhi_ = convertDegToRad(params[4]);
    deltaPhi_ = convertDegToRad(params[5]) - startPhi_;
    // Limit to range -pi to pi
    Geom::NormalizeWrapper<double, Geom::MinusPiToPi>::normalize(startPhi_);

    dd4hep::Rotation3D cutRotation(
        params[6], params[7], params[8], params[9], params[10], params[11], params[12], params[13], params[14]);
    double translation = params[15];
    DD3Vector xUnitVec(1., 0., 0.);
    DD3Vector rotatedVec = cutRotation(xUnitVec);
    double cosAlpha = xUnitVec.Dot(rotatedVec);
    double sinAlpha = sqrt(1. - cosAlpha * cosAlpha);
    if (sinAlpha == 0.)
      sinAlpha = 1.;  // Prevent divide by 0
    cutInside_ = true;
    cutAtStart_ = translation + (rOut_ / sinAlpha);
    if (cutAtStart_ > rOut_ || cutAtStart_ < 0.) {
      cutAtStart_ = translation - (rOut_ / sinAlpha);
      cutInside_ = false;
    }
    double alpha = std::acos(cosAlpha);
    if (std::abs(deltaPhi_) != 1._pi)
      cutAtDelta_ = cutAtStart_ * (sinAlpha / std::sin(deltaPhi_ + alpha));

    /*
     * If we need to check the parameters in the TGeoCompositeShape
    const TGeoCompositeShape *compShape = fv.getShapePtr<TGeoCompositeShape>();
    const TGeoBoolNode *boolNode = compShape->GetBoolNode();
    const TGeoMatrix *lmatrix = boolNode->GetLeftMatrix();
    auto showMats = [] (const TGeoMatrix *matrix) -> void {
      const Double_t *rotMatrix = matrix->GetRotationMatrix();
      const Double_t *translat = matrix->GetTranslation();
      edm::LogVerbatim("DDShapes DDTruncTubs") << "translation (" << translat[0] << ", "
        << translat[1] << ", " << translat[2] << ")\n";
      edm::LogVerbatim("DDShapes DDTruncTubs") << "rotation 1 (" << rotMatrix[0] << ", "
        << rotMatrix[1] << ", " << rotMatrix[2] << ")\n";
      edm::LogVerbatim("DDShapes DDTruncTubs") << "rotation 2 (" << rotMatrix[3] << ", "
        << rotMatrix[4] << ", " << rotMatrix[2] << ")\n";
      edm::LogVerbatim("DDShapes DDTruncTubs") << "rotation 3 (" << rotMatrix[6] << ", "
        << rotMatrix[7] << ", " << rotMatrix[8] << ")\n";
    };
    edm::LogVerbatim("DDShapes DDTruncTubs") << "Left matrix";
    showMats(lmatrix);
    const TGeoMatrix *rmatrix = boolNode->GetRightMatrix();
    edm::LogVerbatim("DDShapes DDTruncTubs") << "Right matrix";
    showMats(rmatrix);
    */
  }
}

// *** end of DDTruncTubs
