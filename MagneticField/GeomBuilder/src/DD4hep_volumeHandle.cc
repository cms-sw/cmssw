/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "DD4hep_volumeHandle.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include <iterator>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;
using namespace edm;
using namespace angle_units::operators;

using DDBox = dd4hep::Box;
using DDTrap = dd4hep::Trap;
using DDTubs = dd4hep::Tube;
using DDCons = dd4hep::ConeSegment;
using DDTruncTubs = dd4hep::TruncatedTube;

volumeHandle::volumeHandle(const cms::DDFilteredView &fv, bool expand2Pi, bool debugVal)
    : BaseVolumeHandle(expand2Pi, debugVal), theShape(fv.legacyShape(fv.shape())), solid(fv) {
  name = fv.name();
  copyno = fv.copyNum();
  const auto *const transArray = fv.trans();
  center_ = GlobalPoint(transArray[0], transArray[1], transArray[2]);

  // ASSUMPTION: volume names ends with "_NUM" where NUM is the volume number
  string volName = name;
  volName.erase(0, volName.rfind('_') + 1);
  volumeno = static_cast<unsigned short>(std::atoi(volName.c_str()));

  for (int i = 0; i < 6; ++i) {
    isAssigned[i] = false;
  }
  referencePlane(fv);
  switch (theShape) {
    case DDSolidShape::ddbox:
      buildBox();
      break;
    case DDSolidShape::ddtrap:
      buildTrap();
      break;
    case DDSolidShape::ddcons:
      buildCons();
      break;
    case DDSolidShape::ddtubs:
      buildTubs();
      break;
    case DDSolidShape::ddpseudotrap: {
      vector<double> d = solid.volume().volume().solid().dimensions();
      buildPseudoTrap(d[0], d[1], d[2], d[3], d[4], d[5], d[6]);
    } break;
    case DDSolidShape::ddtrunctubs:
      buildTruncTubs();
      break;
    default:
      LogError("magneticfield::volumeHandle")
          << "ctor: Unexpected shape # " << static_cast<int>(theShape) << " for vol " << name;
  }

  // Get material for this volume
  if (fv.materialName() == "Iron")
    isIronFlag = true;

  if (debug) {
    LogTrace("MagGeoBuilder") << " RMin =  " << theRMin << newln << " RMax =  " << theRMax;

    if (theRMin < 0 || theRN < theRMin || theRMax < theRN)
      LogTrace("MagGeoBuilder") << "*** WARNING: wrong RMin/RN/RMax";

    LogTrace("MagGeoBuilder") << "Summary: " << name << " " << copyno << " shape = " << theShape << " trasl "
                              << center() << " R " << center().perp() << " phi " << center().phi() << " magFile "
                              << magFile << " Material= " << fv.materialName() << " isIron= " << isIronFlag
                              << " masterSector= " << masterSector;

    LogTrace("MagGeoBuilder") << " Orientation of surfaces:";
    std::string sideName[3] = {"positiveSide", "negativeSide", "onSurface"};
    for (int i = 0; i < 6; ++i) {
      if (surfaces[i] != nullptr)
        LogTrace("MagGeoBuilder") << "  " << i << ":" << sideName[surfaces[i]->side(center_, 0.3)];
    }
  }
}

void volumeHandle::referencePlane(const cms::DDFilteredView &fv) {
  // The refPlane is the "main plane" for the solid. It corresponds to the
  // x,y plane in the DDD local frame, and defines a frame where the local
  // coordinates are the same as in DDD.
  // In the geometry version 85l_030919, this plane is normal to the
  // beam line for all volumes but pseudotraps, so that global R is along Y,
  // global phi is along -X and global Z along Z:
  //
  //   Global(for vol at pi/2)    Local
  //   +R (+Y)                    +Y
  //   +phi(-X)                   -X
  //   +Z                         +Z
  //
  // For pseudotraps the refPlane is parallel to beam line and global R is
  // along Z, global phi is along +-X and and global Z along Y:
  //
  //   Global(for vol at pi/2)    Local
  //   +R (+Y)                    +Z
  //   +phi(-X)                   +X
  //   +Z                         +Y
  //
  // Note that the frame is centered in the DDD volume center, which is
  // inside the volume for DDD boxes and (pesudo)trapezoids, on the beam line
  // for tubs, cons and trunctubs.

  // In geometry version 1103l, trapezoids have X and Z in the opposite direction
  // than the above.  Boxes are either oriented as described above or in some case
  // have opposite direction for Y and X.

  // The global position
  Surface::PositionType &posResult = center_;

  // The reference plane rotation
  math::XYZVector x, y, z;
  dd4hep::Rotation3D refRot;
  fv.rot(refRot);
  refRot.GetComponents(x, y, z);
  if (debug) {
    if (x.Cross(y).Dot(z) < 0.5) {
      LogTrace("MagGeoBuilder") << "*** WARNING: Rotation is not RH ";
    }
  }

  // The global rotation
  Surface::RotationType rotResult(float(x.X()),
                                  float(x.Y()),
                                  float(x.Z()),
                                  float(y.X()),
                                  float(y.Y()),
                                  float(y.Z()),
                                  float(z.X()),
                                  float(z.Y()),
                                  float(z.Z()));

  refPlane = new GloballyPositioned<float>(posResult, rotResult);

  // Check correct orientation
  if (debug) {
    LogTrace("MagGeoBuilder") << "Refplane pos  " << refPlane->position();

    // See comments above for the conventions for orientation.
    LocalVector globalZdir(0., 0., 1.);  // Local direction of the axis along global Z

    if (theShape == DDSolidShape::ddpseudotrap) {
      globalZdir = LocalVector(0., 1., 0.);
    }

    if (refPlane->toGlobal(globalZdir).z() < 0.) {
      globalZdir = -globalZdir;
    }
    float chk = refPlane->toGlobal(globalZdir).dot(GlobalVector(0, 0, 1));
    if (chk < .999)
      LogTrace("MagGeoBuilder") << "*** WARNING RefPlane check failed!***" << chk;
  }
}

std::vector<VolumeSide> volumeHandle::sides() const {
  std::vector<VolumeSide> result;
  for (int i = 0; i < 6; ++i) {
    // If this is just a master volume out of wich a 2pi volume
    // should be built (e.g. central cylinder), skip the phi boundaries.
    if (expand && (i == phiplus || i == phiminus))
      continue;

    // FIXME: Skip null inner degenerate cylindrical surface
    if (theShape == DDSolidShape::ddtubs && i == SurfaceOrientation::inner && theRMin < 0.001)
      continue;

    ReferenceCountingPointer<Surface> s = const_cast<Surface *>(surfaces[i].get());
    result.push_back(VolumeSide(s, GlobalFace(i), surfaces[i]->side(center_, 0.3)));
  }
  return result;
}

// The files included below are used here and in the old DD version of this file.
// To allow them to be used in both places, they call a "convertUnits" function
// that is defined differently between old and new DD.
// For the old DD, another version of this function converts mm to cm.

namespace {
  template <class NumType>
  inline constexpr NumType convertUnits(NumType centimeters) {
    return (centimeters);
  }
}  // namespace

using namespace cms::dd;

/*
 *  Compute parameters for a box 
 *
 *  \author N. Amapane - INFN Torino
 */

void volumeHandle::buildBox() {
  LogTrace("MagGeoBuilder") << "Building box surfaces...: ";

  DDBox box(solid.solid());
  // Old DD needs mm to cm conversion, but DD4hep needs no conversion.
  // convertUnits should be defined appropriately.
  double halfX = convertUnits(box.x());
  double halfY = convertUnits(box.y());
  double halfZ = convertUnits(box.z());

  // Global vectors of the normals to X, Y, Z axes
  GlobalVector planeXAxis = refPlane->toGlobal(LocalVector(1, 0, 0));
  GlobalVector planeYAxis = refPlane->toGlobal(LocalVector(0, 1, 0));
  GlobalVector planeZAxis = refPlane->toGlobal(LocalVector(0, 0, 1));

  // FIXME Assumption: it is assumed that in the following that
  // local Z is always along global Z
  // (true for version 1103l, not necessarily in the future)

  // To determine the orientation of other local axes,
  // find local axis closest to global R
  GlobalVector Rvol(refPlane->position().x(), refPlane->position().y(), refPlane->position().z());
  double rnX = planeXAxis.dot(Rvol);
  double rnY = planeYAxis.dot(Rvol);

  GlobalPoint pos_outer;
  GlobalPoint pos_inner;
  GlobalPoint pos_phiplus;
  GlobalPoint pos_phiminus;
  GlobalPoint pos_zplus(refPlane->toGlobal(LocalPoint(0., 0., halfZ)));
  GlobalPoint pos_zminus(refPlane->toGlobal(LocalPoint(0., 0., -halfZ)));

  Surface::RotationType rot_R;
  Surface::RotationType rot_phi;
  Surface::RotationType rot_Z = Surface::RotationType(planeXAxis, planeYAxis);

  if (std::abs(rnX) > std::abs(rnY)) {
    // X is ~parallel to global R dir, Y is along +/- phi
    theRN = std::abs(rnX);
    if (rnX < 0) {
      halfX = -halfX;
      halfY = -halfY;
    }
    pos_outer = GlobalPoint(refPlane->toGlobal(LocalPoint(halfX, 0., 0.)));
    pos_inner = GlobalPoint(refPlane->toGlobal(LocalPoint(-halfX, 0., 0.)));
    pos_phiplus = GlobalPoint(refPlane->toGlobal(LocalPoint(0., halfY, 0.)));
    pos_phiminus = GlobalPoint(refPlane->toGlobal(LocalPoint(0., -halfY, 0.)));

    rot_R = Surface::RotationType(planeZAxis, planeYAxis);
    rot_phi = Surface::RotationType(planeZAxis, planeXAxis);  // opposite to y axis
  } else {
    // Y is ~parallel to global R dir, X is along +/- phi
    theRN = std::abs(rnY);
    if (rnY < 0) {
      halfX = -halfX;
      halfY = -halfY;
    }
    pos_outer = GlobalPoint(refPlane->toGlobal(LocalPoint(0., halfY, 0.)));
    pos_inner = GlobalPoint(refPlane->toGlobal(LocalPoint(0., -halfY, 0.)));
    pos_phiplus = GlobalPoint(refPlane->toGlobal(LocalPoint(-halfX, 0., 0.)));
    pos_phiminus = GlobalPoint(refPlane->toGlobal(LocalPoint(halfX, 0., 0.)));

    rot_R = Surface::RotationType(planeZAxis, planeXAxis);
    rot_phi = Surface::RotationType(planeZAxis, planeYAxis);  // opposite to x axis
  }

  LogTrace("MagGeoBuilder") << " halfX: " << halfX << "  halfY: " << halfY << "  halfZ: " << halfZ << "  RN: " << theRN;

  LogTrace("MagGeoBuilder") << "pos_outer    " << pos_outer << " " << pos_outer.perp() << " " << pos_outer.phi()
                            << newln << "pos_inner    " << pos_inner << " " << pos_inner.perp() << " "
                            << pos_inner.phi() << newln << "pos_zplus    " << pos_zplus << " " << pos_zplus.perp()
                            << " " << pos_zplus.phi() << newln << "pos_zminus   " << pos_zminus << " "
                            << pos_zminus.perp() << " " << pos_zminus.phi() << newln << "pos_phiplus  " << pos_phiplus
                            << " " << pos_phiplus.perp() << " " << pos_phiplus.phi() << newln << "pos_phiminus "
                            << pos_phiminus << " " << pos_phiminus.perp() << " " << pos_phiminus.phi();

  // Check ordering.
  if (debug) {
    if (pos_outer.perp() < pos_inner.perp()) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_outer < pos_inner for box";
    }
    if (pos_zplus.z() < pos_zminus.z()) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_zplus < pos_zminus for box";
    }
    if (Geom::Phi<float>(pos_phiplus.phi() - pos_phiminus.phi()) < 0.) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_phiplus < pos_phiminus for box";
    }
  }

  // FIXME: use builder
  surfaces[outer] = new Plane(pos_outer, rot_R);
  surfaces[inner] = new Plane(pos_inner, rot_R);
  surfaces[zplus] = new Plane(pos_zplus, rot_Z);
  surfaces[zminus] = new Plane(pos_zminus, rot_Z);
  surfaces[phiplus] = new Plane(pos_phiplus, rot_phi);
  surfaces[phiminus] = new Plane(pos_phiminus, rot_phi);

  LogTrace("MagGeoBuilder") << "rot_R   " << surfaces[outer]->toGlobal(LocalVector(0., 0., 1.)) << newln << "rot_Z   "
                            << surfaces[zplus]->toGlobal(LocalVector(0., 0., 1.)) << newln << "rot_phi "
                            << surfaces[phiplus]->toGlobal(LocalVector(0., 0., 1.));

  // Save volume boundaries
  theRMin = std::abs(surfaces[inner]->toLocal(GlobalPoint(0, 0, 0)).z());
  theRMax = std::abs(surfaces[outer]->toLocal(GlobalPoint(0, 0, 0)).z());
  // FIXME: use phi of middle plane of phiminus surface. Is not the absolute phimin!
  thePhiMin = surfaces[phiminus]->position().phi();
}

/*
 *  Compute parameters for a trapezoid 
 *
 *  \author N. Amapane - INFN Torino
 */

void volumeHandle::buildTrap() {
  LogTrace("MagGeoBuilder") << "Building trapezoid surfaces...: ";

  DDTrap trap(solid.solid());  //FIXME

  // Old DD needs mm to cm conversion, but DD4hep needs no conversion.
  // convertUnits should be defined appropriately.
  double x1 = convertUnits(trap.bottomLow1());
  double x2 = convertUnits(trap.topLow1());
  double x3 = convertUnits(trap.bottomLow2());
  double x4 = convertUnits(trap.topLow2());
  double y1 = convertUnits(trap.high1());
  double y2 = convertUnits(trap.high2());
  double theta = convertDegToRad(trap.theta());
  double phi = convertDegToRad(trap.phi());
  double halfZ = convertUnits(trap->GetDz());  // FIXME: missing accessor
  double alpha1 = convertDegToRad(trap.alpha1());
  double alpha2 = convertDegToRad(trap.alpha2());

  LogTrace("MagGeoBuilder") << "x1 " << x1 << newln << "x2 " << x2 << newln << "x3 " << x3 << newln << "x4 " << x4
                            << newln << "y1 " << y1 << newln << "y2 " << y2 << newln << "theta  " << theta << newln
                            << "phi    " << phi << newln << "halfZ  " << halfZ << newln << "alpha1 " << alpha1 << newln
                            << "alpha2 " << alpha2;

  // Just check assumptions on parameters...
  if (debug) {
    const double epsilon = 1e-5;
    if (theta > epsilon || phi > epsilon || y1 - y2 > epsilon || x1 - x3 > epsilon || x2 - x4 > epsilon ||
        alpha1 - alpha2 > epsilon) {
      LogTrace("MagGeoBuilder") << "*** WARNING: unexpected trapezoid parameters.";
    }
  }

  //  Used parameters halfZ, x1, x2, y1, alpha1
  GlobalVector planeXAxis = refPlane->toGlobal(LocalVector(1, 0, 0));
  GlobalVector planeYAxis = refPlane->toGlobal(LocalVector(0, 1, 0));
  GlobalVector planeZAxis = refPlane->toGlobal(LocalVector(0, 0, 1));

  // For the correct definition of inner, outer, phiplus, phiminus,
  // Zplus, zminus the orientation of axes must be checked.
  // Normal (convention in version 85l_030919):
  //  planeXAxis in the direction of decreasing global phi;
  //  planeZAxis in the direction of global Z;
  // => Invert the sign of local X and Z if planeZAxis points to - global Z

  // FIXME Assumption: it is assumed that local Y is always along global R
  // (true for version 1103l, not necessarily in the future)
  GlobalVector Rvol(refPlane->position().x(), refPlane->position().y(), refPlane->position().z());
  theRN = std::abs(planeYAxis.dot(Rvol));

  if (planeZAxis.z() < 0.) {
    x1 = -x1;
    x2 = -x2;
    halfZ = -halfZ;
  }

  double halfX((x1 + x2) / 2.);

  GlobalPoint pos_outer(refPlane->toGlobal(LocalPoint(0., y1, 0.)));
  GlobalPoint pos_inner(refPlane->toGlobal(LocalPoint(0., -y1, 0.)));
  GlobalPoint pos_zplus(refPlane->toGlobal(LocalPoint(0., 0., halfZ)));
  GlobalPoint pos_zminus(refPlane->toGlobal(LocalPoint(0., 0., -halfZ)));
  GlobalPoint pos_phiplus(refPlane->toGlobal(LocalPoint(-halfX, 0., 0.)));
  GlobalPoint pos_phiminus(refPlane->toGlobal(LocalPoint(halfX, 0., 0.)));

  LogTrace("MagGeoBuilder") << "RN            " << theRN << newln  // RN line needs extra space to line up
                            << "pos_outer    " << pos_outer << " " << pos_outer.perp() << " " << pos_outer.phi()
                            << newln << "pos_inner    " << pos_inner << " " << pos_inner.perp() << " "
                            << pos_inner.phi() << newln << "pos_zplus    " << pos_zplus << " " << pos_zplus.perp()
                            << " " << pos_zplus.phi() << newln << "pos_zminus   " << pos_zminus << " "
                            << pos_zminus.perp() << " " << pos_zminus.phi() << newln << "pos_phiplus  " << pos_phiplus
                            << " " << pos_phiplus.perp() << " " << pos_phiplus.phi() << newln << "pos_phiminus "
                            << pos_phiminus << " " << pos_phiminus.perp() << " " << pos_phiminus.phi();

  // Check ordering.
  if (debug) {
    if (pos_outer.perp() < pos_inner.perp()) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_outer < pos_inner for trapezoid";
    }
    if (pos_zplus.z() < pos_zminus.z()) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_zplus < pos_zminus for trapezoid";
    }
    if (Geom::Phi<float>(pos_phiplus.phi() - pos_phiminus.phi()) < 0.) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_phiplus < pos_phiminus for trapezoid";
    }
  }

  // Local Y axis of the faces at +-phi. The local X is = global Z.
  GlobalVector y_phiplus = (refPlane->toGlobal(LocalVector((tan(alpha1) * y1 - (x2 - x1) / 2.), y1, 0.))).unit();
  GlobalVector y_phiminusV = (refPlane->toGlobal(LocalVector((tan(alpha1) * y1 + (x2 - x1) / 2.), y1, 0.)));
  GlobalVector y_phiminus = y_phiminusV.unit();

  LogTrace("MagGeoBuilder") << "y_phiplus  " << y_phiplus << newln << "y_phiminus " << y_phiminus;

  Surface::RotationType rot_R(planeZAxis, planeXAxis);
  Surface::RotationType rot_Z(planeXAxis, planeYAxis);
  Surface::RotationType rot_phiplus(planeZAxis, y_phiplus);
  Surface::RotationType rot_phiminus(planeZAxis, y_phiminus);

  // FIXME: use builder
  surfaces[outer] = new Plane(pos_outer, rot_R);
  surfaces[inner] = new Plane(pos_inner, rot_R);
  surfaces[zplus] = new Plane(pos_zplus, rot_Z);
  surfaces[zminus] = new Plane(pos_zminus, rot_Z);
  surfaces[phiplus] = new Plane(pos_phiplus, rot_phiplus);
  surfaces[phiminus] = new Plane(pos_phiminus, rot_phiminus);

  // Save volume boundaries
  theRMin = std::abs(surfaces[inner]->toLocal(GlobalPoint(0, 0, 0)).z());
  theRMax = std::abs(surfaces[outer]->toLocal(GlobalPoint(0, 0, 0)).z());

  // Setting "absolute" phi boundary spots a problem in V85l
  // e.g. vol 139 (origin: small inconsistency in volumes.txt)
  // So let's keep phi at middle phi- plane...
  //  FIXME: Remove this workaround once support for v85l is dropped.
  if (name.substr(0, 3) == "V_Z") {
    thePhiMin = surfaces[phiminus]->position().phi();
  } else {
    thePhiMin = min((pos_phiminus + y_phiminusV).phi(), (pos_phiminus - y_phiminusV).phi());
  }
  LogTrace("MagGeoBuilder") << "rot_R    " << surfaces[outer]->toGlobal(LocalVector(0., 0., 1.)) << newln << "rot_Z    "
                            << surfaces[zplus]->toGlobal(LocalVector(0., 0., 1.)) << newln << "rot_phi+ "
                            << surfaces[phiplus]->toGlobal(LocalVector(0., 0., 1.)) << " phi "
                            << surfaces[phiplus]->toGlobal(LocalVector(0., 0., 1.)).phi() << newln << "rot_phi- "
                            << surfaces[phiminus]->toGlobal(LocalVector(0., 0., 1.)) << " phi "
                            << surfaces[phiminus]->toGlobal(LocalVector(0., 0., 1.)).phi();
  if (debug) {
    if (std::abs(surfaces[phiminus]->position().phi() - thePhiMin) > 1e-5) {
      LogTrace("MagGeoBuilder") << " Trapez. phiMin = " << thePhiMin;
    }
  }
}

void volumeHandle::buildTubs() {
  LogTrace("MagGeoBuilder") << "Building tubs surfaces...: ";

  DDTubs tubs(solid.solid());

  // Old DD needs mm to cm conversion, but DD4hep needs no conversion.
  // convertUnits should be defined appropriately.
  double zhalf = convertUnits(tubs.dZ());
  double rIn = convertUnits(tubs.rMin());
  double rOut = convertUnits(tubs.rMax());
  double startPhi = convertDegToRad(tubs.startPhi());
  double deltaPhi = convertDegToRad(tubs.endPhi()) - startPhi;

  // Limit to range -pi to pi
  Geom::NormalizeWrapper<double, Geom::MinusPiToPi>::normalize(startPhi);  // FIXME: is it needed?

  LogTrace("MagGeoBuilder") << "zhalf    " << zhalf << newln << "rIn      " << rIn << newln << "rOut     " << rOut
                            << newln << "startPhi " << startPhi << newln << "deltaPhi " << deltaPhi;

  // recalculate center: (for a DDTubs, DDD gives 0,0,Z)
  double rCentr = (rIn + rOut) / 2.;
  Geom::Phi<double> phiCenter(startPhi + deltaPhi / 2.);
  center_ = refPlane->toGlobal(LocalPoint(rCentr * cos(phiCenter), rCentr * sin(phiCenter), 0.));
  // For cons and tubs RN = R.
  theRN = rCentr;

  // FIXME: use builder
  surfaces[outer] = new Cylinder(rOut, Surface::PositionType(0, 0, center_.z()), Surface::RotationType());

  // The inner cylider may be degenreate. Not easy to have a null surface
  // in the current implementation (surfaces[inner] is a RCP!)

  surfaces[inner] = new Cylinder(rIn, Surface::PositionType(0, 0, center_.z()), Surface::RotationType());

  // All other surfaces
  buildPhiZSurf(startPhi, deltaPhi, zhalf, rCentr);

  // Check ordering.
  if (debug) {
    if (dynamic_cast<const Cylinder *>(&(*surfaces[outer]))->radius() <
        dynamic_cast<const Cylinder *>(&(*surfaces[inner]))->radius()) {
      LogTrace("MagGeometry") << "*** WARNING: pos_outer < pos_inner ";
    }
  }
  // Save volume boundaries
  theRMin = rIn;
  theRMax = rOut;
  thePhiMin = surfaces[phiminus]->position().phi();
}

/*
 *  Compute parameters for a cone section 
 *
 *  \author N. Amapane - INFN Torino
 */

#include "DataFormats/GeometrySurface/interface/SimpleConeBounds.h"

void volumeHandle::buildCons() {
  LogTrace("MagGeoBuilder") << "Building cons surfaces...: ";

  DDCons cons(solid.solid());

  // Old DD needs mm to cm conversion, but DD4hep needs no conversion.
  // convertUnits should be defined appropriately.
  double zhalf = convertUnits(cons.dZ());
  double rInMinusZ = convertUnits(cons.rMin1());
  double rOutMinusZ = convertUnits(cons.rMax1());
  double rInPlusZ = convertUnits(cons.rMin2());
  double rOutPlusZ = convertUnits(cons.rMax2());
  double startPhi = convertDegToRad(cons.startPhi());
  double deltaPhi = convertDegToRad(cons.endPhi()) - startPhi;

  // Limit to range -pi to pi
  Geom::NormalizeWrapper<double, Geom::MinusPiToPi>::normalize(startPhi);

  LogTrace("MagGeoBuilder") << "zhalf      " << zhalf << newln << "rInMinusZ  " << rInMinusZ << newln << "rOutMinusZ "
                            << rOutMinusZ << newln << "rInPlusZ   " << rInPlusZ << newln << "rOutPlusZ  " << rOutPlusZ
                            << newln << "phiFrom    " << startPhi << newln << "deltaPhi   " << deltaPhi;

  // recalculate center: (for a DDCons, DDD gives 0,0,Z)
  double rZmin = (rInMinusZ + rOutMinusZ) / 2.;
  double rZmax = (rInPlusZ + rOutPlusZ) / 2.;
  double rCentr = (rZmin + rZmax) / 2.;
  Geom::Phi<double> phiCenter(startPhi + deltaPhi / 2.);
  center_ = refPlane->toGlobal(LocalPoint(rCentr * cos(phiCenter), rCentr * sin(phiCenter), 0.));
  // For cons and tubs RN = R.
  theRN = rCentr;

  const double epsilon = 1e-5;

  if (std::abs(rInPlusZ - rInMinusZ) < epsilon) {  // Cylinder
    // FIXME: use builder
    surfaces[inner] = new Cylinder(rInMinusZ, Surface::PositionType(), Surface::RotationType());

  } else {  // Cone
    // FIXME: trick to compute vertex and angle...
    SimpleConeBounds cb(center_.z() - zhalf, rInMinusZ, rInMinusZ, center_.z() + zhalf, rInPlusZ, rInPlusZ);

    surfaces[inner] =
        new Cone(Surface::PositionType(0, 0, center_.z()), Surface::RotationType(), cb.vertex(), cb.openingAngle());
  }
  if (std::abs(rOutPlusZ - rOutMinusZ) < epsilon) {  // Cylinder
    surfaces[outer] = new Cylinder(rOutMinusZ, Surface::PositionType(0, 0, center_.z()), Surface::RotationType());
  } else {  // Cone
    // FIXME: trick to compute vertex and angle...
    SimpleConeBounds cb(center_.z() - zhalf, rOutMinusZ, rOutMinusZ, center_.z() + zhalf, rOutPlusZ, rOutPlusZ);

    surfaces[outer] =
        new Cone(Surface::PositionType(0, 0, center_.z()), Surface::RotationType(), cb.vertex(), cb.openingAngle());

    LogTrace("MagGeoBuilder") << "Outer surface: cone, vtx: " << cb.vertex() << " angle " << cb.openingAngle();
  }
  // All other surfaces
  buildPhiZSurf(startPhi, deltaPhi, zhalf, rCentr);

  // Save volume boundaries
  theRMin = min(rInMinusZ, rInPlusZ);
  theRMax = max(rOutMinusZ, rOutPlusZ);
  thePhiMin = surfaces[phiminus]->position().phi();
}

/*
 *  Compute parameters for a truncated cylinder section. 
 *  In the current geometry TruncTubs are described starting from startPhi = 0,
 *  so that the volume at ~pi/2 has a nonzero rotation, contrary to any other 
 *  volume type in the same geometry. This is not a problem as long as proper
 *  local to global transformation are used.
 *  
 *
 *  \author N. Amapane - INFN Torino
 */

void volumeHandle::buildTruncTubs() {
  LogTrace("MagGeoBuilder") << "Building TruncTubs surfaces...: ";

  DDTruncTubs tubs(solid.solid());

  // Old DD needs mm to cm conversion, but DD4hep needs no conversion.
  // convertUnits should be defined appropriately.
  double zhalf = convertUnits(tubs.zHalf());            // half of the z-Axis
  double rIn = convertUnits(tubs.rMin());               // inner radius
  double rOut = convertUnits(tubs.rMax());              // outer radius
  double startPhi = tubs.startPhi();                    // angular start of the tube-section
  double deltaPhi = tubs.deltaPhi();                    // angular span of the tube-section
  double cutAtStart = convertUnits(tubs.cutAtStart());  // truncation at begin of the tube-section
  double cutAtDelta = convertUnits(tubs.cutAtDelta());  // truncation at end of the tube-section
  bool cutInside = tubs.cutInside();                    // true, if truncation is on the inner side of the tube-section

  LogTrace("MagGeoBuilder") << "zhalf      " << zhalf << newln << "rIn        " << rIn << newln << "rOut       " << rOut
                            << newln << "startPhi   " << startPhi << newln << "deltaPhi   " << deltaPhi << newln
                            << "cutAtStart " << cutAtStart << newln << "cutAtDelta " << cutAtDelta << newln
                            << "cutInside  " << cutInside;

  //
  //  GlobalVector planeXAxis = refPlane->toGlobal(LocalVector( 1, 0, 0));
  //  GlobalVector planeYAxis = refPlane->toGlobal(LocalVector( 0, 1, 0));
  GlobalVector planeZAxis = refPlane->toGlobal(LocalVector(0, 0, 1));

  Sides cyl_side;
  Sides plane_side;
  double rCyl = 0;
  double rCentr = 0;
  if (cutInside) {
    cyl_side = outer;
    rCyl = rOut;
    plane_side = inner;
    rCentr = (max(max(rIn, cutAtStart), cutAtDelta) + rOut) / 2.;
  } else {
    cyl_side = inner;
    rCyl = rIn;
    plane_side = outer;
    rCentr = (rIn + min(min(rOut, cutAtStart), cutAtDelta)) / 2.;
  }

  // Recalculate center: (for a DDTruncTubs, DDD gives 0,0,Z)
  // The R of center is in the middle of the arc of cylinder fully contained
  // in the trunctubs.
  Geom::Phi<double> phiCenter(startPhi + deltaPhi / 2.);
  center_ = refPlane->toGlobal(LocalPoint(rCentr * cos(phiCenter), rCentr * sin(phiCenter), 0.));

  // FIXME!! Actually should recompute RN from pos_Rplane; should not
  // matter anyhow
  theRN = rCentr;

  // For simplicity, the position of the cut plane is taken at one of the
  // phi edges, where R is known
  // FIXME! move to center of plane
  // FIXME: compute. in double prec (not float)
  GlobalPoint pos_Rplane(refPlane->toGlobal(LocalPoint(LocalPoint::Cylindrical(cutAtStart, startPhi, 0.))));

  // Compute angle of the cut plane
  // Got any easier formula?
  // FIXME: check that it still holds for cutAtStart > cutAtDelta
  double c = sqrt(cutAtDelta * cutAtDelta + cutAtStart * cutAtStart - 2 * cutAtDelta * cutAtStart * cos(deltaPhi));
  double alpha = startPhi - asin(sin(deltaPhi) * cutAtDelta / c);
  GlobalVector x_Rplane = refPlane->toGlobal(LocalVector(cos(alpha), sin(alpha), 0));
  Surface::RotationType rot_R(planeZAxis, x_Rplane);

  // FIXME: use builder
  surfaces[plane_side] = new Plane(pos_Rplane, rot_R);
  surfaces[cyl_side] = new Cylinder(rCyl, Surface::PositionType(0, 0, center_.z()), Surface::RotationType());

  // Build lateral surfaces.
  // Note that with the choice of center descrived above, the
  // plane position (origin of r.f.) of the smallest phi face
  // will be its center, while this is not true for the largest phi face.
  buildPhiZSurf(startPhi, deltaPhi, zhalf, rCentr);

  if (debug) {
    LogTrace("MagGeoBuilder") << "pos_Rplane    " << pos_Rplane << " " << pos_Rplane.perp() << " " << pos_Rplane.phi()
                              << newln << "rot_R         " << surfaces[plane_side]->toGlobal(LocalVector(0., 0., 1.))
                              << " phi " << surfaces[plane_side]->toGlobal(LocalVector(0., 0., 1.)).phi() << newln
                              << "cyl radius     " << rCyl;

    //   // Check ordering.
    if ((pos_Rplane.perp() < rCyl) != cutInside) {
      LogTrace("MagGeoBuilder") << "*** WARNING: pos_outer < pos_inner ";
    }
  }
  // Save volume boundaries
  theRMin = rIn;
  theRMax = rOut;
  thePhiMin = surfaces[phiminus]->position().phi();
}

#include "buildPseudoTrap.icc"
