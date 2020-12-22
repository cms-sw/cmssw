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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <DD4hep/DD4hepUnits.h>

#include <string>
#include <iterator>

using namespace SurfaceOrientation;
using namespace std;
using namespace magneticfield;
using namespace edm;

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

  // Convert from DD4hep units to cm
  center_ = GlobalPoint(transArray[0] / dd4hep::cm, transArray[1] / dd4hep::cm, transArray[2] / dd4hep::cm);

  // ASSUMPTION: volume names ends with "_NUM" where NUM is the volume number
  string volName = name;
  volName.erase(0, volName.rfind('_') + 1);
  volumeno = static_cast<unsigned short>(std::atoi(volName.c_str()));

  for (int i = 0; i < 6; ++i) {
    isAssigned[i] = false;
  }
  referencePlane(fv);
  switch (theShape) {
    case DDSolidShape::ddbox: {
      DDBox box(solid.solid());
      // Convert from DD4hep units to cm
      double halfX = box.x() / dd4hep::cm;
      double halfY = box.y() / dd4hep::cm;
      double halfZ = box.z() / dd4hep::cm;
      buildBox(halfX, halfY, halfZ);
    } break;
    case DDSolidShape::ddtrap: {
      DDTrap trap(solid.solid());
      double x1 = trap.bottomLow1() / dd4hep::cm;
      double x2 = trap.topLow1() / dd4hep::cm;
      double x3 = trap.bottomLow2() / dd4hep::cm;
      double x4 = trap.topLow2() / dd4hep::cm;
      double y1 = trap.high1() / dd4hep::cm;
      double y2 = trap.high2() / dd4hep::cm;
      double theta = trap.theta();
      double phi = trap.phi();
      double halfZ = trap.dZ() / dd4hep::cm;
      double alpha1 = trap.alpha1();
      double alpha2 = trap.alpha2();
      buildTrap(x1, x2, x3, x4, y1, y2, theta, phi, halfZ, alpha1, alpha2);
    } break;

    case DDSolidShape::ddcons: {
      DDCons cons(solid.solid());
      double zhalf = cons.dZ() / dd4hep::cm;
      double rInMinusZ = cons.rMin1() / dd4hep::cm;
      double rOutMinusZ = cons.rMax1() / dd4hep::cm;
      double rInPlusZ = cons.rMin2() / dd4hep::cm;
      double rOutPlusZ = cons.rMax2() / dd4hep::cm;
      double startPhi = cons.startPhi();
      double deltaPhi = reco::deltaPhi(cons.endPhi(), startPhi);
      buildCons(zhalf, rInMinusZ, rOutMinusZ, rInPlusZ, rOutPlusZ, startPhi, deltaPhi);
    } break;
    case DDSolidShape::ddtubs: {
      DDTubs tubs(solid.solid());
      double zhalf = tubs.dZ() / dd4hep::cm;
      double rIn = tubs.rMin() / dd4hep::cm;
      double rOut = tubs.rMax() / dd4hep::cm;
      double startPhi = tubs.startPhi();
      double deltaPhi = tubs.endPhi() - startPhi;
      buildTubs(zhalf, rIn, rOut, startPhi, deltaPhi);
    } break;
    case DDSolidShape::ddpseudotrap: {
      vector<double> d = solid.parameters();

      // The pseudo-trapezoid parameters are:
      // d[0] -- x1
      // d[1] -- x2
      // d[2] -- y1
      // d[3] -- y2
      // d[4] -- halfZ
      // d[5] -- radius
      // d[6] -- atMinusZ (0 or 1)
      // Note all are lengths except for the last one. The lengths come from
      // DD4hep in DD4hep units and must be converted to cm for use.

      if (d.size() >= 7)
        LogTrace("MagGeoBuilder") << " Pseudo trap params raw =  " << d[0] << ", " << d[1] << ", " << d[2] << ", "
                                  << d[3] << ", " << d[4] << ", " << d[5] << ", " << d[6];

      // Convert all but last parameter to cm (last one is a boolean).
      transform(d.begin(), --(d.end()), d.begin(), [](double val) { return val / dd4hep::cm; });

      if (d.size() >= 7)
        LogTrace("MagGeoBuilder") << " Pseudo trap params converted =  " << d[0] << ", " << d[1] << ", " << d[2] << ", "
                                  << d[3] << ", " << d[4] << ", " << d[5] << ", " << d[6];

      buildPseudoTrap(d[0], d[1], d[2], d[3], d[4], d[5], static_cast<bool>(d[6]));
    } break;
    case DDSolidShape::ddtrunctubs: {
      DDTruncTubs tubs(solid.solid());
      double zhalf = tubs.dZ() / dd4hep::cm;               // half of the z-Axis
      double rIn = tubs.rMin() / dd4hep::cm;               // inner radius
      double rOut = tubs.rMax() / dd4hep::cm;              // outer radius
      double startPhi = tubs.startPhi();                   // angular start of the tube-section
      double deltaPhi = tubs.deltaPhi();                   // angular span of the tube-section
      double cutAtStart = tubs.cutAtStart() / dd4hep::cm;  // truncation at begin of the tube-section
      double cutAtDelta = tubs.cutAtDelta() / dd4hep::cm;  // truncation at end of the tube-section
      bool cutInside = tubs.cutInside();  // true, if truncation is on the inner side of the tube-section
      buildTruncTubs(zhalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside);
    } break;
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

#include "buildBox.icc"
#include "buildTrap.icc"
#include "buildTubs.icc"
#include "buildCons.icc"
#include "buildPseudoTrap.icc"
#include "buildTruncTubs.icc"
