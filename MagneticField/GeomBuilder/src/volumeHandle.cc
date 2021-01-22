/*
 *  See header file for a description of this class.
 *
 *  \author N. Amapane - INFN Torino (original developer)
 */

#include "volumeHandle.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/CoordinateSets.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>
#include <iterator>
#include <iomanip>
#include <iostream>
#include <boost/lexical_cast.hpp>

using namespace SurfaceOrientation;
using namespace std;

#include "DataFormats/Math/interface/GeantUnits.h"

namespace {
  // Old DD returns lengths in mm, but CMS code uses cm
  template <class NumType>
  inline constexpr NumType convertUnits(NumType millimeters)  // Millimeters -> centimeters
  {
    return (geant_units::operators::convertMmToCm(millimeters));
  }
}  // namespace

MagGeoBuilderFromDDD::volumeHandle::volumeHandle(const DDExpandedView &fv, bool expand2Pi, bool debugVal)
    : magneticfield::BaseVolumeHandle(expand2Pi, debugVal) {
  name = fv.logicalPart().name().name();
  copyno = fv.copyno();
  solid = fv.logicalPart().solid();
  center_ = GlobalPoint(fv.translation().x() / cm, fv.translation().y() / cm, fv.translation().z() / cm);

  // ASSUMPTION: volume names ends with "_NUM" where NUM is the volume number
  string volName = name;
  volName.erase(0, volName.rfind('_') + 1);
  volumeno = boost::lexical_cast<unsigned short>(volName);

  for (int i = 0; i < 6; ++i) {
    isAssigned[i] = false;
  }

  if (debug) {
    cout.precision(7);
  }

  referencePlane(fv);

  if (solid.shape() == DDSolidShape::ddbox) {
    DDBox box(solid);
    double halfX = convertUnits(box.halfX());
    double halfY = convertUnits(box.halfY());
    double halfZ = convertUnits(box.halfZ());
    buildBox(halfX, halfY, halfZ);
  } else if (solid.shape() == DDSolidShape::ddtrap) {
    DDTrap trap(solid);
    double x1 = convertUnits(trap.x1());
    double x2 = convertUnits(trap.x2());
    double x3 = convertUnits(trap.x3());
    double x4 = convertUnits(trap.x4());
    double y1 = convertUnits(trap.y1());
    double y2 = convertUnits(trap.y2());
    double theta = trap.theta();
    double phi = trap.phi();
    double halfZ = convertUnits(trap.halfZ());
    double alpha1 = trap.alpha1();
    double alpha2 = trap.alpha2();
    buildTrap(x1, x2, x3, x4, y1, y2, theta, phi, halfZ, alpha1, alpha2);
  } else if (solid.shape() == DDSolidShape::ddcons) {
    DDCons cons(solid);
    double zhalf = convertUnits(cons.zhalf());
    double rInMinusZ = convertUnits(cons.rInMinusZ());
    double rOutMinusZ = convertUnits(cons.rOutMinusZ());
    double rInPlusZ = convertUnits(cons.rInPlusZ());
    double rOutPlusZ = convertUnits(cons.rOutPlusZ());
    double startPhi = cons.phiFrom();
    double deltaPhi = cons.deltaPhi();
    buildCons(zhalf, rInMinusZ, rOutMinusZ, rInPlusZ, rOutPlusZ, startPhi, deltaPhi);
  } else if (solid.shape() == DDSolidShape::ddtubs) {
    DDTubs tubs(solid);
    double zhalf = convertUnits(tubs.zhalf());
    double rIn = convertUnits(tubs.rIn());
    double rOut = convertUnits(tubs.rOut());
    double startPhi = tubs.startPhi();
    double deltaPhi = tubs.deltaPhi();
    buildTubs(zhalf, rIn, rOut, startPhi, deltaPhi);
  } else if (solid.shape() == DDSolidShape::ddpseudotrap) {
    DDPseudoTrap ptrap(solid);
    double x1 = convertUnits(ptrap.x1());
    double x2 = convertUnits(ptrap.x2());
    double y1 = convertUnits(ptrap.y1());
    double y2 = convertUnits(ptrap.y2());
    double halfZ = convertUnits(ptrap.halfZ());
    double radius = convertUnits(ptrap.radius());
    bool atMinusZ = ptrap.atMinusZ();
    buildPseudoTrap(x1, x2, y1, y2, halfZ, radius, atMinusZ);
  } else if (solid.shape() == DDSolidShape::ddtrunctubs) {
    DDTruncTubs tubs(solid);
    double zhalf = convertUnits(tubs.zHalf());            // half of the z-Axis
    double rIn = convertUnits(tubs.rIn());                // inner radius
    double rOut = convertUnits(tubs.rOut());              // outer radius
    double startPhi = tubs.startPhi();                    // angular start of the tube-section
    double deltaPhi = tubs.deltaPhi();                    // angular span of the tube-section
    double cutAtStart = convertUnits(tubs.cutAtStart());  // truncation at begin of the tube-section
    double cutAtDelta = convertUnits(tubs.cutAtDelta());  // truncation at end of the tube-section
    bool cutInside = tubs.cutInside();  // true, if truncation is on the inner side of the tube-section
    buildTruncTubs(zhalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside);
  } else {
    cout << "volumeHandle ctor: Unexpected solid: " << DDSolidShapesName::name(solid.shape()) << endl;
  }

  // NOTE: Table name and master sector are no longer taken from xml!
  //   DDsvalues_type sv(fv.mergedSpecifics());

  //   { // Extract the name of associated field file.
  //     std::vector<std::string> temp;
  //     std::string pname = "table";
  //     DDValue val(pname);
  //     DDsvalues_type sv(fv.mergedSpecifics());
  //     if (DDfetch(&sv,val)) {
  //       temp = val.strings();
  //       if (temp.size() != 1) {
  // 	cout << "*** WARNING: volume has > 1 SpecPar " << pname << endl;
  //       }
  //       magFile = temp[0];

  //       string find="[copyNo]";
  //       std::size_t j;
  //       for ( ; (j = magFile.find(find)) != string::npos ; ) {
  // 	stringstream conv;
  // 	conv << setfill('0') << setw(2) << copyno;
  // 	string repl;
  // 	conv >> repl;
  // 	magFile.replace(j, find.length(), repl);
  //       }

  //     } else {
  //       cout << "*** WARNING: volume does not have a SpecPar " << pname << endl;
  //       cout << " DDsvalues_type:  " << fv.mergedSpecifics() << endl;
  //     }
  //   }

  //   { // Extract the number of the master sector.
  //     std::vector<double> temp;
  //     const std::string pname = "masterSector";
  //     DDValue val(pname);
  //     if (DDfetch(&sv,val)) {
  //       temp = val.doubles();
  //       if (temp.size() != 1) {
  //  	cout << "*** WARNING: volume has > 1 SpecPar " << pname << endl;
  //       }
  //       masterSector = int(temp[0]+.5);
  //     } else {
  //       if (MagGeoBuilderFromDDD::debug) {
  // 	cout << "Volume does not have a SpecPar " << pname
  // 	     << " using: " << copyno << endl;
  // 	cout << " DDsvalues_type:  " << fv.mergedSpecifics() << endl;
  //       }
  //       masterSector = copyno;
  //     }
  //   }

  // Get material for this volume
  if (fv.logicalPart().material().name().name() == "Iron")
    isIronFlag = true;

  if (debug) {
    cout << " RMin =  " << theRMin << endl;
    cout << " RMax =  " << theRMax << endl;

    if (theRMin < 0 || theRN < theRMin || theRMax < theRN)
      cout << "*** WARNING: wrong RMin/RN/RMax , shape: " << DDSolidShapesName::name(shape()) << endl;

    cout << "Summary: " << name << " " << copyno << " Shape= " << DDSolidShapesName::name(shape()) << " trasl "
         << center() << " R " << center().perp() << " phi " << center().phi() << " magFile " << magFile
         << " Material= " << fv.logicalPart().material().name() << " isIron= " << isIronFlag
         << " masterSector= " << masterSector << std::endl;

    cout << " Orientation of surfaces:";
    std::string sideName[3] = {"positiveSide", "negativeSide", "onSurface"};
    for (int i = 0; i < 6; ++i) {
      cout << "  " << i << ":" << sideName[surfaces[i]->side(center_, 0.3)];
    }
    cout << endl;
  }
}

void MagGeoBuilderFromDDD::volumeHandle::referencePlane(const DDExpandedView &fv) {
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
  DD3Vector x, y, z;
  fv.rotation().GetComponents(x, y, z);
  if (debug) {
    if (x.Cross(y).Dot(z) < 0.5) {
      cout << "*** WARNING: Rotation is not RH " << endl;
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
    cout << "Refplane pos  " << refPlane->position() << endl;

    // See comments above for the conventions for orientation.
    LocalVector globalZdir(0., 0., 1.);  // Local direction of the axis along global Z
    if (solid.shape() == DDSolidShape::ddpseudotrap) {
      globalZdir = LocalVector(0., 1., 0.);
    }
    if (refPlane->toGlobal(globalZdir).z() < 0.) {
      globalZdir = -globalZdir;
    }

    float chk = refPlane->toGlobal(globalZdir).dot(GlobalVector(0, 0, 1));
    if (chk < .999)
      cout << "*** WARNING RefPlane check failed!***" << chk << endl;
  }
}

std::vector<VolumeSide> MagGeoBuilderFromDDD::volumeHandle::sides() const {
  std::vector<VolumeSide> result;
  for (int i = 0; i < 6; ++i) {
    // If this is just a master volume out of wich a 2pi volume
    // should be built (e.g. central cylinder), skip the phi boundaries.
    if (expand && (i == phiplus || i == phiminus))
      continue;

    // FIXME: Skip null inner degenerate cylindrical surface
    if (solid.shape() == DDSolidShape::ddtubs && i == SurfaceOrientation::inner && theRMin < 0.001)
      continue;

    ReferenceCountingPointer<Surface> s = const_cast<Surface *>(surfaces[i].get());
    result.push_back(VolumeSide(s, GlobalFace(i), surfaces[i]->side(center_, 0.3)));
  }
  return result;
}

using volumeHandle = MagGeoBuilderFromDDD::volumeHandle;
using namespace magneticfield;

#include "buildBox.icc"
#include "buildTrap.icc"
#include "buildTubs.icc"
#include "buildCons.icc"
#include "buildPseudoTrap.icc"
#include "buildTruncTubs.icc"
