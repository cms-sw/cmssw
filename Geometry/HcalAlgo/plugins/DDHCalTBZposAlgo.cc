///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTBZposAlgo.cc
// Description: Position inside the mother by shifting along Z given by eta
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBZposAlgo.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalTBZposAlgo::DDHCalTBZposAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Creating an instance";
#endif
}

DDHCalTBZposAlgo::~DDHCalTBZposAlgo() {}

void DDHCalTBZposAlgo::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments&,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments&) {
  eta = nArgs["Eta"];
  theta = 2.0 * atan(exp(-eta));
  shiftX = nArgs["ShiftX"];
  shiftY = nArgs["ShiftY"];
  zoffset = nArgs["Zoffset"];
  dist = nArgs["Distance"];
  tilt = nArgs["TiltAngle"];
  copyNumber = int(nArgs["Number"]);
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parameters for position"
                               << "ing--"
                               << " Eta " << eta << "\tTheta " << convertRadToDeg(theta) << "\tShifts " << shiftX
                               << ", " << shiftY << " along x, y "
                               << "axes; \tZoffest " << zoffset << "\tRadial Distance " << dist << "\tTilt angle "
                               << convertRadToDeg(tilt) << "\tcopyNumber " << copyNumber;

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: Parent " << parent().name() << "\tChild " << childName
                               << " NameSpace " << idNameSpace;
#endif
}

void DDHCalTBZposAlgo::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);

  double thetax = 90._deg - theta;
  double z = zoffset + dist * tan(thetax);
  double x = shiftX - shiftY * sin(tilt);
  double y = shiftY * cos(tilt);
  DDTranslation tran(x, y, z);
  DDRotation rot;
  if (tilt != 0) {
    std::string rotstr = "R" + formatAsDegrees(tilt);
    rot = DDRotation(DDName(rotstr, idNameSpace));
    if (!rot) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalZposAlgo: Creating a rotation " << DDName(rotstr, idNameSpace) << "\t90, "
                                   << convertRadToDeg(tilt) << ",90," << (90 + convertRadToDeg(tilt)) << ", 0, 0";
#endif
      rot = DDrot(DDName(rotstr, idNameSpace), 90._deg, tilt, 90._deg, (90._deg + tilt), 0.0, 0.0);
    }
  }
  cpv.position(child, mother, copyNumber, tran, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBZposAlgo: " << child << " number " << copyNumber << " positioned in "
                               << mother << " at " << tran << " with " << rot;
#endif
}
