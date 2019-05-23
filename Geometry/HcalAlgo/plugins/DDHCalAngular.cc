///////////////////////////////////////////////////////////////////////////////
// File: DDHCalAngular.cc
// Description: Position inside the mother according to (eta,phi)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "Geometry/HcalAlgo/plugins/DDHCalAngular.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalAngular::DDHCalAngular() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalAngular: Creating an instance";
#endif
}

DDHCalAngular::~DDHCalAngular() {}

void DDHCalAngular::initialize(const DDNumericArguments& nArgs,
                               const DDVectorArguments&,
                               const DDMapArguments&,
                               const DDStringArguments& sArgs,
                               const DDStringVectorArguments&) {
  startAngle = nArgs["startAngle"];
  rangeAngle = nArgs["rangeAngle"];
  shiftX = nArgs["shiftX"];
  shiftY = nArgs["shiftY"];
  zoffset = nArgs["zoffset"];
  n = int(nArgs["n"]);
  startCopyNo = int(nArgs["startCopyNo"]);
  incrCopyNo = int(nArgs["incrCopyNo"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalAngular: Parameters for positioning "
                               << "-- " << n << " copies in " << convertRadToDeg(rangeAngle) << " from "
                               << convertRadToDeg(startAngle) << "\tShifts " << shiftX << ", " << shiftY
                               << " along x, y axes; \tZoffest " << zoffset << "\tStart and inremental copy nos "
                               << startCopyNo << ", " << incrCopyNo;
#endif
  rotns = sArgs["RotNameSpace"];
  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  DDName parentName = parent().name();
  edm::LogVerbatim("HCalGeom") << "DDHCalAngular: Parent " << parentName << "\tChild " << childName << "\tNameSpace "
                               << idNameSpace << "\tRotation Namespace " << rotns;
#endif
}

void DDHCalAngular::execute(DDCompactView& cpv) {
  double dphi = rangeAngle / n;
  double phix = startAngle;
  int copyNo = startCopyNo;
  double theta = 90._deg;
  for (int ii = 0; ii < n; ii++) {
    double phiy = phix + 90._deg;
    DDRotation rotation;
    std::string rotstr("NULL");

    static const double tol = 0.1;
    if (std::abs(phix) > tol) {
      rotstr = "R" + formatAsDegreesInInteger(phix);
      rotation = DDRotation(DDName(rotstr, rotns));
      if (!rotation) {
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HCalGeom") << "DDHCalAngular: Creating a rotation " << DDName(rotstr, idNameSpace) << "\t90, "
                                     << convertRadToDeg(phix) << ", 90, " << (90 + convertRadToDeg(phix)) << ", 0, 0";
#endif
        rotation = DDrot(DDName(rotstr, rotns), theta, phix, theta, phiy, 0, 0);
      }
    }

    double xpos = shiftX * cos(phix) - shiftY * sin(phix);
    double ypos = shiftX * sin(phix) + shiftY * cos(phix);
    DDTranslation tran(xpos, ypos, zoffset);

    DDName parentName = parent().name();
    cpv.position(DDName(childName, idNameSpace), parentName, copyNo, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalAngular: " << DDName(childName, idNameSpace) << " number " << copyNo
                                 << " positioned in " << parentName << " at " << tran << " with " << rotation;
#endif
    phix += dphi;
    copyNo += incrCopyNo;
  }
}
