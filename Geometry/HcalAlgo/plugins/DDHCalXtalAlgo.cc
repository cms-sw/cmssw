///////////////////////////////////////////////////////////////////////////////
// File: DDHCalXtalAlgo.cc
// Description: Positioning the crystal (layers) in the mother volume
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "Geometry/HcalAlgo/plugins/DDHCalXtalAlgo.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalXtalAlgo::DDHCalXtalAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo info: Creating an instance";
#endif
}

DDHCalXtalAlgo::~DDHCalXtalAlgo() {}

void DDHCalXtalAlgo::initialize(const DDNumericArguments& nArgs,
                                const DDVectorArguments&,
                                const DDMapArguments&,
                                const DDStringArguments& sArgs,
                                const DDStringVectorArguments& vsArgs) {
  radius = nArgs["Radius"];
  offset = nArgs["Offset"];
  dx = nArgs["Dx"];
  dz = nArgs["Dz"];
  angwidth = nArgs["AngWidth"];
  iaxis = int(nArgs["Axis"]);
  names = vsArgs["Names"];

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo::Parameters for positioning:"
                               << " Axis " << iaxis << "\tRadius " << radius << "\tOffset " << offset << "\tDx " << dx
                               << "\tDz " << dz << "\tAngWidth " << convertRadToDeg(angwidth) << "\tNumbers "
                               << names.size();
  for (unsigned int i = 0; i < names.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\tnames[" << i << "] = " << names[i];
#endif
  idNameSpace = DDCurrentNamespace::ns();
  idName = sArgs["ChildName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: Parent " << parent().name() << "\tChild " << idName << " NameSpace "
                               << idNameSpace;
#endif
}

void DDHCalXtalAlgo::execute(DDCompactView& cpv) {
  double theta[3], phi[3], pos[3];
  phi[0] = 0;
  phi[1] = 90._deg;
  theta[1 - iaxis] = 90._deg;
  pos[1 - iaxis] = 0;
  int number = (int)(names.size());
  for (int i = 0; i < number; i++) {
    double angle = 0.5 * angwidth * (2 * i + 1 - number);
    theta[iaxis] = 90._deg + angle;
    if (angle > 0) {
      theta[2] = angle;
      phi[2] = iaxis * 90._deg;
    } else {
      theta[2] = -angle;
      phi[2] = (2 - 3 * iaxis) * 90._deg;
    }
    pos[iaxis] = angle * (dz + radius);
    pos[2] = dx * std::abs(sin(angle)) + offset;

    DDRotation rotation;
    std::string rotstr = names[i];
    DDTranslation tran(pos[0], pos[1], pos[2]);
    DDName parentName = parent().name();

    static const double tol = 0.01_deg;  // 0.01 degree
    if (std::abs(angle) > tol) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: Creating a rotation " << rotstr << "\t"
                                   << convertRadToDeg(theta[0]) << "," << convertRadToDeg(phi[0]) << ","
                                   << convertRadToDeg(theta[1]) << "," << convertRadToDeg(phi[1]) << ","
                                   << convertRadToDeg(theta[2]) << "," << convertRadToDeg(phi[2]);
#endif
      rotation = DDrot(DDName(rotstr, idNameSpace), theta[0], phi[0], theta[1], phi[1], theta[2], phi[2]);
    }
    cpv.position(DDName(idName, idNameSpace), parentName, i + 1, tran, rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalXtalAlgo: " << DDName(idName, idNameSpace) << " number " << i + 1
                                 << " positioned in " << parentName << " at " << tran << " with " << rotation;
#endif
  }
}
