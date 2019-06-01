///////////////////////////////////////////////////////////////////////////////
// File: DDHCalFibreBundle.cc
// Description: Create & Position fibre bundles in mother
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalFibreBundle.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalFibreBundle::DDHCalFibreBundle() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating an instance";
#endif
}

DDHCalFibreBundle::~DDHCalFibreBundle() {}

void DDHCalFibreBundle::initialize(const DDNumericArguments& nArgs,
                                   const DDVectorArguments& vArgs,
                                   const DDMapArguments&,
                                   const DDStringArguments& sArgs,
                                   const DDStringVectorArguments&) {
  deltaPhi = nArgs["DeltaPhi"];
  deltaZ = nArgs["DeltaZ"];
  numberPhi = int(nArgs["NumberPhi"]);
  material = sArgs["Material"];
  areaSection = vArgs["AreaSection"];
  rStart = vArgs["RadiusStart"];
  rEnd = vArgs["RadiusEnd"];
  bundle = dbl_to_int(vArgs["Bundles"]);
  tilt = nArgs["TiltAngle"];

  idNameSpace = DDCurrentNamespace::ns();
  childPrefix = sArgs["Child"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Parent " << parent().name() << " with " << bundle.size()
                               << " children with prefix " << childPrefix << ", material " << material << " with "
                               << numberPhi << " bundles along phi; width of"
                               << " mother " << deltaZ << " along Z, " << convertRadToDeg(deltaPhi)
                               << " along phi and with " << rStart.size() << " different bundle types";
  for (unsigned int i = 0; i < areaSection.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Child[" << i << "] Area " << areaSection[i] << " R at Start "
                                 << rStart[i] << " R at End " << rEnd[i];
  edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: NameSpace " << idNameSpace << " Tilt Angle "
                               << convertRadToDeg(tilt) << " Bundle type at different positions";
  for (unsigned int i = 0; i < bundle.size(); ++i) {
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Position[" << i << "] "
                                 << " with Type " << bundle[i];
  }
#endif
}

void DDHCalFibreBundle::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName matname(DDSplit(material).first, DDSplit(material).second);
  DDMaterial matter(matname);

  // Create the rotation matrices
  double dPhi = deltaPhi / numberPhi;
  std::vector<DDRotation> rotation;
  for (int i = 0; i < numberPhi; ++i) {
    double phi = -0.5 * deltaPhi + (i + 0.5) * dPhi;
    double phideg = convertRadToDeg(phi);
    std::string rotstr = "R0" + std::to_string(phideg);
    DDRotation rot = DDRotation(DDName(rotstr, idNameSpace));
    if (!rot) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating a new "
                                   << "rotation " << rotstr << "\t" << 90 << "," << phideg << "," << 90 << ","
                                   << (phideg + 90) << ", 0, 0";
#endif
      rot = DDrot(DDName(rotstr, idNameSpace), 90._deg, phi, 90._deg, (90._deg + phi), 0, 0);
    }
    rotation.emplace_back(rot);
  }

  // Create the solids and logical parts
  std::vector<DDLogicalPart> logs;
  for (unsigned int i = 0; i < areaSection.size(); ++i) {
    double r0 = rEnd[i] / std::cos(tilt);
    double dStart = areaSection[i] / (2 * dPhi * rStart[i]);
    double dEnd = areaSection[i] / (2 * dPhi * r0);
    std::string name = childPrefix + std::to_string(i);
    DDSolid solid = DDSolidFactory::cons(DDName(name, idNameSpace),
                                         0.5 * deltaZ,
                                         rStart[i] - dStart,
                                         rStart[i] + dStart,
                                         r0 - dEnd,
                                         r0 + dEnd,
                                         -0.5 * dPhi,
                                         dPhi);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: Creating a new solid " << name << " a cons with dZ " << deltaZ
                                 << " rStart " << rStart[i] - dStart << ":" << rStart[i] + dStart << " rEnd "
                                 << r0 - dEnd << ":" << r0 + dEnd << " Phi " << convertRadToDeg(-0.5 * dPhi) << ":"
                                 << convertRadToDeg(0.5 * dPhi);
#endif
    DDLogicalPart log(DDName(name, idNameSpace), matter, solid);
    logs.emplace_back(log);
  }

  // Now posiiton them
  int copy = 0;
  int nY = (int)(bundle.size()) / numberPhi;
  for (unsigned int i = 0; i < bundle.size(); i++) {
    DDTranslation tran(0, 0, 0);
    int ir = (int)(i) / nY;
    if (ir >= numberPhi)
      ir = numberPhi - 1;
    int ib = bundle[i];
    copy++;
    if (ib >= 0 && ib < (int)(logs.size())) {
      cpv.position(logs[ib], mother, copy, tran, rotation[ir]);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalFibreBundle: " << logs[ib].name() << " number " << copy
                                   << " positioned in " << mother << " at " << tran << " with " << rotation[ir];
#endif
    }
  }
}
