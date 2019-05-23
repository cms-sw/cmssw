///////////////////////////////////////////////////////////////////////////////
// File: DDHCalEndcapModuleAlgo.cc
//   adapted from CCal(G4)HcalEndcap.cc
// Description: Geometry factory class for Hcal Endcap
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
#include "Geometry/HcalAlgo/plugins/DDHCalEndcapModuleAlgo.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalEndcapModuleAlgo::DDHCalEndcapModuleAlgo() {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Creating an instance";
#endif
}

DDHCalEndcapModuleAlgo::~DDHCalEndcapModuleAlgo() {}

void DDHCalEndcapModuleAlgo::initialize(const DDNumericArguments& nArgs,
                                        const DDVectorArguments& vArgs,
                                        const DDMapArguments&,
                                        const DDStringArguments& sArgs,
                                        const DDStringVectorArguments& vsArgs) {
  genMaterial = sArgs["MaterialName"];
  absorberMat = sArgs["AbsorberMat"];
  plasticMat = sArgs["PlasticMat"];
  scintMat = sArgs["ScintMat"];
  rotstr = sArgs["Rotation"];
  sectors = int(nArgs["Sectors"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: General material " << genMaterial << "\tAbsorber "
                               << absorberMat << "\tPlastic " << plasticMat << "\tScintillator " << scintMat
                               << "\tRotation " << rotstr << "\tSectors " << sectors;
#endif
  zMinBlock = nArgs["ZMinBlock"];
  zMaxBlock = nArgs["ZMaxBlock"];
  z1Beam = nArgs["Z1Beam"];
  ziDip = nArgs["ZiDip"];
  dzStep = nArgs["DzStep"];
  moduleThick = nArgs["ModuleThick"];
  layerThick = nArgs["LayerThick"];
  scintThick = nArgs["ScintThick"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Zmin " << zMinBlock << "\tZmax " << zMaxBlock << "\tZ1Beam "
                               << z1Beam << "\tZiDip " << ziDip << "\tDzStep " << dzStep << "\tModuleThick "
                               << moduleThick << "\tLayerThick " << layerThick << "\tScintThick " << scintThick;
#endif
  rMaxFront = nArgs["RMaxFront"];
  rMaxBack = nArgs["RMaxBack"];
  trimLeft = nArgs["TrimLeft"];
  trimRight = nArgs["TrimRight"];
  tolAbs = nArgs["TolAbs"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: RMaxFront " << rMaxFront << "\tRmaxBack " << rMaxBack
                               << "\tTrims " << trimLeft << ":" << trimRight << "\tTolAbs " << tolAbs;
#endif
  slopeBot = nArgs["SlopeBottom"];
  slopeTop = nArgs["SlopeTop"];
  slopeTopF = nArgs["SlopeTopFront"];
  modType = (int)(nArgs["ModType"]);
  modNumber = (int)(nArgs["ModNumber"]);
  layerType = (int)(nArgs["LayerType"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: slopeBot " << slopeBot << "\tslopeTop " << slopeTop
                               << "\tslopeTopF " << slopeTopF << "\tmodType " << modType << "\tmodNumber " << modNumber
                               << "\tlayerType " << layerType;
#endif
  layerNumber = dbl_to_int(vArgs["LayerNumber"]);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << layerNumber.size() << " layer Numbers";
  for (unsigned int i = 0; i < layerNumber.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "LayerNumber[" << i << "] = " << layerNumber[i];
#endif
  phiName = vsArgs["PhiName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << phiName.size() << " phi sectors";
  for (unsigned int i = 0; i < phiName.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "PhiName[" << i << "] = " << phiName[i];
#endif
  layerName = vsArgs["LayerName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << layerName.size() << " layers";
  for (unsigned int i = 0; i < layerName.size(); ++i)
    edm::LogVerbatim("HCalGeom") << "LayerName[" << i << "] = " << layerName[i];
#endif
  idName = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  idOffset = int(nArgs["IdOffset"]);
  modName = sArgs["ModName"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Parent " << parent().name() << "   " << modName << " idName "
                               << idName << " NameSpace " << idNameSpace << " Offset " << idOffset;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHCalEndcapModuleAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHCalEndcapModuleAlgo::execute(DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalEndcapModuleAlgo...";
#endif
  if (modType == 0)
    constructInsideModule0(parent(), cpv);
  else
    constructInsideModule(parent(), cpv);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalEndcapModuleAlgo construction ...";
#endif
}

void DDHCalEndcapModuleAlgo::constructInsideModule0(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: \t\tInside module0";
#endif
  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));
  DDName matName(DDSplit(absorberMat).first, DDSplit(absorberMat).second);
  DDMaterial matabsorbr(matName);
  DDName plasName(DDSplit(plasticMat).first, DDSplit(plasticMat).second);
  DDMaterial matplastic(plasName);

  int layer = layerNumber[0];
  int layer0 = layerNumber[1];
  std::string name;
  DDSolid solid;
  DDLogicalPart glog, plog;
  for (unsigned int iphi = 0; iphi < phiName.size(); iphi++) {
    DDHCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer0(iphi);
    name = idName + modName + layerName[0] + phiName[iphi];
    solid = DDSolidFactory::trap(DDName(name, idNameSpace),
                                 0.5 * layerThick,
                                 0,
                                 0,
                                 parm.yh1,
                                 parm.bl1,
                                 parm.tl1,
                                 parm.alp,
                                 parm.yh2,
                                 parm.bl1,
                                 parm.tl2,
                                 parm.alp);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << plasName
                                 << " of dimensions " << 0.5 * layerThick << ", 0, 0, " << parm.yh1 << ", " << parm.bl1
                                 << ", " << parm.tl1 << ", " << convertRadToDeg(parm.alp) << ", " << parm.yh2 << ", "
                                 << parm.bl2 << ", " << parm.tl2 << ", " << convertRadToDeg(parm.alp);
#endif
    glog = DDLogicalPart(solid.ddname(), matplastic, solid);

    DDTranslation r1(parm.xpos, parm.ypos, parm.zpos);
    cpv.position(glog, module, idOffset + layer + 1, r1, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << idOffset + layer + 1
                                 << " positioned in " << module.name() << " at " << r1 << " with " << rot;
#endif
    //Now construct the layer of scintillator inside this
    int copyNo = layer0 * 10 + layerType;
    name = modName + layerName[0] + phiName[iphi];
    constructScintLayer(glog, scintThick, parm, name, copyNo, cpv);
  }

  //Now the absorber layer
  double zi = zMinBlock + layerThick;
  double zo = zi + 0.5 * dzStep;
  double rinF, routF, rinB, routB;
  if (modNumber == 0) {
    rinF = zi * slopeTopF;
    routF = (zi - z1Beam) * slopeTop;
    rinB = zo * slopeTopF;
    routB = (zo - z1Beam) * slopeTop;
  } else if (modNumber > 0) {
    rinF = zi * slopeBot;
    routF = zi * slopeTopF;
    rinB = zo * slopeBot;
    routB = zo * slopeTopF;
  } else {
    rinF = zi * slopeBot;
    routF = (zi - z1Beam) * slopeTop;
    rinB = zo * slopeBot;
    routB = (zo - z1Beam) * slopeTop;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Front " << zi << ", " << rinF << ", " << routF << " Back "
                               << zo << ", " << rinB << ", " << routB;
#endif
  DDHCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer(0, rinF, routF, rinB, routB, zi, zo);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Trim " << tolAbs << " Param " << parm.yh1 << ", " << parm.bl1
                               << ", " << parm.tl1 << ", " << parm.yh2 << ", " << parm.bl2 << ", " << parm.tl2;
#endif
  parm.bl1 -= tolAbs;
  parm.tl1 -= tolAbs;
  parm.bl2 -= tolAbs;
  parm.tl2 -= tolAbs;

  name = idName + modName + layerName[0] + "Absorber";
  solid = DDSolidFactory::trap(DDName(name, idNameSpace),
                               0.5 * moduleThick,
                               parm.theta,
                               parm.phi,
                               parm.yh1,
                               parm.bl1,
                               parm.tl1,
                               parm.alp,
                               parm.yh2,
                               parm.bl2,
                               parm.tl2,
                               parm.alp);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << matName
                               << " of dimensions " << 0.5 * moduleThick << ", " << convertRadToDeg(parm.theta) << ", "
                               << convertRadToDeg(parm.phi) << ", " << parm.yh1 << ", " << parm.bl1 << ", " << parm.tl1
                               << ", " << convertRadToDeg(parm.alp) << ", " << parm.yh2 << ", " << parm.bl2 << ", "
                               << parm.tl2 << ", " << convertRadToDeg(parm.alp);
#endif
  glog = DDLogicalPart(solid.ddname(), matabsorbr, solid);

  DDTranslation r2(parm.xpos, parm.ypos, parm.zpos);
  cpv.position(glog, module, 1, r2, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number 1 positioned in "
                               << module.name() << " at " << r2 << " with " << rot;
#endif
}

void DDHCalEndcapModuleAlgo::constructInsideModule(const DDLogicalPart& module, DDCompactView& cpv) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: \t\tInside module";
#endif
  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));
  DDName matName(DDSplit(genMaterial).first, DDSplit(genMaterial).second);
  DDMaterial matter(matName);
  DDName plasName(DDSplit(plasticMat).first, DDSplit(plasticMat).second);
  DDMaterial matplastic(plasName);

  double alpha = (1._pi) / sectors;
  double zi = zMinBlock;

  for (unsigned int i = 0; i < layerName.size(); i++) {
    std::string name;
    DDSolid solid;
    DDLogicalPart glog, plog;
    int layer = layerNumber[i];
    double zo = zi + 0.5 * dzStep;

    for (unsigned int iphi = 0; iphi < phiName.size(); iphi++) {
      double ziAir = zo - moduleThick;
      double rinF, rinB;
      if (modNumber == 0) {
        rinF = ziAir * slopeTopF;
        rinB = zo * slopeTopF;
      } else {
        rinF = ziAir * slopeBot;
        rinB = zo * slopeBot;
      }
      double routF = getRout(ziAir);
      double routB = getRout(zo);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: Layer " << i << " Phi " << iphi << " Front " << ziAir
                                   << ", " << rinF << ", " << routF << " Back " << zo << ", " << rinB << ", " << routB;
#endif
      DDHCalEndcapModuleAlgo::HcalEndcapPar parm = parameterLayer(iphi, rinF, routF, rinB, routB, ziAir, zo);

      name = idName + modName + layerName[i] + phiName[iphi] + "Air";
      solid = DDSolidFactory::trap(DDName(name, idNameSpace),
                                   0.5 * moduleThick,
                                   parm.theta,
                                   parm.phi,
                                   parm.yh1,
                                   parm.bl1,
                                   parm.tl1,
                                   parm.alp,
                                   parm.yh2,
                                   parm.bl2,
                                   parm.tl2,
                                   parm.alp);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << matName
                                   << " of dimensions " << 0.5 * moduleThick << ", " << convertRadToDeg(parm.theta)
                                   << ", " << convertRadToDeg(parm.phi) << ", " << parm.yh1 << ", " << parm.bl1 << ", "
                                   << parm.tl1 << ", " << convertRadToDeg(parm.alp) << ", " << parm.yh2 << ", "
                                   << parm.bl2 << ", " << parm.tl2 << ", " << convertRadToDeg(parm.alp);
#endif
      glog = DDLogicalPart(solid.ddname(), matter, solid);

      DDTranslation r1(parm.xpos, parm.ypos, parm.zpos);
      cpv.position(glog, module, layer + 1, r1, rot);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << layer + 1
                                   << " positioned in " << module.name() << " at " << r1 << " with " << rot;
#endif
      //Now the plastic with scintillators
      parm.yh1 = 0.5 * (routF - rinB) - getTrim(iphi);
      parm.bl1 = 0.5 * rinB * tan(alpha) - getTrim(iphi);
      parm.tl1 = 0.5 * routF * tan(alpha) - getTrim(iphi);
      name = idName + modName + layerName[i] + phiName[iphi];
      solid = DDSolidFactory::trap(DDName(name, idNameSpace),
                                   0.5 * layerThick,
                                   0,
                                   0,
                                   parm.yh1,
                                   parm.bl1,
                                   parm.tl1,
                                   parm.alp,
                                   parm.yh1,
                                   parm.bl1,
                                   parm.tl1,
                                   parm.alp);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << plasName
                                   << " of dimensions " << 0.5 * layerThick << ", 0, 0, " << parm.yh1 << ", "
                                   << parm.bl1 << ", " << parm.tl1 << ", " << convertRadToDeg(parm.alp) << ", "
                                   << parm.yh1 << ", " << parm.bl1 << ", " << parm.tl1 << ", "
                                   << convertRadToDeg(parm.alp);
#endif
      plog = DDLogicalPart(solid.ddname(), matplastic, solid);

      double ypos = 0.5 * (routF + rinB) - parm.xpos;
      DDTranslation r2(0., ypos, 0.);
      cpv.position(plog, glog, idOffset + layer + 1, r2, DDRotation());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << plog.name() << " number " << idOffset + layer + 1
                                   << " positioned in " << glog.name() << " at " << r2 << " with no rotation";
#endif
      //Constructin the scintillators inside
      int copyNo = layer * 10 + layerType;
      name = modName + layerName[i] + phiName[iphi];
      constructScintLayer(plog, scintThick, parm, name, copyNo, cpv);
      zo += 0.5 * dzStep;
    }  // End of loop over phi indices
    zi = zo - 0.5 * dzStep;
  }  // End of loop on layers
}

DDHCalEndcapModuleAlgo::HcalEndcapPar DDHCalEndcapModuleAlgo::parameterLayer0(unsigned int iphi) {
  DDHCalEndcapModuleAlgo::HcalEndcapPar parm;
  //Given module and layer number compute parameters of trapezoid
  //and positioning parameters
  double alpha = (1._pi) / sectors;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Alpha " << convertRadToDeg(alpha);
#endif
  double zi, zo;
  if (iphi == 0) {
    zi = zMinBlock;
    zo = zi + layerThick;
  } else {
    zo = zMaxBlock;
    zi = zo - layerThick;
  }
  double rin, rout;
  if (modNumber == 0) {
    rin = zo * slopeTopF;
    rout = (zi - z1Beam) * slopeTop;
  } else if (modNumber > 0) {
    rin = zo * slopeBot;
    rout = zi * slopeTopF;
  } else {
    rin = zo * slopeBot;
    rout = (zi - z1Beam) * slopeTop;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "ModNumber " << modNumber << " " << zi << " " << zo << " " << slopeTopF << " "
                               << slopeTop << " " << slopeBot << " " << rin << " " << rout << " " << getTrim(iphi);
#endif
  double yh = 0.5 * (rout - rin);
  double bl = 0.5 * rin * tan(alpha);
  double tl = 0.5 * rout * tan(alpha);
  parm.xpos = 0.5 * (rin + rout);
  parm.ypos = 0.5 * (bl + tl);
  parm.zpos = 0.5 * (zi + zo);
  parm.yh1 = parm.yh2 = yh - getTrim(iphi);
  parm.bl1 = parm.bl2 = bl - getTrim(iphi);
  parm.tl1 = parm.tl2 = tl - getTrim(iphi);
  parm.alp = atan(0.5 * tan(alpha));
  if (iphi == 0) {
    parm.ypos = -parm.ypos;
  } else {
    parm.alp = -parm.alp;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Output Dimensions " << parm.yh1 << " " << parm.bl1 << " " << parm.tl1 << " "
                               << convertRadToDeg(parm.alp) << " Position " << parm.xpos << " " << parm.ypos << " "
                               << parm.zpos;
#endif
  return parm;
}

DDHCalEndcapModuleAlgo::HcalEndcapPar DDHCalEndcapModuleAlgo::parameterLayer(
    unsigned int iphi, double rinF, double routF, double rinB, double routB, double zi, double zo) {
  DDHCalEndcapModuleAlgo::HcalEndcapPar parm;
  //Given rin, rout compute parameters of the trapezoid and
  //position of the trapezoid for a standrd layer
  double alpha = (1._pi) / sectors;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Input " << iphi << " Front " << rinF << " " << routF << " " << zi << " Back " << rinB
                               << " " << routB << " " << zo << " Alpha " << convertRadToDeg(alpha);
#endif
  parm.yh1 = 0.5 * (routF - rinB);
  parm.bl1 = 0.5 * rinB * tan(alpha);
  parm.tl1 = 0.5 * routF * tan(alpha);
  parm.yh2 = 0.5 * (routF - rinB);
  parm.bl2 = 0.5 * rinB * tan(alpha);
  parm.tl2 = 0.5 * routF * tan(alpha);
  double dx = 0.25 * (parm.bl2 + parm.tl2 - parm.bl1 - parm.tl1);
  double dy = 0.5 * (rinB + routF - rinB - routF);
  parm.xpos = 0.25 * (rinB + routF + rinB + routF);
  parm.ypos = 0.25 * (parm.bl2 + parm.tl2 + parm.bl1 + parm.tl1);
  parm.zpos = 0.5 * (zi + zo);
  parm.alp = atan(0.5 * tan(alpha));
  if (iphi == 0) {
    parm.ypos = -parm.ypos;
  } else {
    parm.alp = -parm.alp;
    dx = -dx;
  }
  double r = sqrt(dx * dx + dy * dy);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "dx|dy|r " << dx << ":" << dy << ":" << r;
#endif
  if (r > 1.0e-8) {
    parm.theta = atan(r / (zo - zi));
    parm.phi = atan2(dy, dx);
  } else {
    parm.theta = parm.phi = 0;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "Output Dimensions " << parm.yh1 << " " << parm.bl1 << " " << parm.tl1 << " "
                               << parm.yh2 << " " << parm.bl2 << " " << parm.tl2 << " " << convertRadToDeg(parm.alp)
                               << " " << convertRadToDeg(parm.theta) << " " << convertRadToDeg(parm.phi) << " Position "
                               << parm.xpos << " " << parm.ypos << " " << parm.zpos;
#endif
  return parm;
}

void DDHCalEndcapModuleAlgo::constructScintLayer(const DDLogicalPart& detector,
                                                 double dz,
                                                 DDHCalEndcapModuleAlgo::HcalEndcapPar parm,
                                                 const std::string& nm,
                                                 int id,
                                                 DDCompactView& cpv) {
  DDName matname(DDSplit(scintMat).first, DDSplit(scintMat).second);
  DDMaterial matter(matname);
  std::string name = idName + "Scintillator" + nm;

  DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace),
                                       0.5 * dz,
                                       0,
                                       0,
                                       parm.yh1,
                                       parm.bl1,
                                       parm.tl1,
                                       parm.alp,
                                       parm.yh1,
                                       parm.bl1,
                                       parm.tl1,
                                       parm.alp);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << solid.name() << " Trap made of " << scintMat
                               << " of dimensions " << 0.5 * dz << ", 0, 0, " << parm.yh1 << ", " << parm.bl1 << ", "
                               << parm.tl1 << ", " << convertRadToDeg(parm.alp) << ", " << parm.yh1 << ", " << parm.bl1
                               << ", " << parm.tl1 << ", " << convertRadToDeg(parm.alp);
#endif
  DDLogicalPart glog(solid.ddname(), matter, solid);

  cpv.position(glog, detector, id, DDTranslation(0, 0, 0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalEndcapModuleAlgo: " << glog.name() << " number " << id << " positioned in "
                               << detector.name() << " at (0,0,0) with no rotation";
#endif
}

double DDHCalEndcapModuleAlgo::getTrim(unsigned int j) const {
  if (j == 0)
    return trimLeft;
  else
    return trimRight;
}

double DDHCalEndcapModuleAlgo::getRout(double z) const {
  double r = (modNumber >= 0) ? ((z - z1Beam) * slopeTop) : z * slopeTopF;
  if (z > ziDip) {
    if (r > rMaxBack)
      r = rMaxBack;
  } else {
    if (r > rMaxFront)
      r = rMaxFront;
  }
  return r;
}
