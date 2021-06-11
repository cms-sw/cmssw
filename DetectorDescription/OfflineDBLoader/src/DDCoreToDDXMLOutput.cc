#include <DetectorDescription/Core/interface/DDMaterial.h>
#include <DetectorDescription/Core/interface/DDPartSelection.h>
#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>
#include <DetectorDescription/Core/interface/DDSpecifics.h>
#include <DetectorDescription/OfflineDBLoader/interface/DDCoreToDDXMLOutput.h>
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDPosData.h"
#include "DetectorDescription/Core/interface/DDTransform.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDValuePair.h"

#include "DetectorDescription/DDCMS/interface/DDSolidShapes.h"
#include "DetectorDescription/DDCMS/interface/DDNamespace.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/Rounding.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Rotation3D.h"

#include "DD4hep/Filter.h"
#include "DD4hep/Shapes.h"

#include "TGeoMedium.h"

#include <cstddef>
#include <iomanip>
#include <vector>

using namespace geant_units::operators;

template <class NumType>
static inline constexpr NumType convertGPerCcToMgPerCc(NumType gPerCc)  // g/cm^3 -> mg/cm^3
{
  return (gPerCc * 1000.);
}

namespace cms::rotation_utils {
  /* For debugging 
  static double determinant(const dd4hep::Rotation3D &rot) {
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;
    rot.GetComponents(xx, xy, xz, yx, yy, yz, zx, zy, zz); 
    double term1 = xx * (yy * zz - yz * zy);
    double term2 = yx * (xy * zz - xz * zy);
    double term3 = zx * (xy * yz - yy * xz);
    return (term1 - term2 + term3);
  }
  */

  static const std::string identityHash("1.0000000.0000000.0000000.0000001.0000000.0000000.0000000.0000001.000000");

  static void addRotWithNewName(cms::DDNamespace& ns, std::string& name, const dd4hep::Rotation3D& rot) {
    const dd4hep::Rotation3D& rot2 = rot;
    name = name + "_DdNoNa";  // Name used by old DD to indicate an unnamed rotation
    ns.addRotation(name, rot2);
  }

  static void addRotWithNewName(cms::DDNamespace& ns, std::string& name, const Double_t* rot) {
    using namespace cms_rounding;
    dd4hep::Rotation3D rot2(roundIfNear0(rot[0]),
                            roundIfNear0(rot[1]),
                            roundIfNear0(rot[2]),
                            roundIfNear0(rot[3]),
                            roundIfNear0(rot[4]),
                            roundIfNear0(rot[5]),
                            roundIfNear0(rot[6]),
                            roundIfNear0(rot[7]),
                            roundIfNear0(rot[8]));
    addRotWithNewName(ns, name, rot2);
  }

  template <typename T>
  static const std::string& rotName(const T& rot, const cms::DDParsingContext& context) {
    std::string hashVal = rotHash(rot);
    auto rotNameIter = context.rotRevMap.find(hashVal);
    if (rotNameIter != context.rotRevMap.end()) {
      return (rotNameIter->second);
    }
    static const std::string nullStr{"NULL"};
    return (nullStr);
  }

}  // namespace cms::rotation_utils

std::string DDCoreToDDXMLOutput::trimShapeName(const std::string& solidName) {
  size_t trimPt = solidName.find("_shape_0x");
  if (trimPt != std::string::npos)
    return (solidName.substr(0, trimPt));
  return (solidName);
}

void DDCoreToDDXMLOutput::solid(const dd4hep::Solid& solid, const cms::DDParsingContext& context, std::ostream& xos) {
  cms::DDSolidShape shape = cms::dd::value(cms::DDSolidShapeMap, std::string(solid.title()));
  switch (shape) {
    case cms::DDSolidShape::ddunion:
    case cms::DDSolidShape::ddsubtraction:
    case cms::DDSolidShape::ddintersection: {
      dd4hep::BooleanSolid rs(solid);
      if (shape == cms::DDSolidShape::ddunion) {
        xos << "<UnionSolid ";
      } else if (shape == cms::DDSolidShape::ddsubtraction) {
        xos << "<SubtractionSolid ";
      } else if (shape == cms::DDSolidShape::ddintersection) {
        xos << "<IntersectionSolid ";
      }
      xos << "name=\"" << trimShapeName(solid.name()) << "\">" << std::endl;
      xos << "<rSolid name=\"" << trimShapeName(rs.leftShape().name()) << "\"/>" << std::endl;
      xos << "<rSolid name=\"" << trimShapeName(rs.rightShape().name()) << "\"/>" << std::endl;
      const Double_t* trans = rs.rightMatrix()->GetTranslation();
      xos << "<Translation x=\"" << trans[0] << "*mm\"";
      xos << " y=\"" << trans[1] << "*mm\"";
      xos << " z=\"" << trans[2] << "*mm\"";
      xos << "/>" << std::endl;
      std::string rotNameStr = cms::rotation_utils::rotName(rs.rightMatrix()->GetRotationMatrix(), context);
      xos << "<rRotation name=\"" << rotNameStr << "\"/>" << std::endl;
      if (shape == cms::DDSolidShape::ddunion) {
        xos << "</UnionSolid>" << std::endl;
      } else if (shape == cms::DDSolidShape::ddsubtraction) {
        xos << "</SubtractionSolid>" << std::endl;
      } else if (shape == cms::DDSolidShape::ddintersection) {
        xos << "</IntersectionSolid>" << std::endl;
      }
      break;
    }
    case cms::DDSolidShape::ddbox: {
      dd4hep::Box rs(solid);
      xos << "<Box name=\"" << trimShapeName(rs.name()) << "\""
          << " dx=\"" << rs.x() << "*mm\""
          << " dy=\"" << rs.y() << "*mm\""
          << " dz=\"" << rs.z() << "*mm\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtubs: {
      dd4hep::Tube rs(solid);
      double startPhi = convertRadToDeg(rs.startPhi());
      if (startPhi > 180. && startPhi <= 360.)
        startPhi -= 360.;
      // Convert large positive angles to small negative ones
      // to match how they are usually defined

      xos << "<Tubs name=\"" << trimShapeName(rs.name()) << "\""
          << " rMin=\"" << rs.rMin() << "*mm\""
          << " rMax=\"" << rs.rMax() << "*mm\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " startPhi=\"" << startPhi << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.endPhi() - rs.startPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtrd1: {
      dd4hep::Trd1 rs(solid);
      xos << "<Trd1 name=\"" << trimShapeName(rs.name()) << "\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " dy1=\"" << rs.dY() << "*mm\""
          << " dy2=\"" << rs.dY() << "*mm\""
          << " dx1=\"" << rs.dX1() << "*mm\""
          << " dx2=\"" << rs.dX2() << "*mm\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtrd2: {
      dd4hep::Trd2 rs(solid);
      xos << "<Trd1 name=\"" << trimShapeName(rs.name()) << "\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " dy1=\"" << rs.dY1() << "*mm\""
          << " dy2=\"" << rs.dY2() << "*mm\""
          << " dx1=\"" << rs.dX1() << "*mm\""
          << " dx2=\"" << rs.dX2() << "*mm\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtrap: {
      dd4hep::Trap rs(solid);
      xos << "<Trapezoid name=\"" << trimShapeName(rs.name()) << "\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " theta=\"" << convertRadToDeg(rs.theta()) << "*deg\""
          << " phi=\"" << convertRadToDeg(rs.phi()) << "*deg\""
          << " h1=\"" << rs.high1() << "*mm\""
          << " bl1=\"" << rs.bottomLow1() << "*mm\""
          << " tl1=\"" << rs.topLow1() << "*mm\""
          << " alp1=\"" << convertRadToDeg(rs.alpha1()) << "*deg\""
          << " h2=\"" << rs.high2() << "*mm\""
          << " bl2=\"" << rs.bottomLow2() << "*mm\""
          << " tl2=\"" << rs.topLow2() << "*mm\""
          << " alp2=\"" << convertRadToDeg(rs.alpha2()) << "*deg\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddcons: {
      dd4hep::ConeSegment rs(solid);
      double startPhi = convertRadToDeg(rs.startPhi());
      if (startPhi > 180. && startPhi <= 360.)
        startPhi -= 360.;
      // Convert large positive angles to small negative ones
      // to match how they are usually defined
      //
      xos << "<Cone name=\"" << trimShapeName(rs.name()) << "\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " rMin1=\"" << rs.rMin1() << "*mm\""
          << " rMax1=\"" << rs.rMax1() << "*mm\""
          << " rMin2=\"" << rs.rMin2() << "*mm\""
          << " rMax2=\"" << rs.rMax2() << "*mm\""
          << " startPhi=\"" << startPhi << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.endPhi() - rs.startPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddpolycone: {
      //   <Polycone name="OCMS" startPhi="0*deg" deltaPhi="360*deg" >
      //    <ZSection z="-[CMSZ1]"  rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[HallR]" />
      //    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[HallR]" />
      //    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="[CMSZ1]"   rMin="[Rmin]"  rMax="[CMSR2]" />
      dd4hep::Polycone rs(solid);
      xos << "<Polycone name=\"" << trimShapeName(rs.name()) << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zPlaneZ());
      const std::vector<double>& rMinV(rs.zPlaneRmin());
      const std::vector<double>& rMaxV(rs.zPlaneRmax());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<ZSection z=\"" << zV[i] << "*mm\""
            << " rMin=\"" << rMinV[i] << "*mm\""
            << " rMax=\"" << rMaxV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polycone>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddpolyhedra: {
      dd4hep::Polyhedra rs(solid);
      xos << "<Polyhedra name=\"" << trimShapeName(rs.name()) << "\""
          << " numSide=\"" << rs.numEdges() << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zPlaneZ());
      const std::vector<double>& rMinV(rs.zPlaneRmin());
      const std::vector<double>& rMaxV(rs.zPlaneRmax());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<ZSection z=\"" << zV[i] << "*mm\""
            << " rMin=\"" << rMinV[i] << "*mm\""
            << " rMax=\"" << rMaxV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polyhedra>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtrunctubs: {
      dd4hep::TruncatedTube rs(solid);
      xos << "<TruncTubs name=\"" << trimShapeName(rs.name()) << "\""
          << " zHalf=\"" << rs.dZ() << "*mm\""
          << " rMin=\"" << rs.rMin() << "*mm\""
          << " rMax=\"" << rs.rMax() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\""
          << " cutAtStart=\"" << rs.cutAtStart() << "*mm\""
          << " cutAtDelta=\"" << rs.cutAtDelta() << "*mm\""
          << " cutInside=\"" << (rs.cutInside() ? "true" : "false") << "\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddtorus: {
      dd4hep::Torus rs(solid);
      xos << "<Torus name=\"" << trimShapeName(rs.name()) << "\""
          << " innerRadius=\"" << rs.rMin() << "*mm\""
          << " outerRadius=\"" << rs.rMax() << "*mm\""
          << " torusRadius=\"" << rs.r() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddellipticaltube: {
      dd4hep::EllipticalTube rs(solid);
      xos << "<EllipticalTube name=\"" << trimShapeName(rs.name()) << "\""
          << " xSemiAxis=\"" << rs.a() << "*mm\""
          << " ySemiAxis=\"" << rs.b() << "*mm\""
          << " zHeight=\"" << rs.dZ() << "*mm\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddcuttubs: {
      dd4hep::CutTube rs(solid);
      const std::vector<double>& pLowNorm(rs.lowNormal());
      const std::vector<double>& pHighNorm(rs.highNormal());

      xos << "<CutTubs name=\"" << trimShapeName(solid.name()) << "\""
          << " dz=\"" << rs.dZ() << "*mm\""
          << " rMin=\"" << rs.rMin() << "*mm\""
          << " rMax=\"" << rs.rMax() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.endPhi() - rs.startPhi()) << "*deg\""
          << " lx=\"" << pLowNorm[0] << "\""
          << " ly=\"" << pLowNorm[1] << "\""
          << " lz=\"" << pLowNorm[2] << "\""
          << " tx=\"" << pHighNorm[0] << "\""
          << " ty=\"" << pHighNorm[1] << "\""
          << " tz=\"" << pHighNorm[2] << "\"/>" << std::endl;
      break;
    }
    case cms::DDSolidShape::ddextrudedpolygon: {
      dd4hep::ExtrudedPolygon rs(solid);
      std::vector<double> x = rs.x();
      std::vector<double> y = rs.y();
      std::vector<double> z = rs.z();
      std::vector<double> zx = rs.zx();
      std::vector<double> zy = rs.zy();
      std::vector<double> zs = rs.zscale();

      xos << "<ExtrudedPolygon name=\"" << trimShapeName(rs.name()) << "\"";
      for (unsigned int i = 0; i < x.size(); ++i)
        xos << " <XYPoint x=\"" << x[i] << "*mm\" y=\"" << y[i] << "*mm\"/>\n";
      for (unsigned int k = 0; k < z.size(); ++k)
        xos << " <ZXYSection z=\"" << z[k] << "*mm\" x=\"" << zx[k] << "*mm\" y=\"" << zy[k] << "*mm scale=" << zs[k]
            << "*mm\"/>\n";
      xos << "</ExtrudedPolygon>\n";
      break;
    }
    case cms::DDSolidShape::dd_not_init:
    default:
      throw cms::Exception("DDException")
          << "DDCoreToDDXMLOutput::solid " << solid.name() << ", shape ID = " << static_cast<int>(shape)
          << ", solid title = " << solid.title();
      break;
  }
}

void DDCoreToDDXMLOutput::solid(const DDSolid& solid, std::ostream& xos) {
  switch (solid.shape()) {
    case DDSolidShape::ddunion:
    case DDSolidShape::ddsubtraction:
    case DDSolidShape::ddintersection: {
      DDBooleanSolid rs(solid);
      if (solid.shape() == DDSolidShape::ddunion) {
        xos << "<UnionSolid ";
      } else if (solid.shape() == DDSolidShape::ddsubtraction) {
        xos << "<SubtractionSolid ";
      } else if (solid.shape() == DDSolidShape::ddintersection) {
        xos << "<IntersectionSolid ";
      }
      xos << "name=\"" << rs.toString() << "\">" << std::endl;
      // if translation is == identity there are no parameters.
      // if there is no rotation the name will be ":"
      xos << "<rSolid name=\"" << rs.solidA().toString() << "\"/>" << std::endl;
      xos << "<rSolid name=\"" << rs.solidB().toString() << "\"/>" << std::endl;
      xos << "<Translation x=\"" << rs.translation().X() << "*mm\"";
      xos << " y=\"" << rs.translation().Y() << "*mm\"";
      xos << " z=\"" << rs.translation().Z() << "*mm\"";
      xos << "/>" << std::endl;
      std::string rotName = rs.rotation().toString();
      if (rotName == ":") {
        rotName = "gen:ID";
      }
      xos << "<rRotation name=\"" << rs.rotation().toString() << "\"/>" << std::endl;
      if (solid.shape() == DDSolidShape::ddunion) {
        xos << "</UnionSolid>" << std::endl;
      } else if (solid.shape() == DDSolidShape::ddsubtraction) {
        xos << "</SubtractionSolid>" << std::endl;
      } else if (solid.shape() == DDSolidShape::ddintersection) {
        xos << "</IntersectionSolid>" << std::endl;
      }
      break;
    }
    case DDSolidShape::ddbox: {
      //    <Box name="box1" dx="10*cm" dy="10*cm" dz="10*cm"/>
      DDBox rs(solid);
      xos << "<Box name=\"" << rs.toString() << "\""  //<< rs.toString() << "\"" //
          << " dx=\"" << rs.halfX() << "*mm\""
          << " dy=\"" << rs.halfY() << "*mm\""
          << " dz=\"" << rs.halfZ() << "*mm\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddtubs: {
      //      <Tubs name="TrackerSupportTubeNomex"         rMin="[SupportTubeR1]+[Tol]"
      //            rMax="[SupportTubeR2]-[Tol]"           dz="[SupportTubeL]"
      //            startPhi="0*deg"                       deltaPhi="360*deg"/>
      DDTubs rs(solid);
      xos << "<Tubs name=\"" << rs.toString() << "\""
          << " rMin=\"" << rs.rIn() << "*mm\""
          << " rMax=\"" << rs.rOut() << "*mm\""
          << " dz=\"" << rs.zhalf() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddtrap: {
      //    <Trapezoid name="UpL_CSC_for_TotemT1_Plane_2_5_7" dz="[PCB_Epoxy_Thick_3P]/2."
      //      alp1="-[Up_Signal_Side_alpL_3P]" alp2="-[Up_Signal_Side_alpL_3P]"
      //     bl1="[Max_Base_Signal_SideL_3P]/2." tl1="[Up_Min_Base_Signal_SideL_3P]/2." h1="[Up_Height_Signal_SideL_3P]/2."
      //     h2="[Up_Height_Signal_SideL_3P]/2." bl2="[Max_Base_Signal_SideL_3P]/2." tl2="[Up_Min_Base_Signal_SideL_3P]/2."/>
      DDTrap rs(solid);
      xos << "<Trapezoid name=\"" << rs.toString() << "\""
          << " dz=\"" << rs.halfZ() << "*mm\""
          << " theta=\"" << convertRadToDeg(rs.theta()) << "*deg\""
          << " phi=\"" << convertRadToDeg(rs.phi()) << "*deg\""
          << " h1=\"" << rs.y1() << "*mm\""
          << " bl1=\"" << rs.x1() << "*mm\""
          << " tl1=\"" << rs.x2() << "*mm\""
          << " alp1=\"" << convertRadToDeg(rs.alpha1()) << "*deg\""
          << " h2=\"" << rs.y2() << "*mm\""
          << " bl2=\"" << rs.x3() << "*mm\""
          << " tl2=\"" << rs.x4() << "*mm\""
          << " alp2=\"" << convertRadToDeg(rs.alpha2()) << "*deg\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddcons: {
      DDCons rs(solid);
      xos << "<Cone name=\"" << rs.toString() << "\""
          << " dz=\"" << rs.zhalf() << "*mm\""
          << " rMin1=\"" << rs.rInMinusZ() << "*mm\""
          << " rMax1=\"" << rs.rOutMinusZ() << "*mm\""
          << " rMin2=\"" << rs.rInPlusZ() << "*mm\""
          << " rMax2=\"" << rs.rOutPlusZ() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.phiFrom()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddpolycone_rz: {
      DDPolycone rs(solid);
      xos << "<Polycone name=\"" << rs.toString() << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zVec());
      const std::vector<double>& rV(rs.rVec());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<RZPoint r=\"" << rV[i] << "*mm\""
            << " z=\"" << zV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polycone>" << std::endl;
      break;
    }
    case DDSolidShape::ddpolyhedra_rz: {
      DDPolyhedra rs(solid);
      xos << "<Polyhedra name=\"" << rs.toString() << "\""
          << " numSide=\"" << rs.sides() << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zVec());
      const std::vector<double>& rV(rs.rVec());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<RZPoint r=\"" << rV[i] << "*mm\""
            << " z=\"" << zV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polyhedra>" << std::endl;
      break;
    }
    case DDSolidShape::ddpolycone_rrz: {
      //   <Polycone name="OCMS" startPhi="0*deg" deltaPhi="360*deg" >
      //    <ZSection z="-[CMSZ1]"  rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="-[HallZ]"  rMin="[Rmin]"  rMax="[HallR]" />
      //    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[HallR]" />
      //    <ZSection z="[HallZ]"   rMin="[Rmin]"  rMax="[CMSR2]" />
      //    <ZSection z="[CMSZ1]"   rMin="[Rmin]"  rMax="[CMSR2]" />
      DDPolycone rs(solid);
      xos << "<Polycone name=\"" << rs.toString() << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zVec());
      const std::vector<double>& rMinV(rs.rMinVec());
      const std::vector<double>& rMaxV(rs.rMaxVec());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<ZSection z=\"" << zV[i] << "*mm\""
            << " rMin=\"" << rMinV[i] << "*mm\""
            << " rMax=\"" << rMaxV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polycone>" << std::endl;
      break;
    }
    case DDSolidShape::ddpolyhedra_rrz: {
      DDPolyhedra rs(solid);
      xos << "<Polyhedra name=\"" << rs.toString() << "\""
          << " numSide=\"" << rs.sides() << "\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\">" << std::endl;
      const std::vector<double>& zV(rs.zVec());
      const std::vector<double>& rMinV(rs.rMinVec());
      const std::vector<double>& rMaxV(rs.rMaxVec());
      for (size_t i = 0; i < zV.size(); ++i) {
        xos << "<ZSection z=\"" << zV[i] << "*mm\""
            << " rMin=\"" << rMinV[i] << "*mm\""
            << " rMax=\"" << rMaxV[i] << "*mm\"/>" << std::endl;
      }
      xos << "</Polyhedra>" << std::endl;
      break;
    }
    case DDSolidShape::ddpseudotrap: {
      // <PseudoTrap name="YE3_b" dx1="0.395967*m" dx2="1.86356*m" dy1="0.130*m" dy2="0.130*m" dz="2.73857*m" radius="-1.5300*m" atMinusZ="true"/>
      DDPseudoTrap rs(solid);
      xos << "<PseudoTrap name=\"" << rs.toString() << "\""
          << " dx1=\"" << rs.x1() << "*mm\""
          << " dx2=\"" << rs.x2() << "*mm\""
          << " dy1=\"" << rs.y1() << "*mm\""
          << " dy2=\"" << rs.y2() << "*mm\""
          << " dz=\"" << rs.halfZ() << "*mm\""
          << " radius=\"" << rs.radius() << "*mm\""
          << " atMinusZ=\"" << (rs.atMinusZ() ? "true" : "false") << "\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddtrunctubs: {
      // <TruncTubs name="trunctubs1" zHalf="50*cm" rMin="20*cm" rMax="40*cm"
      //                              startPhi="0*deg" deltaPhi="90*deg"
      //                              cutAtStart="25*cm" cutAtDelta="35*cm"/>
      DDTruncTubs rs(solid);
      xos << "<TruncTubs name=\"" << rs.toString() << "\""
          << " zHalf=\"" << rs.zHalf() << "*mm\""
          << " rMin=\"" << rs.rIn() << "*mm\""
          << " rMax=\"" << rs.rOut() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\""
          << " cutAtStart=\"" << rs.cutAtStart() << "*mm\""
          << " cutAtDelta=\"" << rs.cutAtDelta() << "*mm\""
          << " cutInside=\"" << (rs.cutInside() ? "true" : "false") << "\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddshapeless: {
      DDShapelessSolid rs(solid);
      xos << "<ShapelessSolid name=\"" << rs.toString() << "\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddtorus: {
      // <Torus name="torus" innerRadius="7.5*cm" outerRadius="10*cm"
      //                     torusRadius="30*cm" startPhi="0*deg" deltaPhi="360*deg"/>
      DDTorus rs(solid);
      xos << "<Torus name=\"" << rs.toString() << "\""
          << " innerRadius=\"" << rs.rMin() << "*mm\""
          << " outerRadius=\"" << rs.rMax() << "*mm\""
          << " torusRadius=\"" << rs.rTorus() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddellipticaltube: {
      // <EllipticalTube name="CMSWall"  xSemiAxis="[cavernData:CMSWallEDX]"
      //                                 ySemiAxis="[cavernData:CMSWallEDY]"
      //                                 zHeight="[cms:HallZ]"/>
      DDEllipticalTube rs(solid);
      xos << "<EllipticalTube name=\"" << rs.toString() << "\""
          << " xSemiAxis=\"" << rs.xSemiAxis() << "*mm\""
          << " ySemiAxis=\"" << rs.ySemiAxis() << "*mm\""
          << " zHeight=\"" << rs.zHeight() << "*mm\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddcuttubs: {
      //      <Tubs name="TrackerSupportTubeNomex"         rMin="[SupportTubeR1]+[Tol]"
      //            rMax="[SupportTubeR2]-[Tol]"           dz="[SupportTubeL]"
      //            startPhi="0*deg"                       deltaPhi="360*deg"/>
      DDCutTubs rs(solid);
      const std::array<double, 3>& pLowNorm(rs.lowNorm());
      const std::array<double, 3>& pHighNorm(rs.highNorm());

      xos << "<CutTubs name=\"" << rs.toString() << "\""
          << " dz=\"" << rs.zhalf() << "*mm\""
          << " rMin=\"" << rs.rIn() << "*mm\""
          << " rMax=\"" << rs.rOut() << "*mm\""
          << " startPhi=\"" << convertRadToDeg(rs.startPhi()) << "*deg\""
          << " deltaPhi=\"" << convertRadToDeg(rs.deltaPhi()) << "*deg\""
          << " lx=\"" << pLowNorm[0] << "\""
          << " ly=\"" << pLowNorm[1] << "\""
          << " lz=\"" << pLowNorm[2] << "\""
          << " tx=\"" << pHighNorm[0] << "\""
          << " ty=\"" << pHighNorm[1] << "\""
          << " tz=\"" << pHighNorm[2] << "\"/>" << std::endl;
      break;
    }
    case DDSolidShape::ddextrudedpolygon: {
      DDExtrudedPolygon rs(solid);
      std::vector<double> x = rs.xVec();
      std::vector<double> y = rs.yVec();
      std::vector<double> z = rs.zVec();
      std::vector<double> zx = rs.zxVec();
      std::vector<double> zy = rs.zyVec();
      std::vector<double> zs = rs.zscaleVec();

      xos << "<ExtrudedPolygon name=\"" << rs.toString() << "\"";
      for (unsigned int i = 0; i < x.size(); ++i)
        xos << " <XYPoint x=\"" << x[i] << "*mm\" y=\"" << y[i] << "*mm\"/>\n";
      for (unsigned int k = 0; k < z.size(); ++k)
        xos << " <ZXYSection z=\"" << z[k] << "*mm\" x=\"" << zx[k] << "*mm\" y=\"" << zy[k] << "*mm scale=" << zs[k]
            << "*mm\"/>\n";
      xos << "</ExtrudedPolygon>\n";
      break;
    }
    case DDSolidShape::dd_not_init:
    default:
      throw cms::Exception("DDException")
          << "DDCoreToDDXMLOutput::solid(...) " << solid.name() << " either not inited or no such solid.";
      break;
  }
}

void DDCoreToDDXMLOutput::material(const DDMaterial& material, std::ostream& xos) {
  int noc = material.noOfConstituents();
  if (noc == 0) {
    xos << "<ElementaryMaterial name=\"" << material.toString() << "\""
        << " density=\"" << std::scientific << std::setprecision(5) << convertUnitsTo(1._mg_per_cm3, material.density())
        << "*mg/cm3\""
        << " atomicWeight=\"" << std::fixed << convertUnitsTo(1._g_per_mole, material.a()) << "*g/mole\""
        << std::setprecision(0) << std::fixed << " atomicNumber=\"" << material.z() << "\"/>" << std::endl;
  } else {
    xos << "<CompositeMaterial name=\"" << material.toString() << "\""
        << " density=\"" << std::scientific << std::setprecision(5) << convertUnitsTo(1._mg_per_cm3, material.density())
        << "*mg/cm3\""
        << " method=\"mixture by weight\">" << std::endl;

    int j = 0;
    for (; j < noc; ++j) {
      xos << "<MaterialFraction fraction=\"" << std::fixed << std::setprecision(9) << material.constituent(j).second
          << "\">" << std::endl;
      xos << "<rMaterial name=\"" << material.constituent(j).first.name() << "\"/>" << std::endl;
      xos << "</MaterialFraction>" << std::endl;
    }
    xos << "</CompositeMaterial>" << std::endl;
  }
}

void DDCoreToDDXMLOutput::material(
    const std::pair<std::string, std::pair<double, std::vector<cms::DDParsingContext::CompositeMaterial>>>& material,
    std::ostream& xos) {
  xos << "<CompositeMaterial name=\"" << material.first << "\""
      << " density=\"" << std::scientific << std::setprecision(5) << convertGPerCcToMgPerCc(material.second.first)
      << "*mg/cm3\""
      << " method=\"mixture by weight\">" << std::endl;

  auto compIter = material.second.second.begin();
  for (; compIter != material.second.second.end(); ++compIter) {
    xos << "<MaterialFraction fraction=\"" << std::fixed << std::setprecision(9) << compIter->fraction << "\">"
        << std::endl;
    xos << "<rMaterial name=\"" << compIter->name << "\"/>" << std::endl;
    xos << "</MaterialFraction>" << std::endl;
  }
  xos << "</CompositeMaterial>" << std::endl;
}

void DDCoreToDDXMLOutput::element(const TGeoMaterial* material, std::ostream& xos) {
  int noc = material->GetNelements();
  if (noc == 1) {
    TGeoElement* elem = material->GetElement();
    std::string nameLowerCase(elem->GetTitle());
    // Leave first letter capitalized
    for (size_t index = 1; index < nameLowerCase.size(); ++index) {
      nameLowerCase[index] = tolower(nameLowerCase[index]);
    }
    std::string trimName(dd4hep::dd::noNamespace(material->GetName()));

    // Element title contains element name in all uppercase.
    // Convert to lowercase and check that material name matches element name.
    // Hydrogen is used for vacuum materials. Phosphorus is called "Phosphor"
    // Boron is a special case because there are two isotopes defined: "Bor 10" and "Bor 11".
    if (trimName == nameLowerCase || nameLowerCase == "Hydrogen" || nameLowerCase == "Phosphorus" ||
        (nameLowerCase == "Boron" && trimName.compare(0, 3, "Bor") == 0)) {
      xos << "<ElementaryMaterial name=\"" << material->GetName() << "\""
          << " density=\"" << std::scientific << std::setprecision(5) << convertGPerCcToMgPerCc(material->GetDensity())
          << "*mg/cm3\""
          << " atomicWeight=\"" << std::fixed << material->GetA() << "*g/mole\"" << std::setprecision(0) << std::fixed
          << " atomicNumber=\"" << material->GetZ() << "\"/>" << std::endl;
    }
  }
}

void DDCoreToDDXMLOutput::rotation(const DDRotation& rotation, std::ostream& xos, const std::string& rotn) {
  double tol = 1.0e-3;  // Geant4 compatible
  DD3Vector x, y, z;
  rotation.rotation().GetComponents(x, y, z);
  double a, b, c;
  x.GetCoordinates(a, b, c);
  x.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  y.GetCoordinates(a, b, c);
  y.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  z.GetCoordinates(a, b, c);
  z.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  double check = (x.Cross(y)).Dot(z);  // in case of a LEFT-handed orthogonal system
                                       // this must be -1
  bool reflection((1. - check) > tol);
  std::string rotName = rotation.toString();
  if (rotName == ":") {
    if (!rotn.empty()) {
      rotName = rotn;
      std::cout << "about to try to make a new DDRotation... should fail!" << std::endl;
      DDRotation rot(DDName(rotn), std::make_unique<DDRotationMatrix>(rotation.rotation()));
      std::cout << "new rotation: " << rot << std::endl;
    } else {
      std::cout << "WARNING: MAKING AN UNNAMED ROTATION" << std::endl;
    }
  }
  if (!reflection) {
    xos << "<Rotation ";
  } else {
    xos << "<ReflectionRotation ";
  }
  using namespace cms_rounding;
  xos << "name=\"" << rotName << "\""
      << " phiX=\"" << roundIfNear0(convertRadToDeg(x.phi()), 4.e-4) << "*deg\""
      << " thetaX=\"" << roundIfNear0(convertRadToDeg(x.theta()), 4.e-4) << "*deg\""
      << " phiY=\"" << roundIfNear0(convertRadToDeg(y.phi()), 4.e-4) << "*deg\""
      << " thetaY=\"" << roundIfNear0(convertRadToDeg(y.theta()), 4.e-4) << "*deg\""
      << " phiZ=\"" << roundIfNear0(convertRadToDeg(z.phi()), 4.e-4) << "*deg\""
      << " thetaZ=\"" << roundIfNear0(convertRadToDeg(z.theta()), 4.e-4) << "*deg\"/>" << std::endl;
}

void DDCoreToDDXMLOutput::rotation(const dd4hep::Rotation3D& rotation,
                                   std::ostream& xos,
                                   const cms::DDParsingContext& context,
                                   const std::string& rotn) {
  double tol = 1.0e-3;  // Geant4 compatible
  ROOT::Math::XYZVector x, y, z;
  rotation.GetComponents(x, y, z);
  double a, b, c;
  x.GetCoordinates(a, b, c);
  x.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  y.GetCoordinates(a, b, c);
  y.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  z.GetCoordinates(a, b, c);
  z.SetCoordinates(cms_rounding::roundIfNear0(a), cms_rounding::roundIfNear0(b), cms_rounding::roundIfNear0(c));
  double check = (x.Cross(y)).Dot(z);  // in case of a LEFT-handed orthogonal system
                                       // this must be -1
  bool reflection((1. - check) > tol);
  if (!reflection) {
    xos << "<Rotation ";
  } else {
    xos << "<ReflectionRotation ";
  }
  using namespace cms_rounding;
  xos << "name=\"" << rotn << "\""
      << " phiX=\"" << roundIfNear0(convertRadToDeg(x.phi()), 4.e-4) << "*deg\""
      << " thetaX=\"" << roundIfNear0(convertRadToDeg(x.theta()), 4.e-4) << "*deg\""
      << " phiY=\"" << roundIfNear0(convertRadToDeg(y.phi()), 4.e-4) << "*deg\""
      << " thetaY=\"" << roundIfNear0(convertRadToDeg(y.theta()), 4.e-4) << "*deg\""
      << " phiZ=\"" << roundIfNear0(convertRadToDeg(z.phi()), 4.e-4) << "*deg\""
      << " thetaZ=\"" << roundIfNear0(convertRadToDeg(z.theta()), 4.e-4) << "*deg\"/>" << std::endl;
}

void DDCoreToDDXMLOutput::logicalPart(const DDLogicalPart& lp, std::ostream& xos) {
  xos << "<LogicalPart name=\"" << lp.toString() << "\">" << std::endl;
  xos << "<rSolid name=\"" << lp.solid().toString() << "\"/>" << std::endl;
  xos << "<rMaterial name=\"" << lp.material().toString() << "\"/>" << std::endl;
  xos << "</LogicalPart>" << std::endl;
}

void DDCoreToDDXMLOutput::logicalPart(const std::string& asName, std::ostream& xos) {
  xos << "<LogicalPart name=\"" << asName << "\">" << std::endl;
  xos << "<rSolid name=\"" << asName << "\"/>" << std::endl;
  xos << "<rMaterial name=\"materials:Air\"/>" << std::endl;
  xos << "</LogicalPart>" << std::endl;
}

void DDCoreToDDXMLOutput::logicalPart(const TGeoVolume& lp, std::ostream& xos) {
  xos << "<LogicalPart name=\"" << lp.GetName() << "\">" << std::endl;
  auto solid = lp.GetShape();
  if (solid != nullptr) {
    xos << "<rSolid name=\"" << trimShapeName(solid->GetName()) << "\"/>" << std::endl;
  }
  auto material = lp.GetMaterial();
  if (material != nullptr) {
    xos << "<rMaterial name=\"" << material->GetName() << "\"/>" << std::endl;
  }
  xos << "</LogicalPart>" << std::endl;
}

void DDCoreToDDXMLOutput::position(const TGeoVolume& parent,
                                   const TGeoNode& child,
                                   const std::string& childVolName,
                                   cms::DDParsingContext& context,
                                   std::ostream& xos) {
  xos << "<PosPart copyNumber=\"" << child.GetNumber() << "\">" << std::endl;
  xos << "<rParent name=\"" << parent.GetName() << "\"/>" << std::endl;
  xos << "<rChild name=\"" << childVolName << "\"/>" << std::endl;

  const auto matrix = child.GetMatrix();
  if (matrix != nullptr && !matrix->IsIdentity()) {
    auto rot = matrix->GetRotationMatrix();
    if (cms::rotation_utils::rotHash(rot) != cms::rotation_utils::identityHash) {
      std::string rotNameStr = cms::rotation_utils::rotName(rot, context);
      if (rotNameStr == "NULL") {
        rotNameStr = child.GetName();  // Phys vol name
        rotNameStr += parent.GetName();
        cms::DDNamespace nameSpace(context);
        cms::rotation_utils::addRotWithNewName(nameSpace, rotNameStr, rot);
      }
      xos << "<rRotation name=\"" << rotNameStr << "\"/>" << std::endl;
    }
  }
  auto trans = matrix->GetTranslation();
  using namespace cms_rounding;
  xos << "<Translation x=\"" << roundIfNear0(trans[0]) << "*mm\"";
  xos << " y=\"" << roundIfNear0(trans[1]) << "*mm\"";
  xos << " z=\"" << roundIfNear0(trans[2]) << "*mm\"";
  xos << "/>" << std::endl;
  xos << "</PosPart>" << std::endl;
}

void DDCoreToDDXMLOutput::position(const DDLogicalPart& parent,
                                   const DDLogicalPart& child,
                                   DDPosData* edgeToChild,
                                   int& rotNameSeed,
                                   std::ostream& xos) {
  std::string rotName = edgeToChild->ddrot().toString();
  DDRotationMatrix myIDENT;

  xos << "<PosPart copyNumber=\"" << edgeToChild->copyno() << "\">" << std::endl;
  xos << "<rParent name=\"" << parent.toString() << "\"/>" << std::endl;
  xos << "<rChild name=\"" << child.toString() << "\"/>" << std::endl;
  if ((edgeToChild->ddrot().rotation()) != myIDENT) {
    if (rotName == ":") {
      rotation(edgeToChild->ddrot(), xos);
    } else {
      xos << "<rRotation name=\"" << rotName << "\"/>" << std::endl;
    }
  }  // else let default Rotation matrix be created?
  using namespace cms_rounding;
  xos << "<Translation x=\"" << roundIfNear0(edgeToChild->translation().x()) << "*mm\""
      << " y=\"" << roundIfNear0(edgeToChild->translation().y()) << "*mm\""
      << " z=\"" << roundIfNear0(edgeToChild->translation().z()) << "*mm\"/>" << std::endl;
  xos << "</PosPart>" << std::endl;
}

void DDCoreToDDXMLOutput::specpar(const DDSpecifics& sp, std::ostream& xos) {
  xos << "<SpecPar name=\"" << sp.toString() << "\" eval=\"false\">" << std::endl;

  // ========...  all the selection strings out as strings by using the DDPartSelection's std::ostream function...
  for (const auto& psit : sp.selection()) {
    xos << "<PartSelector path=\"" << psit << "\"/>" << std::endl;
  }

  // =========  ... and iterate over all DDValues...
  for (const auto& vit : sp.specifics()) {
    const DDValue& v = vit.second;
    size_t s = v.size();
    size_t i = 0;
    // ============  ... all actual values with the same name
    const std::vector<std::string>& strvec = v.strings();
    if (v.isEvaluated()) {
      for (; i < s; ++i) {
        xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << v[i] << "\""
            << " eval=\"true\"/>" << std::endl;
      }
    } else {
      for (; i < s; ++i) {
        xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << strvec[i] << "\""
            << " eval=\"false\"/>" << std::endl;
      }
    }
  }
  xos << "</SpecPar>" << std::endl;
}

void DDCoreToDDXMLOutput::specpar(const std::string& name, const dd4hep::SpecPar& specPar, std::ostream& xos) {
  xos << "<SpecPar name=\"" << name << "\" eval=\"false\">" << std::endl;

  for (const auto& psit : specPar.paths) {
    xos << "<PartSelector path=\"" << psit << "\"/>" << std::endl;
  }

  for (const auto& vit : specPar.spars) {
    for (const auto& sit : vit.second) {
      xos << "<Parameter name=\"" << vit.first << "\""
          << " value=\"" << sit << "\""
          << " eval=\"false\"/>" << std::endl;
    }
  }
  for (const auto& vit : specPar.numpars) {
    for (const auto& sit : vit.second) {
      xos << "<Parameter name=\"" << vit.first << "\""
          << " value=\"" << sit << "\""
          << " eval=\"true\"/>" << std::endl;
    }
  }
  xos << "</SpecPar>" << std::endl;
}

void DDCoreToDDXMLOutput::specpar(const std::pair<DDsvalues_type, std::set<const DDPartSelection*>>& pssv,
                                  std::ostream& xos) {
  static const std::string madeName("specparname");
  static int numspecpars(0);
  std::ostringstream ostr;
  ostr << numspecpars++;
  std::string spname = madeName + ostr.str();
  xos << "<SpecPar name=\"" << spname << "\" eval=\"false\">" << std::endl;
  for (const auto& psit : pssv.second) {
    xos << "<PartSelector path=\"" << *psit << "\"/>" << std::endl;
  }

  // =========  ... and iterate over all DDValues...
  for (const auto& vit : pssv.first) {
    const DDValue& v = vit.second;
    size_t s = v.size();
    size_t i = 0;
    // ============  ... all actual values with the same name
    const std::vector<std::string>& strvec = v.strings();
    if (v.isEvaluated()) {
      for (; i < s; ++i) {
        xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << v[i] << "\""
            << " eval=\"true\"/>" << std::endl;
      }
    } else {
      for (; i < s; ++i) {
        xos << "<Parameter name=\"" << v.name() << "\""
            << " value=\"" << strvec[i] << "\""
            << " eval=\"false\"/>" << std::endl;
      }
    }
  }

  xos << "</SpecPar>" << std::endl;
}
