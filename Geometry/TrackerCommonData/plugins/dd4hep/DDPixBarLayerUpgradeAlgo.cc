#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace cms_units::operators;

static long algorithm(dd4hep::Detector&, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);
  std::string parentName = args.parentName();

  std::string genMat = args.value<std::string>("GeneralMaterial");
  int number = args.value<int>("Ladders");
  double layerDz = args.value<double>("LayerDz");
  double coolDz = args.value<double>("CoolDz");
  double coolThick = args.value<double>("CoolThick");
  double coolRadius = args.value<double>("CoolRadius");
  double coolDist = args.value<double>("CoolDist");
  double cool1Offset = args.value<double>("Cool1Offset");
  double cool2Offset = args.value<double>("Cool2Offset");
  std::string coolMat = args.value<std::string>("CoolMaterial");
  std::string tubeMat = args.value<std::string>("CoolTubeMaterial");
  std::string coolMatHalf = args.value<std::string>("CoolMaterialHalf");
  std::string tubeMatHalf = args.value<std::string>("CoolTubeMaterial");
  double phiFineTune = args.value<double>("PitchFineTune");
  double rOuterFineTune = args.value<double>("OuterOffsetFineTune");
  double rInnerFineTune = args.value<double>("InnerOffsetFineTune");

  // FixMe : Would need ns.vecStr
  std::string ladder = args.value<std::string>("LadderName");

  double ladderWidth = args.value<double>("LadderWidth");
  double ladderThick = args.value<double>("LadderThick");
  double ladderOffset = args.value<double>("LadderOffset");
  int outerFirst = args.value<int>("OuterFirst");

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo debug: Parent " << parentName << " NameSpace "
                        << ns.objName(parentName) << "\n\tLadders " << number << "\tGeneral Material " << genMat
                        << "\tLength " << layerDz << "\tSpecification of Cooling Pieces:\n"
                        << "\tLength " << coolDz << "\tThickness of Shell " << coolThick << "\tRadial distance "
                        << coolDist << "\tMaterials " << coolMat << ", " << tubeMat << std::endl;

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo debug: Full Ladder " << ladder << " width/thickness "
                        << ladderWidth << ", " << ladderThick;

  double dphi = 2_pi / (double)number;
  double x2 = coolDist * sin(0.5 * dphi);
  double rtmi = coolDist * cos(0.5 * dphi) - (coolRadius + ladderThick) + rInnerFineTune;
  double rmxh = coolDist * cos(0.5 * dphi) + (coolRadius + ladderThick + ladderOffset) + rOuterFineTune;
  double rtmx = sqrt(rmxh * rmxh + ladderWidth * ladderWidth / 4.);

  std::string name = ns.objName(parentName);

  dd4hep::Solid solid = ns.addSolid(name, dd4hep::Tube(rtmi, rtmx, 0.5 * layerDz, 0., 2._pi));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << ns.name() << " Tubs made of " << genMat
                        << " from 0. to " << convertRadToDeg(2_pi) << " with Rin " << rtmi << " Rout " << rtmx
                        << " ZHalf " << 0.5 * layerDz;

  dd4hep::Volume layer = ns.addVolume(dd4hep::Volume(solid.name(), solid, ns.material(genMat)));

  // Full Tubes
  solid = ns.addSolid(name + "CoolTube", dd4hep::Tube(0., coolRadius, 0.5 * coolDz, 0., 2_pi));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << solid.name() << " Tubs made of " << tubeMat
                        << " from 0 to " << convertRadToDeg(2_pi) << " with Rout " << coolRadius << " ZHalf "
                        << 0.5 * coolDz;

  dd4hep::Volume coolTube = ns.addVolume(dd4hep::Volume(solid.name(), solid, ns.material(tubeMat)));

  // Half Tubes
  solid = ns.addSolid(name + "CoolTubeHalf", dd4hep::Tube(0., coolRadius, 0.5 * coolDz, 0, 1_pi));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << solid.name() << " Tubs made of " << tubeMatHalf
                        << " from 0 to " << convertRadToDeg(2_pi) << " with Rout " << coolRadius << " ZHalf "
                        << 0.5 * coolDz;

  dd4hep::Volume coolTubeHalf = ns.addVolume(dd4hep::Volume(solid.name(), solid, ns.material(tubeMatHalf)));

  // Full Coolant
  name = ns.objName(parentName);

  solid = ns.addSolid(name + "Coolant", dd4hep::Tube(0., coolRadius - coolThick, 0.5 * coolDz, 0., 2_pi));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << solid.name() << " Tubs made of " << tubeMat
                        << " from 0 to " << convertRadToDeg(2._pi) << " with Rout " << coolRadius - coolThick
                        << " ZHalf " << 0.5 * coolDz;

  dd4hep::Volume cool = ns.addVolume(dd4hep::Volume(solid.name(), solid, ns.material(coolMat)));

  coolTube.placeVolume(cool, 1, dd4hep::Transform3D(dd4hep::Rotation3D(), dd4hep::Position(0., 0., 0.)));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << cool.name() << " number 1 positioned in "
                        << coolTube.name() << " at (0,0,0) with no rotation";

  // Half Coolant

  solid = ns.addSolid(name + "CoolantHalf", dd4hep::Tube(0., coolRadius - coolThick, 0.5 * coolDz, 0., 1._pi));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << solid.name() << " Tubs made of " << tubeMatHalf
                        << " from 0 to " << convertRadToDeg(2._pi) << " with Rout " << coolRadius - coolThick
                        << " ZHalf " << 0.5 * coolDz;

  dd4hep::Volume coolHalf = ns.addVolume(dd4hep::Volume(solid.name(), solid, ns.material(coolMatHalf)));

  coolTubeHalf.placeVolume(coolHalf, 1, dd4hep::Transform3D(dd4hep::Rotation3D(), dd4hep::Position(0., 0., 0.)));

  LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << cool.name() << " number 1 positioned in "
                        << coolTube.name() << " at (0,0,0) with no rotation";
  int copy = 1, iup = (-1) * outerFirst;
  int copyoffset = number + 2;

  for (int i = 1; i < number + 1; i++) {
    double phi = i * dphi + 90._deg - 0.5 * dphi + phiFineTune;  //to start with the interface ladder
    double phix, phiy, rrr, rrroffset;
    std::string rots;

    auto tran = dd4hep::Position();
    auto rot = dd4hep::Rotation3D();

    iup = -iup;

    double dr;

    if ((i == 1) || (i == number / 2 + 1)) {
      dr = coolRadius + 0.5 * ladderThick + ladderOffset;  //interface ladder offset
    } else {
      dr = coolRadius + 0.5 * ladderThick;
    }

    if (i % 2 == 1) {
      rrr = coolDist * cos(0.5 * dphi) + iup * dr + rOuterFineTune;
    } else {
      rrr = coolDist * cos(0.5 * dphi) + iup * dr + rInnerFineTune;
    }

    tran = dd4hep::Position(rrr * cos(phi), rrr * sin(phi), 0);
    rots = name + std::to_string(copy);

    if (iup > 0) {
      phix = phi - 90._deg;
    } else {
      phix = phi + 90._deg;
    }

    phiy = phix + 90._deg;

    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: Creating a new "
                          << "rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                          << convertRadToDeg(phiy) << ", 0, 0";

    rot = cms::makeRotation3D(90._deg, phix, 90._deg, phiy, 0., 0.);

    // FixMe : Would need ns.vecStr
    layer.placeVolume(ns.volume(ladder), copy, dd4hep::Transform3D(rot, tran));

    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << ladder << " number " << copy << " positioned in "
                          << layer.name() << " at " << tran << " with " << rot;

    copy++;

    rrr = coolDist * cos(0.5 * dphi) + coolRadius / 2.;

    rots = name + std::to_string(i + 100);
    phix = phi + 90._deg;

    if (iup < 0)
      phix += dphi;

    phiy = phix + 90._deg;

    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: Creating a new "
                          << "rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                          << convertRadToDeg(phiy) << ", 0, 0";

    tran = dd4hep::Position(rrr * cos(phi) - x2 * sin(phi), rrr * sin(phi) + x2 * cos(phi), 0.);

    rot = cms::makeRotation3D(90._deg, phix, 90._deg, phiy, 0., 0.);

    layer.placeVolume(coolTubeHalf, i + 1, dd4hep::Transform3D(rot, tran));

    if ((i == 1) || (i == number / 2 + 1)) {
      rrroffset = coolDist * cos(0.5 * dphi) + iup * ladderOffset + rOuterFineTune;
      tran = dd4hep::Position(
          rrroffset * cos(phi) - cool1Offset * sin(phi), rrroffset * sin(phi) + cool1Offset * cos(phi), 0.);

      layer.placeVolume(coolTube, copyoffset, dd4hep::Transform3D(dd4hep::Rotation3D(), tran));
      ;

      copyoffset++;
      tran = dd4hep::Position(
          rrroffset * cos(phi) - cool2Offset * sin(phi), rrroffset * sin(phi) + cool2Offset * cos(phi), 0.);

      layer.placeVolume(coolTube, copyoffset, dd4hep::Transform3D(dd4hep::Rotation3D(), tran));
      copyoffset++;
    }

    LogDebug("PixelGeom") << "DDPixBarLayerUpgradeAlgo test: " << coolTube.name() << " number " << i + 1
                          << " positioned in " << layer.name() << " at " << tran << " with " << rot;
  }

  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDPixBarLayerUpgradeAlgo, algorithm);
