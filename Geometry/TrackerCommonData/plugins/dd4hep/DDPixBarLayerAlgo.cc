#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

static long algorithm(Detector& description, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  PlacedVolume pv;
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  string parentName = args.parentName();

  LogDebug("PixelGeom") << "+++ Parsing arguments for Algorithm:" << args.name << " rParent:" << parentName;
  string genMat = args.value<string>("GeneralMaterial");
  int number = args.value<int>("Ladders");
  double layerDz = args.value<double>("LayerDz");
  double sensorEdge = args.value<double>("SensorEdge");
  double coolDz = args.value<double>("CoolDz");
  double coolWidth = args.value<double>("CoolWidth");
  double coolSide = args.value<double>("CoolSide");
  double coolThick = args.value<double>("CoolThick");
  double coolDist = args.value<double>("CoolDist");
  string coolMat = args.value<string>("CoolMaterial");
  string tubeMat = args.value<string>("CoolTubeMaterial");
  LogDebug("PixelGeom") << "Parent " << parentName << " NameSpace " << ns.name() << "\n"
                        << "\tLadders " << number << "\tGeneral Material " << genMat << "\tLength " << layerDz
                        << "\tSensorEdge " << sensorEdge << "\tSpecification of Cooling Pieces:\n"
                        << "\tLength " << coolDz << " Width " << coolWidth << " Side " << coolSide
                        << " Thickness of Shell " << coolThick << " Radial distance " << coolDist << " Materials "
                        << coolMat << ", " << tubeMat;
  vector<string> ladder = args.value<vector<string> >("LadderName");
  vector<double> ladderWidth = args.value<vector<double> >("LadderWidth");
  vector<double> ladderThick = args.value<vector<double> >("LadderThick");
  LogDebug("PixelGeom") << "Full Ladder " << ladder[0] << " width/thickness " << ladderWidth[0] << ", "
                        << ladderThick[0] << "\tHalf Ladder " << ladder[1] << " width/thickness " << ladderWidth[1]
                        << ", " << ladderThick[1];

  const std::string idName = ns.objName(parentName);
  double dphi = 2_pi / number;
  double d2 = 0.5 * coolWidth;
  double d1 = d2 - coolSide * sin(0.5 * dphi);
  double x1 = (d1 + d2) / (2. * sin(0.5 * dphi));
  double x2 = coolDist * sin(0.5 * dphi);
  double rmin = (coolDist - 0.5 * (d1 + d2)) * cos(0.5 * dphi) - 0.5 * ladderThick[0];
  double rmax = (coolDist + 0.5 * (d1 + d2)) * cos(0.5 * dphi) + 0.5 * ladderThick[0];
  double rmxh = rmax - 0.5 * ladderThick[0] + ladderThick[1];
  LogDebug("PixelGeom") << "Rmin/Rmax " << rmin << ", " << rmax << " d1/d2 " << d1 << ", " << d2 << " x1/x2 " << x1
                        << ", " << x2;

  double rtmi = rmin + 0.5 * ladderThick[0] - ladderThick[1];
  double rtmx = sqrt(rmxh * rmxh + ladderWidth[1] * ladderWidth[1]);
  Solid solid = ns.addSolid(idName, Tube(rtmi, rtmx, 0.5 * layerDz, 0, 2_pi));
  LogDebug("PixelGeom") << "IDname " << idName << " Tubs made of " << genMat << " from 0 to " << convertRadToDeg(2_pi)
                        << " with Rin " << rtmi << " Rout " << rtmx << " ZHalf " << 0.5 * layerDz;

  Volume layer = ns.addVolume(Volume(idName, solid, ns.material(genMat)));
  double rr = 0.5 * (rmax + rmin);
  double dr = 0.5 * (rmax - rmin);
  double h1 = 0.5 * coolSide * cos(0.5 * dphi);
  std::string name = idName + "CoolTube";
  solid = ns.addSolid(name, Trap(0.5 * coolDz, 0, 0, h1, d2, d1, 0, h1, d2, d1, 0));
  LogDebug("PixelGeom") << "Solid " << solid.name() << " Trap made of " << tubeMat << " of dimensions " << 0.5 * coolDz
                        << ", 0, 0, " << h1 << ", " << d2 << ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " << d1
                        << ", 0";

  Volume coolTube = ns.addVolume(Volume(name, solid, description.material(tubeMat)));
  h1 -= coolThick;
  d1 -= coolThick;
  d2 -= coolThick;
  name = idName + "Coolant";
  solid = ns.addSolid(name, Trap(0.5 * coolDz, 0, 0, h1, d2, d1, 0, h1, d2, d1, 0));
  LogDebug("PixelGeom") << "Solid " << solid.name() << " Trap made of " << coolMat << " of dimensions " << 0.5 * coolDz
                        << ", 0, 0, " << h1 << ", " << d2 << ", " << d1 << ", 0, " << h1 << ", " << d2 << ", " << d1
                        << ", 0";

  Volume cool = ns.addVolume(Volume(name, solid, description.material(coolMat)));
  pv = coolTube.placeVolume(cool, 1);
  LogDebug("PixelGeom") << "Cool " << cool.name() << " number 1 positioned in " << coolTube.name()
                        << " at (0,0,0) with no rotation";

  string ladderFull = ladder[0];
  string ladderHalf = ladder[1];
  int nphi = number / 2, copy = 1, iup = -1;
  double phi0 = 90_deg;
  Volume ladderHalfVol = ns.volume(ladderHalf);
  Volume ladderFullVol = ns.volume(ladderFull);

  for (int i = 0; i < number; i++) {
    double phi = phi0 + i * dphi;
    double phix, phiy, rrr, xx;
    std::string rots;
    Position tran;
    Rotation3D rot;
    if (i == 0 || i == nphi) {
      rrr = rr + dr + 0.5 * (ladderThick[1] - ladderThick[0]);
      xx = (0.5 * ladderWidth[1] - sensorEdge) * sin(phi);
      tran = Position(xx, rrr * sin(phi), 0);
      rots = idName + std::to_string(copy);
      phix = phi - 90_deg;
      phiy = 90_deg + phix;
      LogDebug("PixelGeom") << "Creating a new "
                            << "rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                            << convertRadToDeg(phiy) << ", 0, 0";
      rot = makeRotation3D(90_deg, phix, 90_deg, phiy, 0., 0.);

      //cpv.position(ladderHalf, layer, copy, tran, rot);
      pv = layer.placeVolume(ladderHalfVol, copy, Transform3D(rot, tran));
      if (!pv.isValid()) {
      }
      LogDebug("PixelGeom") << "ladderHalfVol: " << ladderHalfVol.name() << " number " << copy << " positioned in "
                            << layer.name() << " at " << tran << " with " << rot;
      copy++;
      iup = -1;
      rrr = rr - dr - 0.5 * (ladderThick[1] - ladderThick[0]);
      tran = Position(-xx, rrr * sin(phi), 0);
      rots = idName + std::to_string(copy);
      phix = phi + 90_deg;
      phiy = 90_deg + phix;
      LogDebug("PixelGeom") << "Creating a new rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                            << convertRadToDeg(phiy) << ", 0, 0";
      rot = makeRotation3D(90_deg, phix, 90_deg, phiy, 0., 0.);
      //cpv.position(ladderHalf, layer, copy, tran, rot);
      pv = layer.placeVolume(ladderHalfVol, copy, Transform3D(rot, tran));
      if (!pv.isValid()) {
      }
      LogDebug("PixelGeom") << "ladderHalfVol: " << ladderHalfVol.name() << " number " << copy << " positioned in "
                            << layer.name() << " at " << tran << " with " << rot;
      copy++;
    } else {
      iup = -iup;
      rrr = rr + iup * dr;
      tran = Position(rrr * cos(phi), rrr * sin(phi), 0);
      rots = idName + std::to_string(copy);
      if (iup > 0)
        phix = phi - 90_deg;
      else
        phix = phi + 90_deg;
      phiy = phix + 90._deg;
      LogDebug("PixelGeom") << "DDPixBarLayerAlgo test: Creating a new "
                            << "rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                            << convertRadToDeg(phiy) << ", 0, 0";
      rot = makeRotation3D(90_deg, phix, 90_deg, phiy, 0., 0.);

      //cpv.position(ladderFull, layer, copy, tran, rot);
      pv = layer.placeVolume(ladderFullVol, copy, Transform3D(rot, tran));
      if (!pv.isValid()) {
      }
      LogDebug("PixelGeom") << "test: " << ladderFullVol.name() << " number " << copy << " positioned in "
                            << layer.name() << " at " << tran << " with " << rot;
      copy++;
    }
    rrr = coolDist * cos(0.5 * dphi);
    tran = Position(rrr * cos(phi) - x2 * sin(phi), rrr * sin(phi) + x2 * cos(phi), 0);
    rots = idName + std::to_string(i + 100);
    phix = phi + 0.5 * dphi;
    if (iup > 0)
      phix += 180_deg;
    phiy = phix + 90._deg;
    LogDebug("PixelGeom") << "Creating a new rotation: " << rots << "\t90., " << convertRadToDeg(phix) << ", 90.,"
                          << convertRadToDeg(phiy) << ", 0, 0";

    rot = makeRotation3D(90_deg, phix, 90_deg, phiy, 0., 0.);
    pv = layer.placeVolume(coolTube, i + 1, Transform3D(rot, tran));
    if (!pv.isValid()) {
    }
    LogDebug("PixelGeom") << "coolTube: " << coolTube.name() << " number " << i + 1 << " positioned in " << layer.name()
                          << " at " << tran << " with " << rot;
  }
  LogDebug("PixelGeom") << "Layer: " << layer.name();
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDPixBarLayerAlgo, algorithm)
