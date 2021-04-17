#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/Rotation3D.h>
#include <Math/AxisAngle.h>
#include <Math/DisplacementVector3D.h>

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >;
using DDAxisAngle = ROOT::Math::AxisAngle;

namespace {
  DD3Vector fUnitVector(double theta, double phi) {
    return DD3Vector(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
  }
}  // namespace

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);

  int n = args.value<int>("N");
  int startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  int incrCopyNo = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;
  double rangeAngle = args.value<double>("RangeAngle");
  double startAngle = args.value<double>("StartAngle");
  double radius = args.value<double>("Radius");
  vector<double> center = args.value<vector<double> >("Center");
  vector<double> rotateSolid = args.value<vector<double> >("RotateSolid");
  Volume mother = ns.volume(args.parentName());
  string childName = args.value<string>("ChildName");
  childName = ns.prepend(childName);
  Volume child = ns.volume(childName);

  double delta = 0e0;

  if (fabs(rangeAngle - 360.0_deg) < 0.001_deg) {
    delta = rangeAngle / double(n);
  } else if (n > 1) {
    delta = rangeAngle / double(n - 1);
  }

  LogDebug("DDAlgorithm") << "debug: Parameters for positioning:: n " << n << " Start, Range, Delta "
                          << convertRadToDeg(startAngle) << " " << convertRadToDeg(rangeAngle) << " "
                          << convertRadToDeg(delta) << " Radius " << radius << " Centre " << center[0] << ", "
                          << center[1] << ", " << center[2] << ", Rotate solid " << rotateSolid[0] << ", "
                          << rotateSolid[1] << ", " << rotateSolid[2];
  LogDebug("DDAlgorithm") << "debug: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace "
                          << ns.name();

  Rotation3D solidRot = Rotation3D();  // Identity rotation
  auto sz = rotateSolid.size();
  if (sz % 3) {
    LogDebug("DDAlgorithm") << "\trotateSolid must occur 3*n times (defining n subsequent rotations)\n"
                            << "\t  currently it appears " << sz << " times!\n";
  }
  for (unsigned int i = 0; i < sz; i += 3) {
    const double thetaValue = rotateSolid[i];
    if ((thetaValue > 180._deg) || (thetaValue < 0._deg)) {
      LogDebug("DDAlgorithm") << "\trotateSolid \'theta\' must be in range [0,180*deg]\n"
                              << "\t  currently it is " << convertRadToDeg(thetaValue) << "*deg in rotateSolid["
                              << double(i) << "]!\n";
    }
    DDAxisAngle temp(fUnitVector(rotateSolid[i], rotateSolid[i + 1]), rotateSolid[i + 2]);
    LogDebug("DDAlgorithm") << "  rotsolid[" << i << "] axis=" << temp.Axis()
                            << " rot.angle=" << convertRadToDeg(temp.Angle());
    solidRot = temp * solidRot;
  }

  double theta = 90._deg;
  int copy = startCopyNo;
  double phi = startAngle;
  for (int i = 0; i < n; ++i) {
    double phix = phi;
    double phiy = phix + 90._deg;
    double phideg = convertRadToDeg(phix);

    Rotation3D rotation = makeRotation3D(theta, phix, theta, phiy, 0., 0.) * solidRot;
    string rotstr = ns.nsName(child.name()) + std::to_string(phideg * 10.);
    auto irot = ctxt.rotations.find(ns.prepend(rotstr));
    if (irot != ctxt.rotations.end()) {
      rotation = ns.rotation(ns.prepend(rotstr));
    }

    double xpos = radius * cos(phi) + center[0];
    double ypos = radius * sin(phi) + center[1];
    double zpos = center[2];
    Position tran(xpos, ypos, zpos);
    mother.placeVolume(child, copy, Transform3D(rotation, tran));
    LogDebug("DDAlgorithm") << "test " << child.name() << " number " << copy << " positioned in " << mother.name()
                            << " at " << tran << " with " << rotstr << ": " << rotation;
    copy += incrCopyNo;
    phi += delta;
  }
  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_global_DDAngular, algorithm)
