#include "DD4hep/DetFactoryHelper.h"
#include "DataFormats/Math/interface/CMSUnits.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <Math/Cartesian3D.h>
#include <Math/DisplacementVector3D.h>

using namespace std;
using namespace dd4hep;
using namespace cms;
using namespace cms_units::operators;

using DD3Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<double> >;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);

  int n = args.value<int>("N");
  int startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  int incrCopyNo = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;
  double theta = args.find("Theta") ? args.value<double>("Theta") : 0.;
  double phi = args.find("Phi") ? args.value<double>("Phi") : 0.;
  double delta = args.find("Delta") ? args.value<double>("Delta") : 0.;
  vector<double> base = args.value<vector<double> >("Base");
  Volume mother = ns.volume(args.parentName());
  Volume child = ns.volume(args.value<string>("ChildName"));
  int copy = startCopyNo;

  LogDebug("DDAlgorithm") << "DDLinear: Parameters for positioning:: n " << n << "  Direction Theta, Phi, Delta "
                          << convertRadToDeg(theta) << " " << convertRadToDeg(phi) << " " << convertRadToDeg(delta)
                          << " Base " << base[0] << ", " << base[1] << ", " << base[2];

  LogDebug("DDAlgorithm") << "DDLinear: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace "
                          << ns.name();

  Position direction(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));

  Position basetr(base[0], base[1], base[2]);

  Rotation3D rotation = Rotation3D();  // Identity rotation

  for (int i = 0; i < n; ++i) {
    Position tran = basetr + (double(copy) * delta) * direction;
    mother.placeVolume(child, copy, Transform3D(rotation, tran));
    LogDebug("DDAlgorithm") << "DDLinear: " << child.name() << " number " << copy << " positioned in " << mother.name()
                            << " at " << tran << " with " << rotation;

    copy += incrCopyNo;
  }

  return 1;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_global_DDLinear, algorithm)
