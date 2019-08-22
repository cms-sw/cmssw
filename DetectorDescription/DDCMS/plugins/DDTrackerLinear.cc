#include "DD4hep/DetFactoryHelper.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace dd4hep;
using namespace cms;

static long algorithm(Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e, SensitiveDetector& /* sens */) {
  cms::DDNamespace ns(ctxt, e, true);
  DDAlgoArguments args(ctxt, e);
  int startcn = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 1;
  int incrcn = args.find("IncrCopyNo") ? args.value<int>("IncrCopyNo") : 1;
  int number = args.value<int>("Number");
  double theta = args.value<double>("Theta");
  double phi = args.value<double>("Phi");
  double offset = args.value<double>("Offset");
  double delta = args.value<double>("Delta");
  vector<double> centre = args.value<vector<double> >("Center");
  string rotMat = args.value<string>("Rotation");
  Volume mother = ns.volume(args.parentName());
  Volume child = ns.volume(args.value<string>("ChildName"));

  LogDebug("TrackerGeom") << "DDTrackerLinear +++ Executing Algorithm. rParent:" << mother.name();
  LogDebug("TrackerGeom") << "debug: Parent " << mother.name() << "\tChild " << child.name() << " NameSpace "
                          << ns.name() << "\tNumber " << number << "\tAxis (theta/phi) " << theta / dd4hep::deg << ", "
                          << phi / dd4hep::deg << "\t(Offset/Delta) " << offset << ", " << delta << "\tCentre "
                          << centre[0] << ", " << centre[1] << ", " << centre[2] << "\tRotation " << rotMat;

  Position direction(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
  Position base(centre[0], centre[1], centre[2]);

  const Rotation3D& rot = ns.rotation(rotMat);

  for (int i = 0, ci = startcn; i < number; i++, ci += incrcn) {
    Position tran = base + (offset + double(i) * delta) * direction;
    mother.placeVolume(child, ci, Transform3D(rot, tran));

    LogDebug("TrackerGeom") << child.name() << " number " << ci << " positioned in " << mother.name() << " at " << tran
                            << " with " << rot;
  }
  return cms::s_executed;
}

// first argument is the type from the xml file
DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDTrackerLinear, algorithm)
