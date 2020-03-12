///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerAngular.cc
// Description: Position n copies at prescribed phi values
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <string>
#include <vector>

using namespace std;

class DDTrackerAngular : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerAngular();
  ~DDTrackerAngular() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  int n;                  //Number of copies
  int startCopyNo;        //Start Copy number
  int incrCopyNo;         //Increment in Copy number
  double rangeAngle;      //Range in angle
  double startAngle;      //Start anle
  double radius;          //Radius
  vector<double> center;  //Phi values
  double delta;           //Increment in phi

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
};

DDTrackerAngular::DDTrackerAngular() { LogDebug("TrackerGeom") << "DDTrackerAngular info: Creating an instance"; }

DDTrackerAngular::~DDTrackerAngular() {}

void DDTrackerAngular::initialize(const DDNumericArguments& nArgs,
                                  const DDVectorArguments& vArgs,
                                  const DDMapArguments&,
                                  const DDStringArguments& sArgs,
                                  const DDStringVectorArguments&) {
  n = int(nArgs["N"]);
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo = int(nArgs["IncrCopyNo"]);
  rangeAngle = nArgs["RangeAngle"];
  startAngle = nArgs["StartAngle"];
  radius = nArgs["Radius"];
  center = vArgs["Center"];

  if (fabs(rangeAngle - 360.0 * CLHEP::deg) < 0.001 * CLHEP::deg) {
    delta = rangeAngle / double(n);
  } else {
    if (n > 1) {
      delta = rangeAngle / double(n - 1);
    } else {
      delta = 0.;
    }
  }

  LogDebug("TrackerGeom") << "DDTrackerAngular debug: Parameters for position"
                          << "ing:: n " << n << " Start, Range, Delta " << startAngle / CLHEP::deg << " "
                          << rangeAngle / CLHEP::deg << " " << delta / CLHEP::deg << " Radius " << radius << " Centre "
                          << center[0] << ", " << center[1] << ", " << center[2];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];

  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerAngular debug: Parent " << parentName << "\tChild " << childName << " NameSpace "
                          << idNameSpace;
}

void DDTrackerAngular::execute(DDCompactView& cpv) {
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  double theta = 90. * CLHEP::deg;
  int copy = startCopyNo;
  double phi = startAngle;
  for (int i = 0; i < n; i++) {
    double phix = phi;
    double phiy = phix + 90. * CLHEP::deg;
    double phideg = phix / CLHEP::deg;

    DDRotation rotation;
    if (phideg != 0) {
      string rotstr = DDSplit(childName).first + to_string(phideg * 10.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
        LogDebug("TrackerGeom") << "DDTrackerAngular test: Creating a new "
                                << "rotation: " << rotstr << "\t90., " << phix / CLHEP::deg << ", 90.,"
                                << phiy / CLHEP::deg << ", 0, 0";
        rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
      }
    }

    double xpos = radius * cos(phi) + center[0];
    double ypos = radius * sin(phi) + center[1];
    double zpos = center[2];
    DDTranslation tran(xpos, ypos, zpos);

    cpv.position(child, mother, copy, tran, rotation);
    LogDebug("TrackerGeom") << "DDTrackerAngular test " << child << " number " << copy << " positioned in " << mother
                            << " at " << tran << " with " << rotation;
    copy += incrCopyNo;
    phi += delta;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerAngular, "track:DDTrackerAngular");
