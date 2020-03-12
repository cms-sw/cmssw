///////////////////////////////////////////////////////////////////////////////
// File: DDTrackerPhiAltAlgo.cc
// Description: Position n copies inside and outside at alternate phi values
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

class DDTrackerPhiAltAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDTrackerPhiAltAlgo();
  ~DDTrackerPhiAltAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  double tilt;        //Tilt of the module
  double startAngle;  //offset in phi
  double rangeAngle;  //Maximum range in phi
  double radiusIn;    //Inner radius
  double radiusOut;   //Outer radius
  double zpos;        //z position
  int number;         //Number of copies
  int startCopyNo;    //Start copy number
  int incrCopyNo;     //Increment in copy number

  string idNameSpace;  //Namespace of this and ALL sub-parts
  string childName;    //Child name
};

DDTrackerPhiAltAlgo::DDTrackerPhiAltAlgo() {
  LogDebug("TrackerGeom") << "DDTrackerPhiAltAlgo info: Creating an instance";
}

DDTrackerPhiAltAlgo::~DDTrackerPhiAltAlgo() {}

void DDTrackerPhiAltAlgo::initialize(const DDNumericArguments& nArgs,
                                     const DDVectorArguments&,
                                     const DDMapArguments&,
                                     const DDStringArguments& sArgs,
                                     const DDStringVectorArguments&) {
  tilt = nArgs["Tilt"];
  startAngle = nArgs["StartAngle"];
  rangeAngle = nArgs["RangeAngle"];
  radiusIn = nArgs["RadiusIn"];
  radiusOut = nArgs["RadiusOut"];
  zpos = nArgs["ZPosition"];
  number = int(nArgs["Number"]);
  startCopyNo = int(nArgs["StartCopyNo"]);
  incrCopyNo = int(nArgs["IncrCopyNo"]);

  LogDebug("TrackerGeom") << "DDTrackerPhiAltAlgo debug: Parameters for "
                          << "positioning--"
                          << " Tilt " << tilt << "\tStartAngle " << startAngle / CLHEP::deg << "\tRangeAngle "
                          << rangeAngle / CLHEP::deg << "\tRin " << radiusIn << "\tRout " << radiusOut << "\t ZPos "
                          << zpos << "\tCopy Numbers " << number << " Start/Increment " << startCopyNo << ", "
                          << incrCopyNo;

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDTrackerPhiAltAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace;
}

void DDTrackerPhiAltAlgo::execute(DDCompactView& cpv) {
  if (number > 0) {
    double theta = 90. * CLHEP::deg;
    double dphi;
    if (number == 1 || fabs(rangeAngle - 360.0 * CLHEP::deg) < 0.001 * CLHEP::deg)
      dphi = rangeAngle / number;
    else
      dphi = rangeAngle / (number - 1);
    int copyNo = startCopyNo;

    DDName mother = parent().name();
    DDName child(DDSplit(childName).first, DDSplit(childName).second);
    for (int i = 0; i < number; i++) {
      double phi = startAngle + i * dphi;
      double phix = phi - tilt + 90. * CLHEP::deg;
      double phiy = phix + 90. * CLHEP::deg;
      double phideg = phix / CLHEP::deg;

      DDRotation rotation;
      if (phideg != 0) {
        string rotstr = DDSplit(childName).first + to_string(phideg * 10.);
        rotation = DDRotation(DDName(rotstr, idNameSpace));
        if (!rotation) {
          LogDebug("TrackerGeom") << "DDTrackerPhiAltAlgo test: Creating a new"
                                  << " rotation " << rotstr << "\t"
                                  << "90., " << phix / CLHEP::deg << ", 90.," << phiy / CLHEP::deg << ", 0, 0";
          rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy, 0., 0.);
        }
      }

      double xpos, ypos;
      if (i % 2 == 0) {
        xpos = radiusIn * cos(phi);
        ypos = radiusIn * sin(phi);
      } else {
        xpos = radiusOut * cos(phi);
        ypos = radiusOut * sin(phi);
      }
      DDTranslation tran(xpos, ypos, zpos);

      cpv.position(child, mother, copyNo, tran, rotation);
      LogDebug("TrackerGeom") << "DDTrackerPhiAltAlgo test: " << child << " number " << copyNo << " positioned in "
                              << mother << " at " << tran << " with " << rotation;
      copyNo += incrCopyNo;
    }
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDTrackerPhiAltAlgo, "track:DDTrackerPhiAltAlgo");
