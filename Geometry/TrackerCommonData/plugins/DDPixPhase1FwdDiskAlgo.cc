///////////////////////////////////////////////////////////////////////////////
// File: DDPixPhase1FwdDiskAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDTypes.h"
#include "DetectorDescription/Core/interface/DDAlgorithm.h"
#include "DetectorDescription/Core/interface/DDAlgorithmFactory.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

#include <cmath>
#include <algorithm>
#include <map>
#include <string>
#include <vector>

using namespace std;

class DDPixPhase1FwdDiskAlgo : public DDAlgorithm {
public:
  //Constructor and Destructor
  DDPixPhase1FwdDiskAlgo();
  ~DDPixPhase1FwdDiskAlgo() override;

  void initialize(const DDNumericArguments& nArgs,
                  const DDVectorArguments& vArgs,
                  const DDMapArguments& mArgs,
                  const DDStringArguments& sArgs,
                  const DDStringVectorArguments& vsArgs) override;

  void execute(DDCompactView& cpv) override;

private:
  string idNameSpace;          //Namespace of this and ALL sub-parts
  string childName;            //Child name
  string rotName;              //Name of the base rotation matrix
  string flagString;           //Flag if a blade is present
  int nBlades;                 //Number of blades
  int startCopyNo;             //Start Copy number
  double bladeAngle;           //Angle of blade rotation aroung y-axis
  double zPlane;               //Common shift in z for all blades
  vector<double> bladeZShift;  //Shift in Z of individual blades
  double anchorR;              //Distance of beam line to anchor point
  double bladeTilt;            //Tilt of the blade around x-axis
};

DDPixPhase1FwdDiskAlgo::DDPixPhase1FwdDiskAlgo() {
  LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo info: Creating an instance";
}

DDPixPhase1FwdDiskAlgo::~DDPixPhase1FwdDiskAlgo() {}

void DDPixPhase1FwdDiskAlgo::initialize(const DDNumericArguments& nArgs,
                                        const DDVectorArguments& vArgs,
                                        const DDMapArguments&,
                                        const DDStringArguments& sArgs,
                                        const DDStringVectorArguments& vsArgs) {
  startCopyNo = int(nArgs["StartCopyNo"]);
  nBlades = int(nArgs["NumberOfBlades"]);
  bladeAngle = nArgs["BladeAngle"];
  bladeTilt = -nArgs["BladeTilt"];
  zPlane = nArgs["BladeCommonZ"];
  bladeZShift = vArgs["BladeZShift"];
  anchorR = nArgs["AnchorRadius"];

  idNameSpace = DDCurrentNamespace::ns();
  childName = sArgs["ChildName"];
  rotName = sArgs["RotationName"];
  flagString = sArgs["FlagString"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo debug: Parent " << parentName << "\tChild " << childName
                          << " NameSpace " << idNameSpace << "\tRot Name " << rotName << "\tCopyNo (Start/Total) "
                          << startCopyNo << ", " << nBlades << "\tAngles " << bladeAngle / CLHEP::deg << ", "
                          << bladeTilt / CLHEP::deg << "\tZshifts " << zPlane << "\tAmnchor Radius " << anchorR;

  for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
    LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo: Blade " << iBlade << " flag " << flagString[iBlade]
                            << " zshift " << bladeZShift[iBlade];
  }
}

void DDPixPhase1FwdDiskAlgo::execute(DDCompactView& cpv) {
  int copy = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  string flagSelector = "Y";

  double deltaPhi = (360. / nBlades) * CLHEP::deg;
  string rotns = DDSplit(rotName).second;
  for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
    if (flagString[iBlade] == flagSelector[0]) {
      string rotstr = DDSplit(rotName).first + to_string(double(copy));

      double phi = (iBlade + 0.5) * deltaPhi;
      double phiy = atan2(cos(phi), -sin(phi));
      double thety = acos(sin(bladeTilt));
      double phix = atan2(cos(bladeAngle) * sin(phi) + cos(phi) * sin(bladeTilt) * sin(bladeAngle),
                          cos(phi) * cos(bladeAngle) - sin(phi) * sin(bladeTilt) * sin(bladeAngle));
      double thetx = acos(-cos(bladeTilt) * sin(bladeAngle));
      double phiz = atan2(sin(phi) * sin(bladeAngle) - cos(phi) * cos(bladeAngle) * sin(bladeTilt),
                          cos(phi) * sin(bladeAngle) + cos(bladeAngle) * sin(phi) * sin(bladeTilt));
      double thetz = acos(cos(bladeTilt) * cos(bladeAngle));
      DDRotation rot = DDRotation(DDName(rotstr, rotns));
      if (!rot) {
        LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo test: Creating a new "
                                << "rotation: " << rotstr << "\t" << thetx / CLHEP::deg << ", " << phix / CLHEP::deg
                                << ", " << thety / CLHEP::deg << ", " << phiy / CLHEP::deg << ", " << thetz / CLHEP::deg
                                << ", " << phiz / CLHEP::deg;
        LogDebug("TrackerGeom") << "Rotation Matrix (" << phi / CLHEP::deg << ", " << bladeAngle / CLHEP::deg << ", "
                                << bladeTilt / CLHEP::deg << ") " << cos(phi) * cos(bladeAngle) << ", "
                                << (-sin(phi) * cos(bladeTilt) + cos(phi) * sin(bladeAngle) * sin(bladeTilt)) << ", "
                                << (sin(phi) * sin(bladeTilt) + cos(phi) * sin(bladeAngle) * cos(bladeTilt)) << ", "
                                << sin(phi) * cos(bladeAngle) << ", "
                                << (cos(phi) * cos(bladeTilt) + sin(phi) * sin(bladeAngle) * sin(bladeTilt)) << ", "
                                << (-cos(phi) * sin(bladeTilt) + sin(phi) * sin(bladeAngle) * cos(bladeTilt)) << ", "
                                << -sin(bladeAngle) << ", " << cos(bladeAngle) * sin(bladeTilt) << ", "
                                << cos(bladeAngle) * cos(bladeTilt);
        rot = DDrot(DDName(rotstr, rotns), thetx, phix, thety, phiy, thetz, phiz);
      }
      double xpos = -anchorR * sin(phi);
      double ypos = anchorR * cos(phi);
      double zpos = zPlane + bladeZShift[iBlade % nBlades];
      DDTranslation tran(xpos, ypos, zpos);
      cpv.position(child, mother, copy, tran, rot);
      LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo test: " << child << " number " << copy << " positioned in "
                              << mother << " at " << tran << " with " << rot;
    }
    copy++;
  }
}

DEFINE_EDM_PLUGIN(DDAlgorithmFactory, DDPixPhase1FwdDiskAlgo, "track:DDPixPhase1FwdDiskAlgo");
