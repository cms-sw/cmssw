///////////////////////////////////////////////////////////////////////////////
// File: DDPixFwdDiskAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"
#include <sstream>

using namespace std;
using namespace dd4hep;
using namespace cms;

static long algorithm(Detector& /* description */,
                      cms::DDParsingContext& ctxt,
                      xml_h e,
                      SensitiveDetector& /* sens */)
{

  cms::DDNamespace ns(ctxt,e,true);
  DDAlgoArguments args(ctxt, e);

  int nBlades; //Number of blades
  int startCopyNo; //Start Copy number
  double bladeAngle; //Angle of blade rotation aroung y-axis
  double bladeTilt; //Tilt of the blade around x-axis
  double zPlane; //Common shift in z for all blades
  vector<double> bladeZShift; //Shift in Z of individual blades
  double anchorR; //Distance of beam line to anchor point

  string childName; //Child name
  string rotName; //Name of the base rotation matrix
  string flagString; //Flag if a blade is present

  Volume mother = ns.volume(args.parentName());
  PlacedVolume pv;

  startCopyNo = args.find("startCopyNo") ? args.value<int>("startCopyNo") : 1;
  nBlades = args.value<int>("NumberOfBlades");
  bladeAngle = args.value<double>("BladeAngle");
  bladeTilt = args.value<double>("BladeCommonZ");
  zPlane = args.value<double>("BladeCommonZ");
  bladeZShift = args.value<vector<double> >("BladeZShift");
  anchorR = args.value<double>("AnchorRadius");

  childName = args.value<string>("ChildName");
  rotName = args.value<string>("RotationName");
  flagString = args.value<string>("FlagString");

  LogDebug("TrackerGeom") << "DDPixFwdDiskAlgo debug: Parent " << mother.name()
                          << "\tChild " << childName << " NameSpace "
                          << ns.name() << "\tRot Name " << rotName
                          << "\tCopyNo (Start/Total) " << startCopyNo << ", "
                          << nBlades << "\tAngles " << ConvertTo(bladeAngle,deg)
                          << ", " << ConvertTo(bladeTilt,deg) << "\tZshifts "
                          << zPlane << "\tAnchor Radius " << anchorR;

    for (int iBlade=0; iBlade<nBlades; ++iBlade) {
    LogDebug("TrackerGeom") << "DDPixFwdDiskAlgo: Blade " << iBlade
                            << " flag " << flagString[iBlade] << " zshift "
                            << bladeZShift[iBlade];
    }
    double deltaPhi = 360.0_deg/(double)nBlades;
    int copyNo = startCopyNo;
    string flagSelector = "Y";

    for (int iBlade=0; iBlade<nBlades; ++iBlade) {
      if (flagString[iBlade] == flagSelector[0]) {
        string rotstr = rotName[0] + to_string(double(copyNo));
        double phi  = (iBlade+0.5)*deltaPhi;
        double phix = atan2(sin(phi)*cos(bladeAngle),
                                 cos(phi)*cos(bladeAngle));
        double thetx= acos(-sin(bladeAngle));
        double phiy = atan2((cos(phi)*cos(bladeTilt)+sin(phi)
                                  *sin(bladeAngle)*sin(bladeTilt)),
                                 (-sin(phi)*cos(bladeTilt)+cos(phi)
                                  *sin(bladeAngle)*sin(bladeTilt)));

        double thety= acos(cos(bladeAngle)*sin(bladeTilt));
        double phiz = atan2((-cos(phi)*sin(bladeTilt)+sin(phi)
                             *sin(bladeAngle)*cos(bladeTilt)),
                            (sin(phi)*sin(bladeTilt)+cos(phi)
                             *sin(bladeAngle)*cos(bladeTilt)));

        double thetz= acos(cos(bladeAngle)*cos(bladeTilt));
        Rotation3D rot = Rotation3D();

        auto irot = ctxt.rotations.find( ns.prepend( rotstr ) );

        if ( irot != ctxt.rotations.end() ) {
          LogDebug("TrackerGeom") << "DDPixFwdDiskAlgo test: Creating a new "
                                  << "rotation: " << rotstr << "\t"
                                  << ConvertTo(thetx,deg) << ", " << ConvertTo(phix,deg)
                                  << ", " << ConvertTo(thety,deg) << ", "
                                  << ConvertTo(phiy,deg) << ", " << ConvertTo(thetz,deg)
                                  << ", " << ConvertTo(phiz,deg);

          LogDebug("TrackerGeom") << "Rotation Matrix (" << ConvertTo(phi,deg) << ", "
                                  << ConvertTo(bladeAngle,deg)
                                  << ", " << ConvertTo(bladeTilt,deg) << ") "
                                  << cos(phi)*cos(bladeAngle)
                                  << ", "<< (-sin(phi)*cos(bladeTilt)
                                      +cos(phi)*sin(bladeAngle)*sin(bladeTilt))
                                  << ", " << (sin(phi)*sin(bladeTilt)+cos(phi)
                                              *sin(bladeAngle)*cos(bladeTilt))
                                  << ", " << sin(phi)*cos(bladeAngle) << ", "
                                  << (cos(phi)*cos(bladeTilt)+sin(phi)
                                      *sin(bladeAngle)*sin(bladeTilt))
                                  << ", " << (-cos(phi)*sin(bladeTilt)+sin(phi)
                                              *sin(bladeAngle)*cos(bladeTilt))
                                  << ", " << -sin(bladeAngle) << ", "
                                  << cos(bladeAngle)*sin(bladeTilt)
                                  << ", " << cos(bladeAngle)*cos(bladeTilt);
          rot = makeRotation3D(thetx, phix, thety, phiy, thetz, phiz);
	}
        double xpos = anchorR*(-sin(phi)*cos(bladeTilt)+cos(phi)
                               *sin(bladeAngle)*sin(bladeTilt));
        double ypos = anchorR*(cos(phi)*cos(bladeTilt)+sin(phi)
                               *sin(bladeAngle)*sin(bladeTilt));
        double zpos = anchorR*(cos(bladeAngle)*sin(bladeTilt))+zPlane+
          bladeZShift[iBlade];

        Position tran(xpos, ypos, zpos);
        pv = mother.placeVolume(mother,copyNo,Transform3D(rot,tran));
        LogDebug("TrackerGeom") << "DDPixFwdDiskAlgo test: " << childName
                                << " number " << copyNo << " positioned in "
                                << mother << " at " << tran << " with " << rot;
      }
      copyNo++;
    }

    LogDebug("TrackerGeom") << "Finished....";
    return 1;

}
