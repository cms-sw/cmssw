///////////////////////////////////////////////////////////////////////////////
// File: DDPixPhase1FwdDiskAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/DDCMS/interface/DDPlugins.h"
#include "DD4hep/DetFactoryHelper.h"
#include "DD4hep/Printout.h"

#include "DataFormats/Math/interface/CMSUnits.h"

#include <sstream>

using namespace cms_units::operators;  // _deg and convertRadToDeg

static long algorithm(dd4hep::Detector& /* description */, cms::DDParsingContext& ctxt, xml_h e) {
  cms::DDNamespace ns(ctxt, e, true);
  cms::DDAlgoArguments args(ctxt, e);

  int nBlades;                      //Number of blades
  int startCopyNo;                  //Start Copy number
  double bladeAngle;                //Angle of blade rotation aroung y-axis
  double bladeTilt;                 //Tilt of the blade around x-axis
  double zPlane;                    //Common shift in z for all blades
  std::vector<double> bladeZShift;  //Shift in Z of individual blades
  double anchorR;                   //Distance of beam line to anchor point

  std::string childName;   //Child name
  std::string rotName;     //Name of the base rotation matrix
  std::string flagString;  //Flag if a blade is present

  dd4hep::Volume mother = ns.volume(args.parentName());
  dd4hep::PlacedVolume pv;

  startCopyNo = args.find("StartCopyNo") ? args.value<int>("StartCopyNo") : 0;
  nBlades = args.value<int>("NumberOfBlades");
  bladeAngle = args.value<double>("BladeAngle");
  bladeTilt = -1. * args.value<double>("BladeTilt");
  zPlane = args.value<double>("BladeCommonZ");
  anchorR = args.value<double>("AnchorRadius");

  bladeZShift = args.value<std::vector<double> >("BladeZShift");

  childName = args.value<std::string>("ChildName");
  rotName = args.value<std::string>("RotationName");
  flagString = args.value<std::string>("FlagString");

  dd4hep::Volume child = ns.volume(childName);

  edm::LogVerbatim("TrackerGeom") << "DDPixFwdDiskAlgo debug: Parent " << mother.name() << "\tChild " << child.name()
                                  << " NameSpace " << ns.name() << "\tRot Name " << rotName << "\tCopyNo (Start/Total) "
                                  << startCopyNo << ", " << nBlades << "\tAngles " << convertRadToDeg(bladeAngle)
                                  << ", " << convertRadToDeg(bladeTilt) << "\tZshifts " << zPlane << "\tAnchor Radius "
                                  << anchorR;

  for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
    edm::LogVerbatim("TrackerGeom") << "DDPixFwdDiskAlgo: Blade " << iBlade << " flag " << flagString[iBlade]
                                    << " zshift " << bladeZShift[iBlade];
  }

  double deltaPhi = 360.0_deg / (double)nBlades;
  int copyNo = startCopyNo;
  std::string flagSelector = "Y";

  for (int iBlade = 0; iBlade < nBlades; ++iBlade) {
    if (flagString[iBlade] == flagSelector[0]) {
      double phi = (iBlade + 0.5) * deltaPhi;
      double phiy = atan2(cos(phi), -sin(phi));
      double thety = acos(sin(bladeTilt));
      double phix = atan2(cos(bladeAngle) * sin(phi) + cos(phi) * sin(bladeTilt) * sin(bladeAngle),
                          cos(phi) * cos(bladeAngle) - sin(phi) * sin(bladeTilt) * sin(bladeAngle));
      double thetx = acos(-cos(bladeTilt) * sin(bladeAngle));
      double phiz = atan2(sin(phi) * sin(bladeAngle) - cos(phi) * cos(bladeAngle) * sin(bladeTilt),
                          cos(phi) * sin(bladeAngle) + cos(bladeAngle) * sin(phi) * sin(bladeTilt));
      double thetz = acos(cos(bladeTilt) * cos(bladeAngle));

      dd4hep::Rotation3D rot = cms::makeRotation3D(thetx, phix, thety, phiy, thetz, phiz);

      double xpos = -anchorR * sin(phi);
      double ypos = anchorR * cos(phi);
      double zpos = zPlane + bladeZShift[iBlade % nBlades];

      dd4hep::Position tran(xpos, ypos, zpos);
      pv = mother.placeVolume(child, copyNo, dd4hep::Transform3D(rot, tran));
    }
    copyNo++;
  }
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDPixPhase1FwdDiskAlgo, algorithm)
