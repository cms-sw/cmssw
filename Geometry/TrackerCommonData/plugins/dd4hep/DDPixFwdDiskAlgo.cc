///////////////////////////////////////////////////////////////////////////////
// File: DDPixFwdDiskAlgo.cc
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
  bladeTilt = args.value<double>("BladeTilt");
  zPlane = args.value<double>("BladeCommonZ");
  anchorR = args.value<double>("AnchorRadius");

  bladeZShift = args.value<std::vector<double> >("BladeZShift");

  childName = args.value<std::string>("ChildName");
  rotName = args.value<std::string>("RotationName");
  flagString = args.value<std::string>("FlagString");

  if (strchr(childName.c_str(), NAMESPACE_SEP) == nullptr)
    childName = ns.prepend(childName);

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
      std::string rotstr = rotName[0] + std::to_string(double(copyNo));
      double phi = (iBlade + 0.5) * deltaPhi;
      double phix = atan2(sin(phi) * cos(bladeAngle), cos(phi) * cos(bladeAngle));
      double thetx = acos(-sin(bladeAngle));
      double phiy = atan2((cos(phi) * cos(bladeTilt) + sin(phi) * sin(bladeAngle) * sin(bladeTilt)),
                          (-sin(phi) * cos(bladeTilt) + cos(phi) * sin(bladeAngle) * sin(bladeTilt)));

      double thety = acos(cos(bladeAngle) * sin(bladeTilt));
      double phiz = atan2((-cos(phi) * sin(bladeTilt) + sin(phi) * sin(bladeAngle) * cos(bladeTilt)),
                          (sin(phi) * sin(bladeTilt) + cos(phi) * sin(bladeAngle) * cos(bladeTilt)));

      double thetz = acos(cos(bladeAngle) * cos(bladeTilt));

      auto rot = dd4hep::Rotation3D();

      auto irot = ctxt.rotations.find(ns.prepend(rotstr));

      if (irot != ctxt.rotations.end()) {
        edm::LogVerbatim("TrackerGeom") << "DDPixFwdDiskAlgo test: Creating a new "
                                        << "rotation: " << rotstr << "\t" << convertRadToDeg(thetx) << ", "
                                        << convertRadToDeg(phix) << ", " << convertRadToDeg(thety) << ", "
                                        << convertRadToDeg(phiy) << ", " << convertRadToDeg(thetz) << ", "
                                        << convertRadToDeg(phiz);

        edm::LogVerbatim("TrackerGeom") << "Rotation Matrix (" << convertRadToDeg(phi) << ", "
                                        << convertRadToDeg(bladeAngle) << ", " << convertRadToDeg(bladeTilt) << ") "
                                        << cos(phi) * cos(bladeAngle) << ", "
                                        << (-sin(phi) * cos(bladeTilt) + cos(phi) * sin(bladeAngle) * sin(bladeTilt))
                                        << ", "
                                        << (sin(phi) * sin(bladeTilt) + cos(phi) * sin(bladeAngle) * cos(bladeTilt))
                                        << ", " << sin(phi) * cos(bladeAngle) << ", "
                                        << (cos(phi) * cos(bladeTilt) + sin(phi) * sin(bladeAngle) * sin(bladeTilt))
                                        << ", "
                                        << (-cos(phi) * sin(bladeTilt) + sin(phi) * sin(bladeAngle) * cos(bladeTilt))
                                        << ", " << -sin(bladeAngle) << ", " << cos(bladeAngle) * sin(bladeTilt) << ", "
                                        << cos(bladeAngle) * cos(bladeTilt);
        rot = cms::makeRotation3D(thetx, phix, thety, phiy, thetz, phiz);
      }
      double xpos = anchorR * (-sin(phi) * cos(bladeTilt) + cos(phi) * sin(bladeAngle) * sin(bladeTilt));
      double ypos = anchorR * (cos(phi) * cos(bladeTilt) + sin(phi) * sin(bladeAngle) * sin(bladeTilt));
      double zpos = anchorR * (cos(bladeAngle) * sin(bladeTilt)) + zPlane + bladeZShift[iBlade];

      dd4hep::Position tran(xpos, ypos, zpos);
      pv = mother.placeVolume(child, copyNo, dd4hep::Transform3D(rot, tran));
      edm::LogVerbatim("TrackerGeom") << "DDPixFwdDiskAlgo test: " << pv.name() << ": " << childName << " number "
                                      << copyNo << " positioned in " << mother.name() << " at " << tran << " with "
                                      << rot;
    }
    copyNo++;
  }

  edm::LogVerbatim("TrackerGeom") << "Finished....";
  return cms::s_executed;
}

DECLARE_DDCMS_DETELEMENT(DDCMS_track_DDPixFwdDiskAlgo, algorithm)
