///////////////////////////////////////////////////////////////////////////////
// File: DDPixPhase1FwdDiskAlgo.cc
// Description: Position n copies at given z-values
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDPixPhase1FwdDiskAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

DDPixPhase1FwdDiskAlgo::DDPixPhase1FwdDiskAlgo() {
  LogDebug("TrackerGeom") <<"DDPixPhase1FwdDiskAlgo info: Creating an instance";
}

DDPixPhase1FwdDiskAlgo::~DDPixPhase1FwdDiskAlgo() {}

void DDPixPhase1FwdDiskAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & vsArgs) {

  startCopyNo = int(nArgs["StartCopyNo"]);
  nBlades     = int(nArgs["NumberOfBlades"]);
  bladeAngle  = nArgs["BladeAngle"];
  bladeTilt   = -nArgs["BladeTilt"];
  zPlane      = nArgs["BladeCommonZ"];
  bladeZShift = vArgs["BladeZShift"];
  anchorR     = nArgs["AnchorRadius"];
 
  idNameSpace = DDCurrentNamespace::ns();
  childName   = sArgs["ChildName"]; 
  rotName     = sArgs["RotationName"]; 
  flagString  = sArgs["FlagString"];
  DDName parentName = parent().name();
  LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo debug: Parent " << parentName 
			  << "\tChild " << childName << " NameSpace " 
			  << idNameSpace << "\tRot Name " << rotName
			  << "\tCopyNo (Start/Total) " << startCopyNo << ", " 
			  << nBlades << "\tAngles " << bladeAngle/CLHEP::deg 
			  << ", " << bladeTilt/CLHEP::deg << "\tZshifts " 
			  << zPlane << "\tAmnchor Radius " << anchorR;

  for (int iBlade=0; iBlade<nBlades; ++iBlade) {
    LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo: Blade " << iBlade 
			    << " flag " << flagString[iBlade] << " zshift "
			    << bladeZShift[iBlade];
  }
}

void DDPixPhase1FwdDiskAlgo::execute(DDCompactView& cpv) {

  int    copy   = startCopyNo;
  DDName mother = parent().name();
  DDName child(DDSplit(childName).first, DDSplit(childName).second);
  std::string flagSelector = "Y";

  double deltaPhi   = (360./nBlades)*CLHEP::deg;
  std::string rotns = DDSplit(rotName).second;
  for (int iBlade=0; iBlade<nBlades; ++iBlade) {
	
    if (flagString[iBlade] == flagSelector[0]) {
      std::string rotstr = DDSplit(rotName).first + std::to_string(double(copy));

      double phi  = (iBlade+0.5)*deltaPhi;
      double phiy  = std::atan2(std::cos(phi), -std::sin(phi));
      double thety = std::acos(std::sin(bladeTilt));
      double phix = std::atan2(std::cos(bladeAngle)*std::sin(phi) +
                               std::cos(phi)*std::sin(bladeTilt)*std::sin(bladeAngle), 
                               std::cos(phi)*std::cos(bladeAngle) - 
                               std::sin(phi)*std::sin(bladeTilt)*std::sin(bladeAngle));
      double thetx = std::acos(-std::cos(bladeTilt) * std::sin(bladeAngle));
      double phiz = std::atan2(std::sin(phi)*std::sin(bladeAngle) -
                               std::cos(phi)*std::cos(bladeAngle)*std::sin(bladeTilt), 
                               std::cos(phi)*std::sin(bladeAngle) +
                               std::cos(bladeAngle)*std::sin(phi)*std::sin(bladeTilt));
      double thetz = std::acos(std::cos(bladeTilt) * std::cos(bladeAngle));
      DDRotation rot = DDRotation(DDName(rotstr, rotns));
      if (!rot) {
	LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo test: Creating a new "
				<< "rotation: " << rotstr << "\t" 
				<< thetx/CLHEP::deg << ", " << phix/CLHEP::deg 
				<< ", " << thety/CLHEP::deg << ", " 
				<< phiy/CLHEP::deg << ", " << thetz/CLHEP::deg
				<< ", " << phiz/CLHEP::deg;
	LogDebug("TrackerGeom") << "Rotation Matrix (" << phi/CLHEP::deg << ", " << bladeAngle/CLHEP::deg << ", " << bladeTilt/CLHEP::deg << ") " << std::cos(phi)*std::cos(bladeAngle) << ", " << (-std::sin(phi)*std::cos(bladeTilt)+std::cos(phi)*std::sin(bladeAngle)*std::sin(bladeTilt)) << ", " << (std::sin(phi)*std::sin(bladeTilt)+std::cos(phi)*std::sin(bladeAngle)*std::cos(bladeTilt)) << ", " << std::sin(phi)*std::cos(bladeAngle) << ", " << (std::cos(phi)*std::cos(bladeTilt)+std::sin(phi)*std::sin(bladeAngle)*std::sin(bladeTilt)) << ", " << (-std::cos(phi)*std::sin(bladeTilt)+std::sin(phi)*std::sin(bladeAngle)*std::cos(bladeTilt)) << ", " << -std::sin(bladeAngle) << ", " << std::cos(bladeAngle)*std::sin(bladeTilt) << ", " << std::cos(bladeAngle)*std::cos(bladeTilt);
	rot = DDrot(DDName(rotstr, rotns), thetx,phix, thety,phiy, thetz,phiz);
      }
      double xpos = -anchorR * std::sin(phi);
      double ypos = anchorR * std::cos(phi);
      double zpos = zPlane + bladeZShift[iBlade % nBlades];
      DDTranslation tran(xpos, ypos, zpos);
      cpv.position (child, mother, copy, tran, rot);
      LogDebug("TrackerGeom") << "DDPixPhase1FwdDiskAlgo test: " << child 
			      << " number " << copy << " positioned in "
			      << mother << " at " << tran << " with " << rot;
    }
    copy++;
  }
}
