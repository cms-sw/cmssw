///////////////////////////////////////////////////////////////////////////////
// File: DDTECOptoHybAlgo.cc
// Description: Placing cooling pieces in the petal material of a TEC petal
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/plugins/DDTECOptoHybAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"


DDTECOptoHybAlgo::DDTECOptoHybAlgo(): angles(0) {
  LogDebug("TECGeom") << "DDTECOptoHybAlgo info: Creating an instance";
}

DDTECOptoHybAlgo::~DDTECOptoHybAlgo() {}

void DDTECOptoHybAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {
  DDCurrentNamespace ns;
  idNameSpace  = *ns;
  childName    = sArgs["ChildName"];

  DDName parentName = parent().name(); 

  LogDebug("TECGeom") << "DDTECOptoHybAlgo debug: Parent " << parentName 
		      << " Child " << childName << " NameSpace " <<idNameSpace;

  optoHeight     = nArgs["OptoHeight"];
  optoWidth      = nArgs["OptoWidth"];
  rpos           = nArgs["Rpos"];
  zpos           = nArgs["Zpos"];
  startCopyNo    = int (nArgs["StartCopyNo"]);
  angles         = vArgs["Angles"];

  LogDebug("TECGeom") << "DDTECOptoHybAlgo debug: Height of the Hybrid "
		      << optoHeight << " and Width " << optoWidth
		      <<"Rpos " << rpos << " Zpos " << zpos 
		      << " StartCopyNo " << startCopyNo << " Number " 
		      << angles.size();

  for (int i = 0; i < (int)(angles.size()); i++)
    LogDebug("TECGeom") << "\tAngles[" << i << "] = " << angles[i];

}

void DDTECOptoHybAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("TECGeom") << "==>> Constructing DDTECOptoHybAlgo...";

  DDName mother = parent().name();
  DDName child  = DDName(DDSplit(childName).first, DDSplit(childName).second);

  // given r positions are for the lower left corner
  rpos += optoHeight/2;
  int    copyNo = startCopyNo;
  for (double angle : angles) {
    double phix = -angle;
    // given phi positions are for the lower left corner
    phix += asin(optoWidth/2/rpos);
    double xpos = rpos * cos(phix);
    double ypos = rpos * sin(phix);
    DDTranslation tran(xpos, ypos, zpos);

    DDRotation rotation;
    double phiy = phix + 90.*CLHEP::deg;
    double phideg = phix/CLHEP::deg;
    if (phideg != 0) {
      std::string rotstr= DDSplit(childName).first + std::to_string(phideg*1000.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	double theta = 90.*CLHEP::deg;
	LogDebug("TECGeom") << "DDTECOptoHybAlgo test: Creating a new "
			    << "rotation: " << rotstr << "\t90., " 
			    << phix/CLHEP::deg << ", 90.," << phiy/CLHEP::deg
			    << ", 0, 0";
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
    }

   cpv.position(child, mother, copyNo, tran, rotation);
    LogDebug("TECGeom") << "DDTECOptoHybAlgo test " << child << " number " 
			<< copyNo << " positioned in " << mother << " at "
			<< tran  << " with " << rotation;
    copyNo++;
  }
  
  LogDebug("TECGeom") << "<<== End of DDTECOptoHybAlgo construction ...";
}
