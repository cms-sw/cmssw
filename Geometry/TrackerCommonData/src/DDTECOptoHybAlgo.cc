///////////////////////////////////////////////////////////////////////////////
// File: DDTECOptoHybAlgo.cc
// Description: Placing cooling pieces in the petal material of a TEC petal
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;
#include "DetectorDescription/Parser/interface/DDLParser.h"
#include "DetectorDescription/Base/interface/DDdebug.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTECOptoHybAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTECOptoHybAlgo::DDTECOptoHybAlgo(): angles(0) {
  DCOUT('a', "DDTECOptoHybAlgo info: Creating an instance");
}

DDTECOptoHybAlgo::~DDTECOptoHybAlgo() {}

void DDTECOptoHybAlgo::initialize(const DDNumericArguments & nArgs,
				  const DDVectorArguments & vArgs,
				  const DDMapArguments & ,
				  const DDStringArguments & sArgs,
				  const DDStringVectorArguments & ) {

  idNameSpace  = DDCurrentNamespace::ns();
  childName    = sArgs["ChildName"];

  DDName parentName = parent().name(); 

  DCOUT('A', "DDTECOptoHybAlgo debug: Parent " << parentName << " Child " << childName << " NameSpace " << idNameSpace);

  rmin           = nArgs["Rmin"];
  rmax           = nArgs["Rmax"];
  zpos           = nArgs["Zpos"];
  startCopyNo    = int (nArgs["StartCopyNo"]);
  angles         = vArgs["Angles"];

  DCOUT('A', "DDTECOptoHybAlgo debug: Rmin " << rmin << " Rmax " << rmax << " Zpos " << zpos << " StartCopyNo " << startCopyNo << " Number " << angles.size());

  for (unsigned int i = 0; i < angles.size(); i++)
    DCOUT('A', "\t " << i << " " << angles[i]);

}

void DDTECOptoHybAlgo::execute() {
  
  DCOUT('a', "==>> Constructing DDTECOptoHybAlgo...");

  DDName mother = parent().name(); 
  DDName child  = DDName(DDSplit(childName).first, DDSplit(childName).second);

  int    copyNo = startCopyNo;
  double rpos   = 0.5*(rmin+rmax);
  for (unsigned int i = 0; i < angles.size(); i++) {
    double phix = angles[i];
    double xpos = rpos * cos(phix);
    double ypos = rpos * sin(phix);
    DDTranslation tran(xpos, ypos, zpos);

    DDRotation rotation;
    double phiy = phix + 90.*deg;
    double phideg = phix/deg;
    if (phideg != 0) {
      string rotstr = DDSplit(childName).first + dbl_to_string(phideg*1000.);
      rotation = DDRotation(DDName(rotstr, idNameSpace));
      if (!rotation) {
	double theta = 90.*deg;
	DCOUT('a', "DDTECOptoHybAlgo test: Creating a new rotation: " << rotstr << "\t90., " << phix/deg << ", 90.," << phiy/deg << ", 0, 0");
	rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, theta, phiy,
			 0., 0.);
      }
    }

    DDpos (child, mother, copyNo, tran, rotation);
    DCOUT('a', "DDTECOptoHybAlgo test " << child << " number " << copyNo << " positioned in " << mother << " at " << tran  << " with " << rotation);
    copyNo++;
  }
  
  DCOUT('a', "<<== End of DDTECOptoHybAlgo construction ...");
}
