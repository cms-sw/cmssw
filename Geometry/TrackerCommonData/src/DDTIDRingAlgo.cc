///////////////////////////////////////////////////////////////////////////////
// File: DDTIDRingAlgo.cc
// Description: Position n copies of detectors in alternate positions and
//              also associated ICC's and cool inserts
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
#include "Geometry/TrackerCommonData/interface/DDTIDRingAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTIDRingAlgo::DDTIDRingAlgo() {
  DCOUT('a', "DDTIDRingAlgo info: Creating an instance");
}

DDTIDRingAlgo::~DDTIDRingAlgo() {}

void DDTIDRingAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments & vsArgs) {

  idNameSpace        = DDCurrentNamespace::ns();
  moduleName         = vsArgs["ModuleName"]; 
  iccName            = sArgs["ICCName"]; 
  coolName           = sArgs["CoolName"]; 
  DDName parentName = parent().name();
  DCOUT('A', "DDTIDRingAlgo debug: Parent " << parentName << "\tModule " << moduleName[0] << ", " << moduleName[1] << "\tICC " << iccName << "\tCool " << coolName << "\tNameSpace " << idNameSpace);

  number            = int (nArgs["Number"]);
  startAngle        = nArgs["StartAngle"];
  rModule           = nArgs["ModuleR"];
  zModule           = vArgs["ModuleZ"];
  rICC              = nArgs["ICCR"];
  zICC              = vArgs["ICCZ"];
  rCool             = vArgs["CoolR"];
  zCool             = vArgs["CoolZ"];

  DCOUT('A', "DDTIDRingAlgo debug: Parameters for positioning--" << " StartAngle " << startAngle/deg << " Copy Numbers " << number << " Modules at R " << rModule << " Z " << zModule[0] << ", " << zModule[1] << " ICCs at R " << rICC << " Z " << zICC[0] << ", " << zICC[1] << " Cools at R " << rCool[0] << ", " << rCool[1] << " Z " << zCool[0] << ", " << zCool[1]);

  fullHeight        = nArgs["FullHeight"];
  dlTop             = nArgs["DlTop"];
  dlBottom          = nArgs["DlBottom"];
  dlHybrid          = nArgs["DlHybrid"];

  DCOUT('A', "DDTIDRingAlgo debug: Height " << fullHeight << " dl(Top) " << dlTop << " dl(Bottom) " << dlBottom << " dl(Hybrid) " << dlHybrid);

  topFrameHeight    = nArgs["TopFrameHeight"];
  bottomFrameHeight = nArgs["BottomFrameHeight"];
  bottomFrameOver   = nArgs["BottomFrameOver"];
  sideFrameWidth    = nArgs["SideFrameWidth"];
  sideFrameOver     = nArgs["SideFrameOver"];
  DCOUT('A', "DDTIDRingAlgo debug: Top Frame Height " << topFrameHeight	<< " Extra Height at Bottom " << bottomFrameHeight << " Overlap " << bottomFrameOver << " Side Frame Width " << sideFrameWidth << " Overlap " << sideFrameOver);

  hybridHeight      = nArgs["HybridHeight"];
  hybridWidth       = nArgs["HybridWidth"];
  coolWidth         = nArgs["CoolWidth"];
  coolSide          = int(nArgs["CoolSide"]);
  DCOUT('A', "DDTIDRingAlgo debug: Hybrid Height " << hybridHeight << " Width " << hybridWidth << " Cool Width " << coolWidth << " on sides " << coolSide);
}

void DDTIDRingAlgo::execute() {

  double theta = 90.*deg;
  double phiy  = 0.*deg;
  double dphi  = twopi/number;

  DDName mother = parent().name();
  DDName module;
  DDName icc(DDSplit(iccName).first, DDSplit(iccName).second);
  DDName cool(DDSplit(coolName).first, DDSplit(coolName).second);

  //R and Phi values for cooling pieces
  double rr[3], dy[3], fi[3];
  double dzdif = fullHeight + topFrameHeight;
  double topfr = bottomFrameHeight - bottomFrameOver;
  double dxbot, dxtop;
  if (dlHybrid > dlTop) {
    dxbot = 0.5*dlBottom + sideFrameWidth - sideFrameOver;
    dxtop = 0.5*dlHybrid + sideFrameWidth - sideFrameOver;
    dxbot = dxtop - (dxtop-dxbot)*(topfr+dzdif)/dzdif;
    rr[0] = rModule + 0.5*(dzdif+topfr) - 0.5*hybridHeight;
  } else {
    dxbot = 0.5*dlHybrid + sideFrameWidth - sideFrameOver;
    dxtop = 0.5*dlTop    + sideFrameWidth - sideFrameOver;
    dxtop = dxbot + (dxtop-dxbot)*(topfr+dzdif)/dzdif;
    rr[0] = rModule - 0.5*(dzdif+topfr) + 0.5*hybridHeight;
  }
  dy[0] = 0.5*(hybridWidth+coolWidth);
  fi[0] = 0;
  for (int i=0; i<2; i++) {
    rr[i+1] = rCool[i];
    dy[i+1] = 0.5*(dxbot+dxtop)-0.5*(sideFrameWidth-sideFrameOver)+
      (rr[i+1]-rModule)*(dxtop-dxbot)/(dzdif+topfr);
    fi[i+1] = atan((dlTop-dlBottom)/(2.*fullHeight));
  }
  DCOUT('A', "DDTIDRingAlgo:: dy Calc " << dxbot << " " << dxtop << " " << sideFrameWidth << " " << sideFrameOver << " " << (dxtop-dxbot) << " " << dzdif << " " << topfr << " " << rModule << " R " << rr[0] << " " << rr[1] << " " << rr[2] << " Phi " << fi[0]/deg << " " << fi[1]/deg << " " << fi[2]/deg);

  //Loop over modules
  int copy = 0;
  for (int i=0; i<number; i++) {

    //First the module
    double phiz = startAngle + i*dphi;
    double xpos = rModule*cos(phiz);
    double ypos = rModule*sin(phiz);
    double zpos, thetay, phix;
    if (i%2 == 0) {
      phix   = phiz + 90.*deg;
      thetay = 0*deg;
      zpos   = zModule[0];
      module = DDName(DDSplit(moduleName[0]).first, 
		      DDSplit(moduleName[0]).second);
    } else {
      phix   = phiz - 90.*deg;
      thetay = 180*deg;
      zpos   = zModule[1];
      module = DDName(DDSplit(moduleName[1]).first, 
		      DDSplit(moduleName[1]).second);
    }
    
    // stereo face inside toward structure, rphi face outside
    phix   = phix   - 180.*deg;
    thetay = thetay + 180.*deg;
    //
    
    DDTranslation trmod(xpos, ypos, zpos);
    double phideg = phiz/deg;
    DDRotation rotation;
    string rotstr = DDSplit(mother).first + dbl_to_string(phideg*10.);
    rotation = DDRotation(DDName(rotstr, idNameSpace));
    if (!rotation) {
      DCOUT('a', "DDTIDRingAlgo test: Creating a new rotation " << rotstr << "\t" << theta/deg << ", " << phix/deg << ", " << thetay/deg << ", " << phiy/deg << ", " << theta/deg << ", " << phiz/deg);
      rotation = DDrot(DDName(rotstr, idNameSpace), theta, phix, thetay, phiy,
		       theta, phiz);
    }
  
    DDpos (module, mother, i+1, trmod, rotation);
    DCOUT('a', "DDTIDRingAlgo test: " << module << " number " << i+1 << " positioned in " << mother << " at " << trmod << " with " << rotation);

    //Now the ICC
    xpos = rICC*cos(phiz);
    ypos = rICC*sin(phiz);
    if (i%2 == 0) zpos = zICC[0];
    else          zpos = zICC[1];
    DDTranslation tricc(xpos, ypos, zpos);
    DDpos (icc, mother, i+1, tricc, rotation);
    DCOUT('a', "DDTIDRingAlgo test: " << icc << " number " << i+1 << " positioned in " << mother << " at " << tricc << " with " << rotation);

    //and the Cooling inserts
    if (i%2 == 0) zpos = zCool[0];
    else          zpos = zCool[1];
    for (int j = 0; j < 2; j++) {
      int l0 = 2;
      if (j != 0)            l0 = 0;
      else if (coolSide > 1) l0 = 0;
      for (int l = l0; l < 2; l++) {
	double yy   = dy[j];
	if (l == 0) yy = -yy;
	double phi2 = phiz + (2*l-1)*fi[j];
        xpos = rr[j]*cos(phiz)-yy*sin(phiz);
	ypos = rr[j]*sin(phiz)+yy*cos(phiz);

	DDTranslation trcool(xpos, ypos, zpos);
	phideg = phi2/deg;
	DDRotation rotcool;
	if (phideg != 0) {
	  string rot = DDSplit(coolName).first+dbl_to_string(phideg*1000);
	  rotcool = DDRotation(DDName(rot, idNameSpace));
	  if (!rotcool) {
	    DCOUT('a', "DDTIDRingAlgo test: Creating a new rotation: " << rot << "\t90., " << phideg << ", 90.," << 90.+phideg << ", 0, 0");
	    rotcool = DDrot(DDName(rot, idNameSpace), 90.*deg, phi2, 
			    90.*deg, 90.*deg+phi2, 0., 0.);
	  }
	}
	copy++;
	DDpos (cool, mother, copy, trcool, rotcool);
	DCOUT('a', "DDTIDRingAlgo test: " << cool << " number " << copy << " positioned in " << mother << " at " << trcool << " with " << rotcool);
      }
    }
  }
}
