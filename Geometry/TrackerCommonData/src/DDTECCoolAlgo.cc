///////////////////////////////////////////////////////////////////////////////
// File: DDTECCoolAlgo.cc
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
#include "Geometry/TrackerCommonData/interface/DDTECCoolAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTECCoolAlgo::DDTECCoolAlgo(): coolName(0),coolR(0),coolInsert(0) {
  DCOUT('a', "DDTECCoolAlgo info: Creating an instance");
}

DDTECCoolAlgo::~DDTECCoolAlgo() {}

void DDTECCoolAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & ,
			       const DDStringVectorArguments & vsArgs) {

  idNameSpace    = DDCurrentNamespace::ns();
  startCopyNo    = int(nArgs["StartCopyNo"]);

  DDName parentName = parent().name(); 
  petalName      = vsArgs["PetalName"];
  petalRmax      = vArgs["PetalR"];
  petalWidth     = vArgs["PetalWidth"];
  DCOUT('A', "DDTECCoolAlgo debug: Parent " << parentName <<" NameSpace " << idNameSpace << " with " << petalName.size() << " possible petals:");
  for (unsigned int i=0; i<petalName.size(); i++)
    DCOUT('A', "\t " << petalName[i] << " R " << petalRmax[i] << " W " << petalWidth[i]/deg);
  coolName       = vsArgs["CoolName"];
  coolR          = vArgs["CoolR"];
  coolInsert     = dbl_to_int (vArgs["CoolInsert"]);

  DCOUT('A', "DDTECCoolAlgo debug: Start Copy Number " << startCopyNo << " with " << coolName.size() << " possible Cool pieces for " << coolInsert.size() << " modules");
  for (unsigned int i=0; i<coolName.size(); i++)
    DCOUT('A', "                   Piece " << i << " " << coolName[i] << " R = " << coolR[i]);
  DCOUT('A', "DDTECCoolAlgo debug: Inserts for modules:");
  for (unsigned int i=0; i<coolInsert.size(); i++)
    DCOUT('A', "\t " << coolInsert[i]);

  startAngle     = nArgs["StartAngle"];
  incrAngle      = nArgs["IncrAngle"];
  DCOUT('A', "DDTECCoolAlgo debug: StartAngle " << startAngle/deg << " IncrementAngle " << incrAngle/deg);
  rmin           = nArgs["Rmin"];
  fullHeight     = nArgs["FullHeight"];
  dlTop          = nArgs["DlTop"];
  dlBottom       = nArgs["DlBottom"];
  dlHybrid       = nArgs["DlHybrid"];
  topFrameHeight = nArgs["TopFrameHeight"];
  frameWidth     = nArgs["FrameWidth"];
  frameOver      = nArgs["FrameOver"];
  hybridHeight   = nArgs["HybridHeight"];
  hybridWidth    = nArgs["HybridWidth"];

  DCOUT('A', "DDTECCoolAlgo debug: Rmin " << rmin  << " FullHeight " << fullHeight << " DlTop " << dlTop << " DlBottom " << dlBottom << " DlHybrid " << dlHybrid << " Top Frame Height " << topFrameHeight << " Width " << frameWidth << " Overlap " << frameOver <<" Hybrid Height " << hybridHeight << " Width " << hybridWidth);
}

void DDTECCoolAlgo::execute() {
  DCOUT('a', "==>> Constructing DDTECCoolAlgo...");
  int copyNo  = startCopyNo;
  double phi  = startAngle;
  double rr[4];
  if (dlHybrid > dlTop) {
    rr[0] = rmin + 0.5*fullHeight;
    rr[2] = rmin + fullHeight + topFrameHeight - 0.5*hybridHeight;
  } else {
    rr[0] = rmin + 0.5*fullHeight + topFrameHeight;
    rr[2] = rmin + 0.5*hybridHeight;
  }
  rr[1] = rr[0];
  rr[3] = rr[2];
  for (unsigned int i = 0; i < coolInsert.size(); i++) {
    int nd = coolInsert[i];
    for (int kk = 0; kk < 4; kk++) {
      int cool = nd%10 - 1;
      nd      /= 10;
      if (cool >= 0 && cool <= int(coolName.size())) {
	DDName child  = DDName(DDSplit(coolName[cool]).first, 
			       DDSplit(coolName[cool]).second);
	double yy, rp=rr[kk];
	if (kk > 1) yy = 0.5*hybridWidth + 0.5*coolR[cool];
	else        yy = 0.25*(dlBottom+dlTop) + 0.5*frameWidth - frameOver;
	double dy = yy + 0.5*coolR[cool];
	if (kk%2 == 0) {
	  dy = -dy;
	  yy = -yy;
	}
	double xpsl = rp*cos(phi)-dy*sin(phi);
	double ypsl = rp*sin(phi)+dy*cos(phi);
	double phi1 = atan2(ypsl,xpsl);
	DCOUT('a', "DDTECCoolAlgo debug: kk " << kk << " R " << rp << " DY " << yy << ", " << dy << " X, Y " << xpsl << ", " << ypsl);
	DDName mother;
	unsigned int mm = petalName.size();
	for (unsigned int ii=petalName.size(); ii>0; ii--)
	  if (rp < petalRmax[ii-1]) mm = ii-1;
	if (mm < petalName.size() && abs(phi1) <= 0.5*petalWidth[mm]) {
	  mother = DDName(DDSplit(petalName[mm]).first, 
			  DDSplit(petalName[mm]).second);
	} else {
	  mother = parent().name(); 
	}
	double xpos = rp*cos(phi)-yy*sin(phi);
	double ypos = rp*sin(phi)+yy*cos(phi);
	DDTranslation tran(xpos, ypos, 0.0);
	DDRotation rotation;
	DDpos (child, mother, copyNo, tran, rotation);
	DCOUT('a', "DDTECCoolAlgo test " << child << " number " << copyNo << " positioned in " << mother << " at " << tran  << " with " << rotation << " phi " << phi1/deg);
	copyNo++;
      }
    }
    phi += incrAngle;
  }

  DCOUT('a', "<<== End of DDTECCoolAlgo construction ...");
}
