///////////////////////////////////////////////////////////////////////////////
// File: DDTECCoolAlgo.cc
// Description: Placing cooling pieces in the petal material of a TEC petal
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDPosPart.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/TrackerCommonData/interface/DDTECCoolAlgo.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"


DDTECCoolAlgo::DDTECCoolAlgo(): coolName(0),coolR(0),coolInsert(0) {
  LogDebug("TECGeom") << "DDTECCoolAlgo info: Creating an instance";
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
  LogDebug("TECGeom") << "DDTECCoolAlgo debug: Parent " << parentName 
		      <<" NameSpace " << idNameSpace << " with " 
		      << petalName.size() << " possible petals:";
  for (int i=0; i<(int)(petalName.size()); i++)
    LogDebug("TECGeom") << "DDTECCoolAlgo debug: Petal[" << i << "]: "
			<< petalName[i] << " R " << petalRmax[i] << " W " 
			<< petalWidth[i]/deg;
  coolName       = vsArgs["CoolName"];
  coolR          = vArgs["CoolR"];
  coolInsert     = dbl_to_int (vArgs["CoolInsert"]);

  LogDebug("TECGeom") << "DDTECCoolAlgo debug: Start Copy Number " 
		      << startCopyNo << " with " << coolName.size() 
		      << " possible Cool pieces for " << coolInsert.size() 
		      << " modules";
  for (int i=0; i<(int)(coolName.size()); i++)
    LogDebug("TECGeom") << "                   Piece " << i << " " 
			<< coolName[i] << " R = " << coolR[i];
  for (int i=0; i<(int)(coolInsert.size()); i++)
    LogDebug("TECGeom") << "DDTECCoolAlgo debug: Inserts for module " << i
			<< ": " << coolInsert[i];
  startAngle     = nArgs["StartAngle"];
  incrAngle      = nArgs["IncrAngle"];
  LogDebug("TECGeom") << "DDTECCoolAlgo debug: StartAngle " 
		      << startAngle/deg << " IncrementAngle " << incrAngle/deg;
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

  LogDebug("TECGeom") << "DDTECCoolAlgo debug: Rmin " << rmin  
		      << " FullHeight " << fullHeight << " DlTop " << dlTop
		      << " DlBottom " << dlBottom << " DlHybrid " << dlHybrid 
		      << " Top Frame Height " << topFrameHeight << " Width "
		      << frameWidth << " Overlap "  << frameOver 
		      << " Hybrid Height " << hybridHeight << " Width " 
		      << hybridWidth;
}

void DDTECCoolAlgo::execute() {

  LogDebug("TECGeom") << "==>> Constructing DDTECCoolAlgo...";
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
  for (int i = 0; i < (int)(coolInsert.size()); i++) {
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
	if (phi1 < 0) phi1 = -phi1;
	LogDebug("TECGeom") << "DDTECCoolAlgo debug: kk " << kk << " R " 
			    << rp << " DY " << yy << ", " << dy << " X, Y "
			    << xpsl << ", " << ypsl;
	double phiMax = 5.;
	DDName mother = parent().name();
	int mm = (int)(petalName.size());
	for (int ii=(int)(petalName.size()); ii>0; ii--) 
	  if (rp < petalRmax[ii-1]) mm = ii-1;
	if (mm < (int)(petalName.size()) && phi1 <= 0.5*petalWidth[mm]) {
	  mother = DDName(DDSplit(petalName[mm]).first, 
			  DDSplit(petalName[mm]).second);
	  phiMax = 0.5*petalWidth[mm];
	}
	double xpos = rp*cos(phi)-yy*sin(phi);
	double ypos = rp*sin(phi)+yy*cos(phi);
	double phi2;
	if (ypos >= 0) phi2 = atan2(ypos+coolR[cool],xpos);
	else           phi2 = atan2(ypos-coolR[cool],xpos);
	double rr1 = sqrt(xpos*xpos+ypos*ypos)-coolR[cool];
	double rr2 = sqrt(xpos*xpos+ypos*ypos)+coolR[cool];
	if (std::abs(phi2) <= phiMax) {
	  DDTranslation tran(xpos, ypos, 0.0);
	  DDRotation rotation;
	  DDpos (child, mother, copyNo, tran, rotation);
	  LogDebug("TECGeom") << "DDTECCoolAlgo test " << child << "["  
			      << copyNo << "] positioned in " << mother 
			      << " at " << tran  << " with " << rotation 
			      << " phi (" << phi1/deg << ":" << phi2/deg 
			      << ") Limit" << phiMax/deg << " R " << rr1 
			      << ":" << rr2 << " (" << petalRmax[mm] << ")";
	} else {
	  edm::LogInfo("TECGeom") << "Error in positioning Cool Tube : Phi " 
				  << phi1/deg  << ":" << phi2/deg << " (Limit "
				  << phiMax/deg << " R " << rr1 << ":" << rr2 
				  << " (Limit " << petalRmax[mm] << ")";
	}
	copyNo++;
      }
    }
    phi += incrAngle;
  }

  LogDebug("TECGeom") << "<<== End of DDTECCoolAlgo construction ...";
}
