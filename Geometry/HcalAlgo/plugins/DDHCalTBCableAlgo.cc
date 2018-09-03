///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTBCableAlgo.cc
// Description: Cable mockup between barrel and endcap gap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

namespace std{} using namespace std;

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBCableAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHCalTBCableAlgo::DDHCalTBCableAlgo(): theta(0),rmax(0),zoff(0) {
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo info: Creating an instance";
}

DDHCalTBCableAlgo::~DDHCalTBCableAlgo() {}


void DDHCalTBCableAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments & ) {

  genMat      = sArgs["MaterialName"];
  nsectors    = int (nArgs["NSector"]);
  nsectortot  = int (nArgs["NSectorTot"]);
  nhalf       = int (nArgs["NHalf"]);
  rin         = nArgs["RIn"];
  theta       = vArgs["Theta"];
  rmax        = vArgs["RMax"];
  zoff        = vArgs["ZOff"];

  absMat      = sArgs["AbsMatName"];
  thick       = nArgs["Thickness"];
  width1      = nArgs["Width1"];
  length1     = nArgs["Length1"];
  width2      = nArgs["Width2"];
  length2     = nArgs["Length2"];
  gap2        = nArgs["Gap2"];

  LogDebug("HCalGeom") << "DDHCalTBCableAlgo debug: General material " 
		       << genMat << "\tSectors "  << nsectors << ", " 
		       << nsectortot << "\tHalves " << nhalf << "\tRin " <<rin;
  for (unsigned int i = 0; i < theta.size(); i++)
    LogDebug("HCalGeom") << "\t" << i << " Theta " << theta[i] << " rmax " 
			 << rmax[i] << " zoff " << zoff[i];
  LogDebug("HCalGeom") << "\tCable mockup made of " << absMat << "\tThick " 
		       << thick << "\tLength and width " << length1 << ", "
		       << width1 <<" and "	<< length2 << ", " << width2 
		       << " Gap " << gap2;

  idName      = sArgs["MotherName"];
  DDCurrentNamespace ns;
  idNameSpace = *ns;
  rotns       = sArgs["RotNameSpace"];
  DDName parentName = parent().name(); 
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo debug: Parent " << parentName
		       << " idName " << idName << " NameSpace " << idNameSpace
		       << " for solids etc. and " << rotns << " for rotations";
}

void DDHCalTBCableAlgo::execute(DDCompactView& cpv) {
  
  LogDebug("HCalGeom") << "==>> Constructing DDHCalTBCableAlgo...";
  unsigned int i=0;

  double alpha = CLHEP::pi/nsectors;
  double dphi  = nsectortot*CLHEP::twopi/nsectors;

  double zstep0 = zoff[1]+rmax[1]*tan(theta[1])+(rin-rmax[1])*tan(theta[2]);
  double zstep1 = zstep0+thick/cos(theta[2]);
  double zstep2 = zoff[3];
 
  double rstep0 = rin + (zstep2-zstep1)/tan(theta[2]);
  double rstep1 = rin + (zstep1-zstep0)/tan(theta[2]);

  vector<double> pgonZ;
  pgonZ.emplace_back(zstep0); 
  pgonZ.emplace_back(zstep1);
  pgonZ.emplace_back(zstep2); 
  pgonZ.emplace_back(zstep2+thick/cos(theta[2])); 

  vector<double> pgonRmin;
  pgonRmin.emplace_back(rin); 
  pgonRmin.emplace_back(rin);
  pgonRmin.emplace_back(rstep0); 
  pgonRmin.emplace_back(rmax[2]); 

  vector<double> pgonRmax;
  pgonRmax.emplace_back(rin); 
  pgonRmax.emplace_back(rstep1); 
  pgonRmax.emplace_back(rmax[2]); 
  pgonRmax.emplace_back(rmax[2]); 

  string name("Null");
  DDSolid solid;
  solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				    nsectortot, -alpha, dphi, pgonZ, 
				    pgonRmin, pgonRmax);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " 
		       << DDName(idName,idNameSpace) << " Polyhedra made of " 
		       << genMat << " with " << nsectortot << " sectors from "
		       << -alpha/CLHEP::deg << " to " 
		       << (-alpha+dphi)/CLHEP::deg << " and with " 
		       << pgonZ.size() << " sections";
  for (i = 0; i <pgonZ.size(); i++) 
    LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " 
			 << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
  
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second); 
  DDMaterial matter(matname);
  DDLogicalPart genlogic(solid.ddname(), matter, solid);

  DDName parentName = parent().name(); 
  DDTranslation r0(0.0, 0.0, 0.0);
  DDRotation rot;
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " 
		       << DDName(idName,idNameSpace) << " number 1 positioned "
		       << "in " << parentName << " at " << r0 << " with "<<rot;
  
  if (nhalf != 1) {
    rot = DDRotation(DDName("180D", rotns));
   cpv.position(DDName(idName, idNameSpace), parentName, 2, r0, rot);
    LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " 
			 << DDName(idName,idNameSpace) <<" number 2 positioned"
			 << "in " << parentName  << " at " << r0 << " with "
			 << rot;
  } 
  
  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " 
		       << DDName(name,idNameSpace) << " Polyhedra made of " 
		       << genMat << " with 1 sector from " <<-alpha/CLHEP::deg 
		       << " to " << alpha/CLHEP::deg << " and with " 
		       << pgonZ.size() << " sections";
  for (i = 0; i < pgonZ.size(); i++) 
    LogDebug("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " 
			 << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      1, -alpha, 2*alpha, pgonZ,
				      pgonRmin, pgonRmax);
  DDLogicalPart seclogic(solid.ddname(), matter, solid);
  
  for (int ii=0; ii<nsectortot; ii++) {
    double phi    = ii*2*alpha;
    double phideg = phi/CLHEP::deg;
    
    DDRotation rotation;
    string rotstr("NULL");
    if (phideg != 0) {
      rotstr = "R"; 
      if (phideg < 100)	rotstr = "R0"; 
      rotstr = rotstr + std::to_string(phideg);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
	LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: Creating a new "
			     << "rotation " << rotstr << "\t90," << phideg 
			     << ", 90, " << (phideg+90) << ", 0, 0";
	rotation = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg, 
			 phideg*CLHEP::deg, 90*CLHEP::deg, 
			 (90+phideg)*CLHEP::deg, 0*CLHEP::deg,  0*CLHEP::deg);
      }
    } 
  
   cpv.position(seclogic, genlogic, ii+1, DDTranslation(0.0, 0.0, 0.0), rotation);
    LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << seclogic.name() 
			 << " number " << ii+1 << " positioned in " 
			 << genlogic.name() << " at (0,0,0) with " << rotation;
  }
  
  //Now a trapezoid of air
  double rinl  = pgonRmin[0] + thick * sin(theta[2]);
  double routl = pgonRmax[2] - thick * sin(theta[2]);
  double dx1   = rinl * tan(alpha);
  double dx2   = 0.90 * routl * tan(alpha);
  double dy    = 0.50 * thick;
  double dz    = 0.50 * (routl -rinl);
  name  = idName + "Trap";
  solid = DDSolidFactory::trap(DDName(name, idNameSpace), dz, 0, 0, dy, dx1, 
			       dx1, 0, dy, dx2, dx2, 0);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << solid.name() 
		       <<" Trap made of " << genMat << " of dimensions " << dz 
		       << ", 0, 0, " << dy << ", " << dx1 << ", " << dx1 
		       << ", 0, " << dy << ", " << dx2 << ", "  << dx2 <<", 0";
  DDLogicalPart glog(solid.ddname(), matter, solid);

  string rotstr = name;
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: Creating a new rotation: " 
		       << rotstr << "\t90, 270, " << (180-theta[2]/CLHEP::deg) 
		       << ", 0, " << (90-theta[2]/CLHEP::deg) << ", 0";
  rot = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg, 270*CLHEP::deg, 
	      180*CLHEP::deg-theta[2], 0, 90*CLHEP::deg-theta[2], 0);
  DDTranslation r1(0.5*(rinl+routl), 0, 0.5*(pgonZ[1]+pgonZ[2]));
  cpv.position(glog, seclogic, 1, r1, rot);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << glog.name() 
		       << " number 1 positioned in " << seclogic.name() 
		       << " at " << r1 << " with " << rot;

  //Now the cable of type 1
  name = idName + "Cable1";
  double phi  = atan((dx2-dx1)/(2*dz));
  double xmid = 0.5*(dx1+dx2)-1.0;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), 0.5*width1,
			      0.5*thick, 0.5*length1);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << solid.name() 
		       << " Box made of " << absMat << " of dimension " 
		       << 0.5*width1 << ", " << 0.5*thick << ", " 
		       << 0.5*length1;
  DDName absname(DDSplit(absMat).first, DDSplit(absMat).second); 
  DDMaterial absmatter(absname);
  DDLogicalPart cablog1(solid.ddname(), absmatter, solid);

  rotstr = idName + "Left";
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: Creating a new rotation " 
		       << rotstr << "\t"  << (90+phi/CLHEP::deg) << "," << 0  
		       << "," << 90 << "," << 90 << "," << phi/CLHEP::deg 
		       << "," << 0;
  DDRotation rot2 = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg+phi, 0.0, 
			  90*CLHEP::deg, 90*CLHEP::deg, phi, 0.0);
  DDTranslation r2((xmid-0.5*width1*cos(phi)), 0, 0);
  cpv.position(cablog1, glog, 1, r2, rot2);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << cablog1.name() 
		       << " number 1 positioned in " << glog.name() << " at "
		       << r2	<< " with " << rot2;

  rotstr = idName + "Right";
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: Creating a new rotation " 
		       << rotstr << "\t"  << (90-phi/CLHEP::deg) 
		       << ", 0, 90, 90, " << -phi/CLHEP::deg << ", 0";
  DDRotation rot3 = DDrot(DDName(rotstr, idNameSpace), 90*CLHEP::deg-phi, 
			  0*CLHEP::deg, 90*CLHEP::deg, 90*CLHEP::deg,
			  -phi, 0*CLHEP::deg);
  DDTranslation r3(-(xmid-0.5*width1*cos(phi)), 0, 0);
  cpv.position(cablog1, glog, 2, r3, rot3);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << cablog1.name() 
		       << " number 2 positioned in "  << glog.name() << " at " 
		       << r3 << " with " << rot3;

  //Now the cable of type 2
  name = idName + "Cable2";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), 0.5*width2,
			      0.5*thick, 0.5*length2);
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << solid.name() 
		       << " Box made of " << absMat << " of dimension " 
		       << 0.5*width2 << ", " << 0.5*thick << ", "<<0.5*length2;
  DDLogicalPart cablog2(solid.ddname(), absmatter, solid);

  double xpos = 0.5*(width2+gap2);
 cpv.position(cablog2, glog, 1, DDTranslation(xpos, 0.0, 0.0), DDRotation());
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << cablog2.name() 
		       << " number 1 positioned in "  << glog.name() << " at ("
		       << xpos << ",  0, 0) with no rotation";
 cpv.position(cablog2, glog, 2, DDTranslation(-xpos, 0.0, 0.0), DDRotation());
  LogDebug("HCalGeom") << "DDHCalTBCableAlgo test: " << cablog2.name() 
		       << " number 2 positioned in "  << glog.name() << " at ("
		       <<-xpos << ", 0, 0) with no rotation";

  LogDebug("HCalGeom") << "<<== End of DDHCalTBCableAlgo construction ...";
}
