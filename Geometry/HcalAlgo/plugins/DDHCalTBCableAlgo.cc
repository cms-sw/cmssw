///////////////////////////////////////////////////////////////////////////////
// File: DDHCalTBCableAlgo.cc
// Description: Cable mockup between barrel and endcap gap
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HcalAlgo/plugins/DDHCalTBCableAlgo.h"

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

DDHCalTBCableAlgo::DDHCalTBCableAlgo(): theta(0),rmax(0),zoff(0) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating an instance";
#endif
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

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: General material " 
			       << genMat << "\tSectors "  << nsectors << ", " 
			       << nsectortot << "\tHalves " << nhalf 
			       << "\tRin " << rin;
  for (unsigned int i = 0; i < theta.size(); i++)
    edm::LogVerbatim("HCalGeom") << "\t" << i << " Theta " << theta[i] 
				 << " rmax " << rmax[i] << " zoff " << zoff[i];
  edm::LogVerbatim("HCalGeom") << "\tCable mockup made of " << absMat
			       << "\tThick " << thick << "\tLength and width "
			       << length1 << ", " << width1 <<" and "
			       << length2 << ", " << width2 << " Gap " << gap2;
#endif
  idName      = sArgs["MotherName"];
  idNameSpace = DDCurrentNamespace::ns();
  rotns       = sArgs["RotNameSpace"];
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Parent " 
			       << parent().name() << " idName " << idName 
			       << " NameSpace " << idNameSpace
			       << " for solids etc. and " << rotns 
			       << " for rotations";
#endif
}

void DDHCalTBCableAlgo::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "==>> Constructing DDHCalTBCableAlgo...";
#endif

  double alpha = 1._pi/nsectors;
  double dphi  = nsectortot*2._pi/nsectors;
  
  double zstep0 = zoff[1]+rmax[1]*tan(theta[1])+(rin-rmax[1])*tan(theta[2]);
  double zstep1 = zstep0+thick/cos(theta[2]);
  double zstep2 = zoff[3];
 
  double rstep0 = rin + (zstep2-zstep1)/tan(theta[2]);
  double rstep1 = rin + (zstep1-zstep0)/tan(theta[2]);

  std::vector<double> pgonZ = {zstep0, zstep1, zstep2, 
			       zstep2+thick/cos(theta[2])}; 
  
  std::vector<double> pgonRmin = {rin, rin, rstep0, rmax[2]};
  std::vector<double> pgonRmax = {rin, rstep1, rmax[2], rmax[2]}; 

  std::string name("Null");
  DDSolid solid;
  solid = DDSolidFactory::polyhedra(DDName(idName, idNameSpace),
				    nsectortot, -alpha, dphi, pgonZ, 
				    pgonRmin, pgonRmax);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " 
			       << DDName(idName,idNameSpace) 
			       << " Polyhedra made of " << genMat << " with " 
			       << nsectortot << " sectors from "
			       << convertRadToDeg(-alpha) << " to " 
			       << convertRadToDeg(-alpha+dphi) << " and with " 
			       << pgonZ.size() << " sections";
  for (unsigned int i = 0; i <pgonZ.size(); i++) 
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " 
				 << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
#endif
  DDName matname(DDSplit(genMat).first, DDSplit(genMat).second); 
  DDMaterial matter(matname);
  DDLogicalPart genlogic(solid.ddname(), matter, solid);

  DDName parentName = parent().name(); 
  DDTranslation r0(0.0, 0.0, 0.0);
  DDRotation rot;
  cpv.position(DDName(idName, idNameSpace), parentName, 1, r0, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " 
			       << DDName(idName,idNameSpace) 
			       << " number 1 positioned in " << parentName 
			       << " at " << r0 << " with "<<rot;
#endif  
  if (nhalf != 1) {
    rot = DDRotation(DDName("180D", rotns));
    cpv.position(DDName(idName, idNameSpace), parentName, 2, r0, rot);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " 
				 << DDName(idName,idNameSpace) 
				 <<" number 2 positioned in " << parentName
				 << " at " << r0 << " with " << rot;
#endif
  } 
  
  //Construct sector (from -alpha to +alpha)
  name = idName + "Module";
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " 
			       << DDName(name,idNameSpace) 
			       << " Polyhedra made of " << genMat 
			       << " with 1 sector from " 
			       << convertRadToDeg(-alpha) << " to " 
			       << convertRadToDeg(alpha) << " and with " 
			       << pgonZ.size() << " sections";
  for (unsigned int i = 0; i < pgonZ.size(); i++) 
    edm::LogVerbatim("HCalGeom") << "\t\tZ = " << pgonZ[i] << "\tRmin = " 
				 << pgonRmin[i] << "\tRmax = " << pgonRmax[i];
#endif
  solid =   DDSolidFactory::polyhedra(DDName(name, idNameSpace),
				      1, -alpha, 2*alpha, pgonZ,
				      pgonRmin, pgonRmax);
  DDLogicalPart seclogic(solid.ddname(), matter, solid);
  
  for (int ii=0; ii<nsectortot; ii++) {
    double phi    = ii*2*alpha;
    DDRotation rotation;
    std::string rotstr("NULL");
    if (phi != 0) {
      rotstr = "R" + formatAsDegreesInInteger(phi);
      rotation = DDRotation(DDName(rotstr, rotns)); 
      if (!rotation) {
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a new "
				     << "rotation " << rotstr << "\t90," 
                                     << convertRadToDeg(phi) << ",90,"
                                     << (90+convertRadToDeg(phi)) << ", 0, 0";
#endif
	rotation = DDrot(DDName(rotstr, idNameSpace), 90._deg, phi, 90._deg, 
			 (90._deg+phi), 0,  0);
      }
    } 
  
    cpv.position(seclogic, genlogic, ii+1, DDTranslation(0.0, 0.0, 0.0), 
		 rotation);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << seclogic.name() 
				 << " number " << ii+1 << " positioned in " 
				 << genlogic.name() << " at (0,0,0) with " 
				 << rotation;
#endif
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
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() 
			       << " Trap made of " << genMat 
			       << " of dimensions " << dz << ", 0, 0, " 
			       << dy << ", " << dx1 << ", " << dx1 
			       << ", 0, " << dy << ", " << dx2 << ", " << dx2
			       <<", 0";
#endif
  DDLogicalPart glog(solid.ddname(), matter, solid);

  std::string rotstr = name;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation: " 
			       << rotstr << "\t90, 270, " 
			       << (180-convertRadToDeg(theta[2]))
			       << ", 0, " << (90-convertRadToDeg(theta[2])) 
			       << ", 0";
#endif
  rot = DDrot(DDName(rotstr, idNameSpace), 90._deg, 270._deg, 
	      (180._deg-theta[2]), 0, (90._deg-theta[2]), 0);
  DDTranslation r1(0.5*(rinl+routl), 0, 0.5*(pgonZ[1]+pgonZ[2]));
  cpv.position(glog, seclogic, 1, r1, rot);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << glog.name() 
			       << " number 1 positioned in " << seclogic.name()
			       << " at " << r1 << " with " << rot;
#endif
  //Now the cable of type 1
  name = idName + "Cable1";
  double phi  = atan((dx2-dx1)/(2*dz));
  double xmid = 0.5*(dx1+dx2)-1.0;
  solid = DDSolidFactory::box(DDName(name, idNameSpace), 0.5*width1,
			      0.5*thick, 0.5*length1);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() 
			       << " Box made of " << absMat << " of dimension "
			       << 0.5*width1 << ", " << 0.5*thick << ", " 
			       << 0.5*length1;
#endif
  DDName absname(DDSplit(absMat).first, DDSplit(absMat).second); 
  DDMaterial absmatter(absname);
  DDLogicalPart cablog1(solid.ddname(), absmatter, solid);

  rotstr = idName + "Left";
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation " 
			       << rotstr << "\t"  << (90+convertRadToDeg(phi))
			       << ", 0, 90, 90, " << convertRadToDeg(phi) 
			       << ", 0";
#endif
  DDRotation rot2 = DDrot(DDName(rotstr, idNameSpace), (90._deg+phi), 0.0, 
			  90._deg, 90._deg, phi, 0.0);
  DDTranslation r2((xmid-0.5*width1*cos(phi)), 0, 0);
  cpv.position(cablog1, glog, 1, r2, rot2);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog1.name() 
			       << " number 1 positioned in " << glog.name() 
			       << " at " << r2 << " with " << rot2;
#endif
  rotstr = idName + "Right";
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: Creating a rotation " 
			       << rotstr << "\t"  << (90-convertRadToDeg(phi))
			       << ", 0, 90, 90, " << convertRadToDeg(-phi)
			       << ", 0";
#endif
  DDRotation rot3 = DDrot(DDName(rotstr, idNameSpace), (90._deg-phi), 0,
			  90._deg, 90._deg, -phi, 0);
  DDTranslation r3(-(xmid-0.5*width1*cos(phi)), 0, 0);
  cpv.position(cablog1, glog, 2, r3, rot3);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog1.name() 
			       << " number 2 positioned in "  << glog.name()
			       << " at " << r3 << " with " << rot3;
#endif
  //Now the cable of type 2
  name = idName + "Cable2";
  solid = DDSolidFactory::box(DDName(name, idNameSpace), 0.5*width2,
			      0.5*thick, 0.5*length2);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << solid.name() 
			       << " Box made of " << absMat << " of dimension "
			       << 0.5*width2 << ", " << 0.5*thick << ", "
			       << 0.5*length2;
#endif
  DDLogicalPart cablog2(solid.ddname(), absmatter, solid);

  double xpos = 0.5*(width2+gap2);
  cpv.position(cablog2, glog, 1, DDTranslation(xpos, 0.0, 0.0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog2.name() 
			       << " number 1 positioned in "  << glog.name() 
			       << " at (" << xpos << ",  0, 0) with no "
			       << "rotation";
#endif
  cpv.position(cablog2, glog, 2, DDTranslation(-xpos, 0.0, 0.0), DDRotation());
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HCalGeom") << "DDHCalTBCableAlgo: " << cablog2.name() 
			       << " number 2 positioned in "  << glog.name()
			       << " at ("  <<-xpos << ", 0, 0) with no "
			       << "rotation";

  edm::LogVerbatim("HCalGeom") << "<<== End of DDHCalTBCableAlgo construction";
#endif
}
