///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalModuleAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalModuleAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

DDHGCalModuleAlgo::DDHGCalModuleAlgo() {
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo info: Creating an instance";
#endif
}

DDHGCalModuleAlgo::~DDHGCalModuleAlgo() {}

void DDHGCalModuleAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments &vsArgs){

  wafer         = vsArgs["WaferName"];
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: " << wafer.size()
			    << " wafers";
  for (unsigned int i=0; i<wafer.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Wafer[" << i << "] " << wafer[i];
#endif
  materials     = vsArgs["MaterialNames"];
  names         = vsArgs["VolumeNames"];
  thick         = vArgs["Thickness"];
  for (unsigned int i=0; i<materials.size(); ++i) {
    copyNumber.push_back(1);
  }
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: " << materials.size()
			    << " types of volumes";
  for (unsigned int i=0; i<names.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Volume [" << i << "] " << names[i]
			      << " of thickness " << thick[i]
			      << " filled with " << materials[i]
			      << " first copy number " << copyNumber[i];
#endif
  layers        = dbl_to_int(vArgs["Layers"]);
  layerThick    = vArgs["LayerThick"];
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: " << layers.size()
			    << " blocks";
  for (unsigned int i=0; i<layers.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] of thickness " 
			      << layerThick[i] << " with " << layers[i]
			      << " layers";
#endif
  layerType     = dbl_to_int(vArgs["LayerType"]);
  layerSense    = dbl_to_int(vArgs["LayerSense"]);
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: " << layerType.size()
			    << " layers";
  for (unsigned int i=0; i<layerType.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Layer [" << i << "] with material type " 
			      << layerType[i] << " sensitive class "
			      << layerSense[i];
#endif
  zMinBlock     = nArgs["zMinBlock"];
  rMaxFine      = nArgs["rMaxFine"];
  waferW        = nArgs["waferW"];
  sectors       = (int)(nArgs["Sectors"]);
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: zStart " << zMinBlock
			    << " rFineCoarse " << rMaxFine << " wafer width "
			    << waferW << " sectors " << sectors;
#endif
  slopeB        = vArgs["SlopeBottom"];
  slopeT        = vArgs["SlopeTop"];
  zFront        = vArgs["ZFront"];
  rMaxFront     = vArgs["RMaxFront"];
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: Bottom slopes " << slopeB[0]
			    << ":" << slopeB[1] << " and " << slopeT.size() 
			    << " slopes for top";
  for (unsigned int i=0; i<slopeT.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i]
			      << " Rmax " << rMaxFront[i] << " Slope " 
			      << slopeT[i];
#endif
  idNameSpace   = DDCurrentNamespace::ns();
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: NameSpace " << idNameSpace;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalModuleAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalModuleAlgo::execute(DDCompactView& cpv) {
  
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalModuleAlgo...";
#endif
  copies.clear();
  constructLayers (parent(), cpv);
  edm::LogInfo("HGCalGeom") << copies.size() << " different wafer copy numbers";
  copies.clear();
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "<<== End of DDHGCalModuleAlgo construction ...";
#endif
}

void DDHGCalModuleAlgo::constructLayers(DDLogicalPart module, 
					DDCompactView& cpv) {
  
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo test: \t\tInside Layers";
#endif
  double       zi(zMinBlock);
  int          laymin(0);
  const double tol(0.01);
  for (unsigned int i=0; i<layers.size(); i++) {
    double  zo     = zi + layerThick[i];
    double  routF  = rMax(zi);
    int     laymax = laymin+layers[i];
    double  zz     = zi;
    for (int ly=laymin; ly<laymax; ++ly) {
      int     ii     = layerType[ly];
      int     copy   = copyNumber[ii];
      double  rinB   = (layerSense[ly] == 0) ? (zo*slopeB[0]) : (zo*slopeB[1]);
      zz            += (0.5*thick[ii]);

      std::string name = "HGCal"+names[ii]+dbl_to_string(copy);
#ifdef DebugLog
      edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo test: Layer " << ly << ":"
				<< ii << " Front " << zi << ", " << routF 
				<< " Back " << zo << ", " << rinB 
				<< " superlayer thickness " << layerThick[i];
#endif
      DDName matName(DDSplit(materials[ii]).first, 
		     DDSplit(materials[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense[ly] == 0) {
	double alpha = CLHEP::pi/sectors;
	double rmax  = routF*cos(alpha) - tol;
	std::vector<double> pgonZ, pgonRin, pgonRout;
	pgonZ.push_back(-0.5*thick[ii]);    pgonZ.push_back(0.5*thick[ii]);
	pgonRin.push_back(rinB);            pgonRin.push_back(rinB);   
	pgonRout.push_back(rmax);           pgonRout.push_back(rmax);   
	DDSolid solid = DDSolidFactory::polyhedra(DDName(name, idNameSpace),
						  sectors, -alpha, CLHEP::twopi,
						  pgonZ, pgonRin, pgonRout);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef DebugLog
	edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo test: " 
				  << solid.name() << " polyhedra of " << sectors
				  << " sectors covering " << -alpha/CLHEP::deg
				  << ":" << (-alpha+CLHEP::twopi)/CLHEP::deg
				  << " with " << pgonZ.size() << " sections";
	for (unsigned int k=0; k<pgonZ.size(); ++k)
	  edm::LogInfo("HGCalGeom") << "[" << k << "] z " << pgonZ[k] << " R "
				    << pgonRin[k] << ":" << pgonRout[k];
#endif
      } else {
	DDSolid solid = DDSolidFactory::tubs(DDName(name, idNameSpace), 
					     0.5*thick[ii], rinB, routF, 0.0,
					     CLHEP::twopi);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef DebugLog
	edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo test: " 
				  << solid.name() << " Tubs made of " << matName
				  << " of dimensions " << rinB << ", " << routF
				  << ", " << 0.5*thick[ii] << ", 0.0, "
				  << CLHEP::twopi/CLHEP::deg;
#endif
	positionSensitive(glog,rinB,routF,cpv);
      }
      DDTranslation r1(0,0,zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber[ii];
#ifdef DebugLog
      edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo test: " << glog.name()
				<< " number " << copy << " positioned in " 
				<< module.name() << " at " << r1 << " with " 
				<< rot;
#endif
      zz += (0.5*thick[ii]);
    } // End of loop over layers in a block
    zi     = zo;
    laymin = laymax;
  }   // End of loop over blocks
}

double DDHGCalModuleAlgo::rMax(double z) {
  double r(0);
#ifdef DebugLog
  unsigned int ik(0);
#endif
  for (unsigned int k=0; k<slopeT.size(); ++k) {
    if (z < zFront[k]) break;
    r  = rMaxFront[k] + (z - zFront[k]) * slopeT[k];
#ifdef DebugLog
    ik = k;
#endif
  }
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << "rMax : " << z << ":" << ik << ":" << r ;
#endif
  return r;
}

void DDHGCalModuleAlgo::positionSensitive(DDLogicalPart& glog, double rin,
					  double rout, DDCompactView& cpv) {
  double dx   = 0.5*waferW;
  double dy   = 3.0*dx*tan(30.0*CLHEP::deg);
  double rr   = 2.0*dx*tan(30.0*CLHEP::deg);
  int    ncol = (int)(2.0*rout/waferW) + 1;
  int    nrow = (int)(rout/(waferW*tan(30.0*CLHEP::deg))) + 1;
  int    incm(0), inrm(0), kount(0);
#ifdef DebugLog
  edm::LogInfo("HGCalGeom") << glog.ddname() << " rout " << rout << " Row "
			    << nrow << " Column " << ncol; 
#endif
  for (int nr=-nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc=-ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr%2 == inc%2) {
	double xpos = nc*dx;
	double ypos = nr*dy;
	double rpos = std::sqrt(xpos*xpos+ypos*ypos);
	if (rpos-rr >= rin && rpos+rr <= rout) {
	  DDTranslation tran(xpos, ypos, 0.0);
	  DDRotation rotation;
	  int copy = inr*100 + inc;
	  if (nc < 0) copy += 10000;
	  if (nr < 0) copy += 100000;
	  DDName name = (rpos < rMaxFine) ? 
	    DDName(DDSplit(wafer[0]).first, DDSplit(wafer[0]).second) : 
	    DDName(DDSplit(wafer[1]).first, DDSplit(wafer[1]).second);
	  cpv.position(name, glog.ddname(), copy, tran, rotation);
	  if (inc > incm) incm = inc;
	  if (inr > inrm) inrm = inr;
	  kount++;
	  if (std::find(copies.begin(),copies.end(),copy) == copies.end())
	    copies.push_back(copy);
#ifdef DebugLog
	  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: " 
				    << name << " number " << copy
				    << " positioned in " << glog.ddname()
				    << " at " << tran << " with " << rotation;
#endif
	}
      }
    }
  }
  edm::LogInfo("HGCalGeom") << "DDHGCalModuleAlgo: # of columns " << incm
			    << " # of rows " << inrm << " and " << kount
			    << " wafers for " <<glog.ddname();
}
