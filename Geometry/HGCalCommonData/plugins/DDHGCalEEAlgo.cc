///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalEEAlgo.cc
// Description: Geometry factory class for HGCal (EE)
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
#include "Geometry/HGCalCommonData/plugins/DDHGCalEEAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHGCalEEAlgo::DDHGCalEEAlgo() {
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo info: Creating an instance";
}

DDHGCalEEAlgo::~DDHGCalEEAlgo() {}

void DDHGCalEEAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments &vsArgs){

  materials     = vsArgs["MaterialNames"];
  names         = vsArgs["VolumeNames"];
  rotstr        = sArgs["Rotation"];
  layers        = dbl_to_int(vArgs["Layers"]);
  thick1        = vArgs["ThicknessType1"];
  thick2        = vArgs["ThicknessType2"];
  thick3        = vArgs["ThicknessType3"];
  thick4        = vArgs["ThicknessType4"];
  thick5        = vArgs["ThicknessType5"];
  zMinBlock     = vArgs["zMinBlock"];
  int ioff(1);
  for (unsigned int i=0; i<layers.size(); ++i) {
    offsets.push_back(ioff);
    ioff += layers[i];
  }
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: " << materials.size()
			    << " volumes to be put with rotation " << rotstr;
  for (unsigned int i=0; i<names.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Volume [" << i << "] " << names[i]
			      << " filled with " << materials[i];
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: " << zMinBlock.size()
			    << " blocks";
  for (unsigned int i=0; i<layers.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] with " << layers[i]
			      << " layers; offset " << offsets[i] << " zmin "
			      << zMinBlock[i] << " thicknesses " << thick1[i]
			      << ":" << thick2[i] << ":" << thick3[i] << ":"
			      << thick4[i] << ":" << thick5[i];

  slopeB        = nArgs["SlopeBottom"];
  slopeT        = nArgs["SlopeTop"];
  zFront        = nArgs["ZFront"];
  rMaxFront     = nArgs["RMaxFront"];
  sectors       = (int)(nArgs["Sectors"]);
  idName        = parent().name().name();
  idNameSpace   = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: Zmin " << zFront << "\tRmax "
			    << rMaxFront << "\tSlopes " << slopeB << ":"
			    << slopeT << "\tSectors " << sectors
			    << "\tNameSpace:Name " << idNameSpace
			    << ":" << idName;

}

////////////////////////////////////////////////////////////////////
// DDHGCalEEAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalEEAlgo::execute(DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalEEAlgo...";
  constructLayers (parent(), cpv);
  edm::LogInfo("HGCalGeom") << "<<== End of DDHGCalEEAlgo construction ...";
}

void DDHGCalEEAlgo::constructLayers(DDLogicalPart module, DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: \t\tInside Layers";

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));

  for (unsigned int i=0; i<layers.size(); i++) {
    double  zi    = zMinBlock[i];
    int     copy  = offsets[i];
    std::vector<double> thick;
    thick.push_back(thick1[i]);
    thick.push_back(thick2[i]);
    thick.push_back(thick3[i]);
    thick.push_back(thick4[i]);
    thick.push_back(thick5[i]);
    for (int k=0; k<layers[i]; ++k) {
      for (unsigned int ii=0; ii<names.size(); ++ii) {
	double  zo     = zi + thick[ii];
	double  rinF   = zi * slopeB;
	double  rinB   = zo * slopeB;
	double  routF  = rMaxFront + (zi - zFront) * slopeT;
	double  routB  = rMaxFront + (zo - zFront) * slopeT;
	std::string name = "HGCal"+names[ii]+dbl_to_string(copy);
	edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: Layer " << i << ":" 
				  << ii << " Front " << zi << ", " << rinF 
				  << ", " << routF << " Back " << zo 
				  << ", " << rinB << ", " << routB;
	DDHGCalEEAlgo::HGCalEEPar parm = parameterLayer(rinF, routF, rinB, 
							routB, zi, zo);
	DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
					     0.5*thick[ii], parm.theta,
					     parm.phi, parm.yh1, parm.bl1, 
					     parm.tl1, parm.alp, parm.yh2,
					     parm.bl2, parm.tl2, parm.alp);

	DDName matName(DDSplit(materials[ii]).first, 
		       DDSplit(materials[ii]).second);
	DDMaterial matter(matName);
	DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
	edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: " 
				  << solid.name() << " Trap made of " << matName
				  << " of dimensions " << 0.5*thick[ii] << ", "
				  << parm.theta/CLHEP::deg << ", " 
				  << parm.phi/CLHEP::deg << ", " << parm.yh1 
				  << ", " << parm.bl1 << ", " << parm.tl1 
				  << ", " << parm.alp/CLHEP::deg << ", " 
				  << parm.yh2 << ", " << parm.bl2 << ", " 
				  << parm.tl2 << ", " << parm.alp/CLHEP::deg;
	DDTranslation r1(parm.xpos, parm.ypos, parm.zpos);
	cpv.position(glog, module, copy, r1, rot);
	edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: " << glog.name()
				  << " number " << copy << " positioned in " 
				  << module.name() << " at " << r1 << " with " 
				  << rot;
	zi = zo;
      } // End of loop on layer types
      copy++;
    }   // End of loop on layers
  }     // End of loop on blocks
}


DDHGCalEEAlgo::HGCalEEPar 
DDHGCalEEAlgo::parameterLayer(double rinF, double routF, double rinB, 
			      double routB, double zi, double zo) {

  DDHGCalEEAlgo::HGCalEEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  edm::LogInfo("HCalGeom") << "Input: Front " << rinF << " " << routF << " "
			   << zi << " Back " << rinB << " " << routB << " "
			   << zo << " Alpha " << alpha/CLHEP::deg;

  parm.yh2  = parm.yh1  = 0.5 * (routF*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = routF * sin(alpha);
  parm.xpos = 0.5*(routF*cos(alpha)+rinB);
  parm.ypos = 0.0;
  parm.zpos = 0.5*(zi+zo);
  parm.alp  = parm.theta  = parm.phi = 0;
  edm::LogInfo("HCalGeom") << "Output Dimensions " << parm.yh1 << " " 
			   << parm.bl1 << " " << parm.tl1 << " " << parm.yh2 
			   << " " << parm.bl2 << " " << parm.tl2 << " " 
			   << parm.alp/CLHEP::deg <<" " <<parm.theta/CLHEP::deg
			   << " " << parm.phi/CLHEP::deg << " Position " 
			   << parm.xpos << " " << parm.ypos << " " <<parm.zpos;
  return parm;
}
