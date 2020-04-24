///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalEEAlgo.cc
// Description: Geometry factory class for HGCal (EE)
///////////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
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
  thick         = vArgs["Thickness"];
  rotstr        = sArgs["Rotation"];
  layerType     = dbl_to_int(vArgs["LayerType"]);
  heightType    = dbl_to_int(vArgs["HeightType"]);
  thickBlock    = vArgs["LayerThick"];
  zMinBlock     = nArgs["zMinBlock"];
  for (unsigned int i=0; i<materials.size(); ++i) {
    copyNumber.emplace_back(1);
  }
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: " << materials.size()
			    << " volumes to be put with rotation " << rotstr
			    << " starting at " << zMinBlock;
  for (unsigned int i=0; i<names.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Volume [" << i << "] " << names[i]
			      << " of thickness " << thick[i]
			      << " filled with " << materials[i]
			      << " first copy number " << copyNumber[i];
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: " << layerType.size()
			    << " layers";
  for (unsigned int i=0; i<layerType.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Layer [" << i << "] with material type " 
			      << layerType[i] << " height type "
			      << heightType[i] << " block thickness "
			      << thickBlock[i];

  sectors       = (int)(nArgs["Sectors"]);
  slopeB        = nArgs["SlopeBottom"];
  slopeT        = vArgs["SlopeTop"];
  zFront        = vArgs["ZFront"];
  rMaxFront     = vArgs["RMaxFront"];
  idName        = parent().name().name();
  idNameSpace   = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: Bottom slope " << slopeB
			    << " " << slopeT.size() << " slopes for top";
  for (unsigned int i=0; i<slopeT.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i]
			      << " Rmax " << rMaxFront[i] << " Slope " 
			      << slopeT[i];
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo: Sectors " << sectors
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

void DDHGCalEEAlgo::constructLayers(const DDLogicalPart& module, DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: \t\tInside Layers";

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));

  double  zi(zMinBlock), zz(zMinBlock);
  for (unsigned int i=0; i<layerType.size(); i++) {
    int     ii    = layerType[i];
    int     copy  = copyNumber[ii];
    ++copyNumber[ii];
    double layer_thick = thickBlock[i];

    if (heightType[i] == 0) zz = zi;
    double  zlayer = zz + layer_thick;
    double  zo     = zi + thick[ii];
    double  rinF   = zi * slopeB;
    double  rinB   = zlayer * slopeB;
    double  routF  = (heightType[i] == 0) ? rMax(zi) : rMax(zz);
    double  routB  = rMax(zo);

    std::string name = "HGCal"+names[ii]+std::to_string(copy);
    edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: Layer " << i << ":" 
			      << ii << " Front " << zi << ", " << rinF << ", " 
			      << routF << " Back " << zo << ", " << rinB 
			      << ", " << routB << " superlayer thickness " 
			      << layer_thick;
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
  }   // End of loop on layers
}


DDHGCalEEAlgo::HGCalEEPar 
DDHGCalEEAlgo::parameterLayer(double rinF, double routF, double rinB, 
			      double routB, double zi, double zo) {

  DDHGCalEEAlgo::HGCalEEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  edm::LogInfo("HGCalGeom") << "Input: Front " << rinF << " " << routF << " "
			    << zi << " Back " << rinB << " " << routB << " "
			    << zo << " Alpha " << alpha/CLHEP::deg;

  parm.yh2  = parm.yh1  = 0.5 * (routF*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = routF * sin(alpha);
  parm.xpos = 0.5*(routF*cos(alpha)+rinB);
  parm.ypos = 0.0;
  parm.zpos = 0.5*(zi+zo);
  parm.alp  = parm.theta  = parm.phi = 0;
  edm::LogInfo("HGCalGeom") << "Output Dimensions " << parm.yh1 << " " 
			    << parm.bl1 << " " << parm.tl1 << " " << parm.yh2 
			    << " " << parm.bl2 << " " << parm.tl2 << " " 
			    << parm.alp/CLHEP::deg <<" "<<parm.theta/CLHEP::deg
			    << " " << parm.phi/CLHEP::deg << " Position " 
			    << parm.xpos << " " << parm.ypos <<" " <<parm.zpos;
  return parm;
}

double DDHGCalEEAlgo::rMax(double z) {

  double r(0);
  unsigned int ik(0);
  for (unsigned int k=0; k<slopeT.size(); ++k) {
    if (z < zFront[k]) break;
    r  = rMaxFront[k] + (z - zFront[k]) * slopeT[k];
    ik = k;
  }
  edm::LogInfo("HGCalGeom") << "rMax : " << z << ":" << ik << ":" << r ;
  return r;
}
