///////////////////////////////////////////////////////////////////////////////
// File: DDHGCalHEAlgo.cc
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
#include "Geometry/HGCalCommonData/plugins/DDHGCalHEAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

DDHGCalHEAlgo::DDHGCalHEAlgo() {
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo info: Creating an instance";
}

DDHGCalHEAlgo::~DDHGCalHEAlgo() {}

void DDHGCalHEAlgo::initialize(const DDNumericArguments & nArgs,
			       const DDVectorArguments & vArgs,
			       const DDMapArguments & ,
			       const DDStringArguments & sArgs,
			       const DDStringVectorArguments &vsArgs){

  materials     = vsArgs["MaterialNames"];
  names         = vsArgs["VolumeNames"];
  thick         = vArgs["Thickness"];
  type          = dbl_to_int(vArgs["Type"]);
  copyNumber    = dbl_to_int(vArgs["Offsets"]);
  zMinBlock     = vArgs["ZMinType"];
  rotstr        = sArgs["Rotation"];
  layerType     = dbl_to_int(vArgs["LayerType"]);
  heightType    = dbl_to_int(vArgs["HeightType"]);
  thickModule   = nArgs["ThickModule"];
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: " << materials.size()
			    << " volumes to be put with rotation " << rotstr
			    << " in " << layerType.size() << " layers and "
			    << "module thickness " << thickModule;
  for (unsigned int i=0; i<names.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Volume [" << i << "] " << names[i]
			      << " filled with " << materials[i] << " of type "
			      << type[i] << " thickness " << thick[i]
			      << " first copy number " << copyNumber[i]
			      << " starting at " << zMinBlock[i];
  for (unsigned int i=0; i<layerType.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Layer [" << i << "] with material type " 
			      << layerType[i] << " height type "
			      << heightType[i];

  sectors       = (int)(nArgs["Sectors"]);
  slopeB        = nArgs["SlopeBottom"];
  slopeT        = vArgs["SlopeTop"];
  zFront        = vArgs["ZFront"];
  rMaxFront     = vArgs["RMaxFront"];
  idName        = parent().name().name();
  idNameSpace   = DDCurrentNamespace::ns();
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: Bottom slope " << slopeB
			    << " " << slopeT.size() << " slopes for top";
  for (unsigned int i=0; i<slopeT.size(); ++i)
    edm::LogInfo("HGCalGeom") << "Block [" << i << "] Zmin " << zFront[i]
			      << " Rmax " << rMaxFront[i] << " Slope " 
			      << slopeT[i];
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo: Sectors " << sectors
			    << "\tNameSpace:Name " << idNameSpace
			    << ":" << idName;

}

////////////////////////////////////////////////////////////////////
// DDHGCalHEAlgo methods...
////////////////////////////////////////////////////////////////////

void DDHGCalHEAlgo::execute(DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "==>> Constructing DDHGCalHEAlgo...";
  constructLayers (parent(), cpv);
  edm::LogInfo("HGCalGeom") << "<<== End of DDHGCalHEAlgo construction ...";
}

void DDHGCalHEAlgo::constructLayers(DDLogicalPart module, DDCompactView& cpv) {
  
  edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: \t\tInside Layers";

  ///////////////////////////////////////////////////////////////
  //Pointers to the Rotation Matrices and to the Materials
  DDRotation rot(DDName(DDSplit(rotstr).first, DDSplit(rotstr).second));

  double zz(zMinBlock[0]);
  for (unsigned int i=0; i<layerType.size(); i++) {
    int   ii       = layerType[i];
    int   copy     = copyNumber[ii];
    ++copyNumber[ii];
    int   ityp     = type[ii];
    type[ii]       =-ityp;
    double zi      = zMinBlock[ii];
    zMinBlock[ii] += thickModule;
    if (heightType[i] == 0) zz = zi;

    double zlayer  = zz + 2*thickModule;
    if((i % 6)>2) zlayer  = zz + thickModule;

    double zo      = zi + thick[ii];
    double rinF    = zi * slopeB;
    double rinB    = zlayer * slopeB;

    double routF   = (heightType[i] == 0) ? rMax(zi) : rMax(zz);
    if((i % 6)>2) routF   = (heightType[i] == 0) ? rMax(zi-thickModule) : rMax(zz-thickModule);
    
    double routB   = rMax(zo);
    std::string name = "HGCal"+names[ii]+dbl_to_string(copy);
    edm::LogInfo("HGCalGeom") << "DDHGCalEEAlgo test: Layer " << i << ":" 
			      << ii << ":" << ityp << " Front " << zi << ", " 
			      << rinF << ", " << routF << " Back " << zo 
			      << ", " << rinB << ", " << routB;
    DDHGCalHEAlgo::HGCalHEPar parm = (ityp == 0) ?
      parameterLayer(rinF, routF, rinB, routB, zi, zo) :
      parameterLayer(ityp, rinF, routF, rinB, routB, zi, zo);
    DDSolid solid = DDSolidFactory::trap(DDName(name, idNameSpace), 
					 0.5*thick[ii], parm.theta,
					 parm.phi, parm.yh1, parm.bl1, 
					 parm.tl1, parm.alp, parm.yh2,
					 parm.bl2, parm.tl2, parm.alp);

    DDName matName(DDSplit(materials[ii]).first,
                   DDSplit(materials[ii]).second);
    DDMaterial matter(matName);
    DDLogicalPart glog = DDLogicalPart(solid.ddname(), matter, solid);
          edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: " 
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
    edm::LogInfo("HGCalGeom") << "DDHGCalHEAlgo test: " << glog.name()
			      << " number " << copy << " positioned in " 
			      << module.name() << " at " << r1 << " with " 
			      << rot;
  }   // End of loop on layers
}


DDHGCalHEAlgo::HGCalHEPar 
DDHGCalHEAlgo::parameterLayer(double rinF, double routF, double rinB, 
			      double routB, double zi, double zo) {

  DDHGCalHEAlgo::HGCalHEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  double rout  = routF;
  edm::LogInfo("HGCalGeom") << "Input: Front " << rinF << " " << routF << " "
			    << zi << " Back " << rinB << " " << routB << " "
			    << zo << " Alpha " << alpha/CLHEP::deg << " Rout "
			    << rout;

  parm.yh2  = parm.yh1  = 0.5 * (rout*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = rout  * sin(alpha);
  parm.xpos = 0.5*(rout*cos(alpha)+rinB);
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

DDHGCalHEAlgo::HGCalHEPar 
DDHGCalHEAlgo::parameterLayer(int type, double rinF, double routF, double rinB,
			      double routB, double zi, double zo) {

  DDHGCalHEAlgo::HGCalHEPar parm;
  //Given rin, rout compute parameters of the trapezoid and 
  //position of the trapezoid for a standrd layer
  double alpha = CLHEP::pi/sectors;
  double rout  = routF;
  edm::LogInfo("HGCalGeom") << "Input " << type << " Front " << rinF << " " 
			    << routF << " " << zi << " Back " << rinB << " " 
			    << routB << " " << zo <<" Alpha " 
			    << alpha/CLHEP::deg << " rout " << rout;

  parm.yh2  = parm.yh1  = 0.5 * (rout*cos(alpha) - rinB);
  parm.bl2  = parm.bl1  = 0.5 * rinB  * tan(alpha);
  parm.tl2  = parm.tl1  = 0.5 * rout  * sin(alpha);
  double dx = 0.25* (parm.bl2+parm.tl2-parm.bl1-parm.tl1);
  double dy = 0.0;
  parm.xpos = 0.5*(rout*cos(alpha)+rinB);
  parm.ypos = 0.25*(parm.bl2+parm.tl2+parm.bl1+parm.tl1);
  parm.zpos = 0.5*(zi+zo);
  parm.alp  = atan(0.5 * tan(alpha));
  if (type > 0) {
    parm.ypos = -parm.ypos;
  } else {
    parm.alp  = -parm.alp;
    dx        = -dx;
  }
  double r    = sqrt (dx*dx + dy*dy);
  edm::LogInfo("HGCalGeom") << "dx|dy|r " << dx << ":" << dy << ":" << r;
  if (r > 1.0e-8) {
    parm.theta  = atan (r/(zo-zi));
    parm.phi    = atan2 (dy, dx);
  } else {
    parm.theta  = parm.phi = 0;
  }
  edm::LogInfo("HGCalGeom") << "Output Dimensions " << parm.yh1 << " " 
			    << parm.bl1 << " " << parm.tl1 << " " << parm.yh2 
			    << " " << parm.bl2 << " " << parm.tl2 << " " 
			    << parm.alp/CLHEP::deg <<" " <<parm.theta/CLHEP::deg
			    << " " << parm.phi/CLHEP::deg << " Position " 
			    << parm.xpos << " " << parm.ypos << " " <<parm.zpos;
  return parm;
}

double DDHGCalHEAlgo::rMax(double z) {

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
