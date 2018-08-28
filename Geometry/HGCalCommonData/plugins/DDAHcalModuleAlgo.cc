///////////////////////////////////////////////////////////////////////////////
// File: DDAHcalModuleAlgo.cc
// Description: Geometry factory class for HGCal (EE and HESil)
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
#include "Geometry/HGCalCommonData/plugins/DDAHcalModuleAlgo.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDAHcalModuleAlgo::DDAHcalModuleAlgo() {
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo info: Creating an instance" << std::endl;
#endif
}

DDAHcalModuleAlgo::~DDAHcalModuleAlgo() {}

void DDAHcalModuleAlgo::initialize(const DDNumericArguments & nArgs,
				   const DDVectorArguments & vArgs,
				   const DDMapArguments & ,
				   const DDStringArguments & sArgs,
				   const DDStringVectorArguments &vsArgs){

  tile          = sArgs["TileName"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: Tile " << tile << std::endl;
#endif
  materials     = vsArgs["MaterialNames"];
  names         = vsArgs["VolumeNames"];
  thick         = vArgs["Thickness"];
  for (unsigned int i=0; i<materials.size(); ++i) {
    copyNumber.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << materials.size()
	    << " types of volumes" << std::endl;
  for (unsigned int i=0; i<names.size(); ++i)
    std::cout << "Volume [" << i << "] " << names[i] << " of thickness " 
	      << thick[i] << " filled with " << materials[i]
	      << " first copy number " << copyNumber[i] << std::endl;
#endif
  layers        = dbl_to_int(vArgs["Layers"]);
  layerThick    = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << layers.size() << " blocks" <<std::endl;
  for (unsigned int i=0; i<layers.size(); ++i)
    std::cout << "Block [" << i << "] of thickness "  << layerThick[i] 
	      << " with " << layers[i] << " layers" << std::endl;
#endif
  layerType     = dbl_to_int(vArgs["LayerType"]);
  layerSense    = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << layerType.size() << " layers" 
	    << std::endl;
  for (unsigned int i=0; i<layerType.size(); ++i)
    std::cout << "Layer [" << i << "] with material type "  << layerType[i]
	      << " sensitive class " << layerSense[i] << std::endl;
#endif
  widths        = vArgs["Widths"];
  heights       = vArgs["Heights"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << widths.size() << " sizes for width " 
	    << "and height:";
  for (unsigned int i=0; i<widths.size(); ++i)
    std::cout << " [" << i << "] " << widths[i] << ":" << heights[i];
  std::cout << std::endl;
#endif
  tileN         = dbl_to_int(vArgs["TileN"]);
  tileStep      = vArgs["TileStep"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << tileN.size() << " tile positioning " 
	    << "parameters:";
  for (unsigned int i=0; i<tileN.size(); ++i)
    std::cout << " [" << i << "] " << tileN[i] << ":" << tileStep[i];
  std::cout << std::endl;
#endif
  zMinBlock     = nArgs["zMinBlock"];
  DDCurrentNamespace ns;
  idNameSpace   = *ns;
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: NameSpace " << idNameSpace << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////
// DDAHcalModuleAlgo methods...
////////////////////////////////////////////////////////////////////

void DDAHcalModuleAlgo::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  std::cout << "==>> Constructing DDAHcalModuleAlgo..." << std::endl;
#endif
  constructLayers (parent(), cpv);
#ifdef EDM_ML_DEBUG
  std::cout << "<<== End of DDAHcalModuleAlgo construction ..." << std::endl;
#endif
}

void DDAHcalModuleAlgo::constructLayers(const DDLogicalPart& module, 
					DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo test: \t\tInside Layers" << std::endl;
#endif
  double       zi(zMinBlock);
  int          laymin(0);
  for (unsigned int i=0; i<layers.size(); i++) {
    double  zo     = zi + layerThick[i];
    int     laymax = laymin+layers[i];
    double  zz     = zi;
    double  thickTot(0);
    for (int ly=laymin; ly<laymax; ++ly) {
      int     ii     = layerType[ly];
      int     copy   = copyNumber[ii];
      zz            += (0.5*thick[ii]);
      thickTot      += thick[ii];

      std::string name = "HGCal"+names[ii]+std::to_string(copy);
#ifdef EDM_ML_DEBUG
      std::cout << "DDAHcalModuleAlgo test: Layer " << ly << ":" << ii 
		<< " Front " << zi << " Back " << zo 
		<< " superlayer thickness " << layerThick[i] << std::endl;
#endif
      DDName matName(DDSplit(materials[ii]).first, 
		     DDSplit(materials[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense[ly] == 0) {
	DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace),
					    0.5*widths[0], 0.5*heights[0],
					    0.5*thick[ii]);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
	std::cout << "DDAHcalModuleAlgo test: " << solid.name() 
		  << " Box made of " << matName << " of dimensions " 
		  << 0.5*widths[0] << ", " << 0.5*heights[0] << ", " 
		  << 0.5*thick[ii] << std::endl;
#endif
      } else {
	DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace), 
					    0.5*widths[1], 0.5*heights[1],
					    0.5*thick[ii]);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
	std::cout << "DDAHcalModuleAlgo test: " << solid.name()
		  << " Box made of " << matName << " of dimensions " 
		  << 0.5*widths[1] << ", " << 0.5*heights[1] << ", " 
		  << 0.5*thick[ii] << std::endl;
#endif
	positionSensitive(glog,cpv);
      }
      DDTranslation r1(0,0,zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber[ii];
#ifdef EDM_ML_DEBUG
      std::cout << "DDAHcalModuleAlgo test: " << glog.name() << " number "
		<< copy << " positioned in " << module.name() << " at " << r1 
		<< " with " << rot << std::endl;
#endif
      zz += (0.5*thick[ii]);
    } // End of loop over layers in a block
    zi     = zo;
    laymin = laymax;
    if (fabs(thickTot-layerThick[i]) < 0.00001) {
    } else if (thickTot > layerThick[i]) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick[i]
				 << " is smaller than thickness " << thickTot
				 << " of all its components **** ERROR ****\n";
    } else if (thickTot < layerThick[i]) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " 
				   << layerThick[i] << " does not match with "
				   << thickTot << " of the components\n";
    }
  }   // End of loop over blocks
}

void DDAHcalModuleAlgo::positionSensitive(DDLogicalPart& glog,
					  DDCompactView& cpv) {
  int    ncol = tileN[0]/2;
  int    nrow = tileN[1]/2;
#ifdef EDM_ML_DEBUG
  int    kount(0);
  std::cout << glog.ddname() << " Row " << nrow << " Column " << ncol 
	    << std::endl; 
#endif
  for (int nr=-nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    double ypos = (nr >= 0) ? (inr-0.5)*tileStep[1] : -(inr-0.5)*tileStep[1];
    for (int nc=-ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      double xpos = (nc >= 0) ? (inc-0.5)*tileStep[0] : -(inc-0.5)*tileStep[0];
      if (nr != 0 && nc != 0) {
	DDTranslation tran(xpos, ypos, 0.0);
	DDRotation rotation;
	int copy = inr*10 + inc;
	if (nc < 0) copy += 100;
	if (nr < 0) copy += 1000;
	DDName name = DDName(DDSplit(tile).first, DDSplit(tile).second);
	cpv.position(name, glog.ddname(), copy, tran, rotation);
#ifdef EDM_ML_DEBUG
	kount++;
	std::cout << "DDAHcalModuleAlgo: " << name << " number " << copy
		  << " positioned in " << glog.ddname() << " at " << tran 
		  << " with " << rotation << std::endl;
#endif
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "DDAHcalModuleAlgo: " << kount << " tiles for " 
	    << glog.ddname() << std::endl;
#endif
}
