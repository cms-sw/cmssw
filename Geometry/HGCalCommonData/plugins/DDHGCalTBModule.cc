#include <cmath>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "Geometry/HGCalCommonData/plugins/DDHGCalTBModule.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

DDHGCalTBModule::DDHGCalTBModule() {
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule info: Creating an instance" << std::endl;
#endif
}

DDHGCalTBModule::~DDHGCalTBModule() {}

void DDHGCalTBModule::initialize(const DDNumericArguments & nArgs,
				 const DDVectorArguments & vArgs,
				 const DDMapArguments & ,
				 const DDStringArguments & sArgs,
				 const DDStringVectorArguments &vsArgs){
  
  wafer_        = vsArgs["WaferName"];
  covers_       = vsArgs["CoverName"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: " << wafer_.size() << " wafers" << std::endl;
  unsigned int i(0);
  for (auto wafer : wafer_) {
    std::cout << "Wafer[" << i << "] " << wafer << std::endl; ++i;}
  std::cout << "DDHGCalTBModule: " << covers_.size() << " covers" << std::endl;
  i = 0;
  for (auto cover : covers_) {
    std::cout << "Cover[" << i << "] " << cover << std::endl; ++i;}
#endif
  materials_    = vsArgs["MaterialNames"];
  names_        = vsArgs["VolumeNames"];
  thick_        = vArgs["Thickness"];
  for (unsigned int i=0; i<materials_.size(); ++i) {
    copyNumber_.emplace_back(1);
  }
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: " << materials_.size()
	    << " types of volumes" << std::endl;
  for (unsigned int i=0; i<names_.size(); ++i)
    std::cout << "Volume [" << i << "] " << names_[i] << " of thickness " 
	      << thick_[i] << " filled with " << materials_[i]
	      << " first copy number " << copyNumber_[i] << std::endl;
#endif
  layers_       = dbl_to_int(vArgs["Layers"]);
  layerThick_   = vArgs["LayerThick"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: " << layers_.size() << " blocks" <<std::endl;
  for (unsigned int i=0; i<layers_.size(); ++i)
    std::cout << "Block [" << i << "] of thickness "  << layerThick_[i] 
	      << " with " << layers_[i] << " layers" << std::endl;
#endif
  layerType_    = dbl_to_int(vArgs["LayerType"]);
  layerSense_   = dbl_to_int(vArgs["LayerSense"]);
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: " << layerType_.size() << " layers" 
	    << std::endl;
  for (unsigned int i=0; i<layerType_.size(); ++i)
    std::cout << "Layer [" << i << "] with material type "  << layerType_[i]
	      << " sensitive class " << layerSense_[i] << std::endl;
#endif
  zMinBlock_    = nArgs["zMinBlock"];
  rMaxFine_     = nArgs["rMaxFine"];
  waferW_       = nArgs["waferW"];
  waferGap_     = nArgs["waferGap"];
  absorbW_      = nArgs["absorberW"];
  absorbH_      = nArgs["absorberH"];
  sectors_      = (int)(nArgs["Sectors"]);
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: zStart " << zMinBlock_ << " rFineCoarse " 
	    << rMaxFine_ << " wafer width " << waferW_ << " gap among wafers "
	    << waferGap_ << " absorber width " << absorbW_ <<" absorber height "
	    << absorbH_ << " sectors " << sectors_ << std::endl;
#endif
  slopeB_       = vArgs["SlopeBottom"];
  slopeT_       = vArgs["SlopeTop"];
  zFront_       = vArgs["ZFront"];
  rMaxFront_    = vArgs["RMaxFront"];
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: Bottom slopes " << slopeB_[0] << ":" 
	    << slopeB_[1] << " and " << slopeT_.size() << " slopes for top" 
	    << std::endl;
  for (unsigned int i=0; i<slopeT_.size(); ++i)
    std::cout << "Block [" << i << "] Zmin " << zFront_[i] << " Rmax "
	      << rMaxFront_[i] << " Slope " << slopeT_[i] << std::endl;
#endif
  idNameSpace_  = DDCurrentNamespace::ns();
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: NameSpace " << idNameSpace_ << std::endl;
#endif
}

////////////////////////////////////////////////////////////////////
// DDHGCalTBModule methods...
////////////////////////////////////////////////////////////////////

void DDHGCalTBModule::execute(DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  std::cout << "==>> Constructing DDHGCalTBModule..." << std::endl;
#endif
  copies_.clear();
  constructLayers (parent(), cpv);
#ifdef EDM_ML_DEBUG
  std::cout << copies_.size() << " different wafer copy numbers" << std::endl;
#endif
  copies_.clear();
#ifdef EDM_ML_DEBUG
  std::cout << "<<== End of DDHGCalTBModule construction ..." << std::endl;
#endif
}

void DDHGCalTBModule::constructLayers(const DDLogicalPart& module, 
				      DDCompactView& cpv) {
  
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule test: \t\tInside Layers" << std::endl;
#endif
  double       zi(zMinBlock_);
  int          laymin(0);
  for (unsigned int i=0; i<layers_.size(); i++) {
    double  zo     = zi + layerThick_[i];
    double  routF  = rMax(zi);
    int     laymax = laymin+layers_[i];
    double  zz     = zi;
    double  thickTot(0);
    for (int ly=laymin; ly<laymax; ++ly) {
      int     ii     = layerType_[ly];
      int     copy   = copyNumber_[ii];
      double  rinB   = (layerSense_[ly] == 0) ? (zo*slopeB_[0]) : (zo*slopeB_[1]);
      zz            += (0.5*thick_[ii]);
      thickTot      += thick_[ii];

      std::string name = "HGCal"+names_[ii]+std::to_string(copy);
#ifdef EDM_ML_DEBUG
      std::cout << "DDHGCalTBModule test: Layer " << ly << ":" << ii 
		<< " Front " << zi << ", " << routF << " Back " << zo << ", " 
		<< rinB << " superlayer thickness " << layerThick_[i] 
		<< std::endl;
#endif
      DDName matName(DDSplit(materials_[ii]).first, 
		     DDSplit(materials_[ii]).second);
      DDMaterial matter(matName);
      DDLogicalPart glog;
      if (layerSense_[ly] == 0) {
	DDSolid solid = DDSolidFactory::box(DDName(name, idNameSpace_),
					    absorbW_, absorbH_, 0.5*thick_[ii]);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
	std::cout << "DDHGCalTBModule test: " << solid.name() 
		  << " box of dimension " << absorbW_ << ":" << absorbH_
		  << ":" << 0.5*thick_[ii] << std::endl;
#endif
      } else {
	DDSolid solid = DDSolidFactory::tubs(DDName(name, idNameSpace_), 
					     0.5*thick_[ii], rinB, routF, 0.0,
					     CLHEP::twopi);
	glog = DDLogicalPart(solid.ddname(), matter, solid);
#ifdef EDM_ML_DEBUG
	std::cout << "DDHGCalTBModule test: " << solid.name()
		  << " Tubs made of " << matName << " of dimensions " << rinB 
		  << ", " << routF << ", " << 0.5*thick_[ii] << ", 0.0, "
		  << CLHEP::twopi/CLHEP::deg << std::endl;
#endif
	positionSensitive(glog,layerSense_[ly],rinB,routF,cpv);
      }
      DDTranslation r1(0,0,zz);
      DDRotation rot;
      cpv.position(glog, module, copy, r1, rot);
      ++copyNumber_[ii];
#ifdef EDM_ML_DEBUG
      std::cout << "DDHGCalTBModule test: " << glog.name() << " number "
		<< copy << " positioned in " << module.name() << " at " << r1 
		<< " with " << rot << std::endl;
#endif
      zz += (0.5*thick_[ii]);
    } // End of loop over layers in a block
    zi     = zo;
    laymin = laymax;
    if (fabs(thickTot-layerThick_[i]) < 0.00001) {
    } else if (thickTot > layerThick_[i]) {
      edm::LogError("HGCalGeom") << "Thickness of the partition " << layerThick_[i]
				 << " is smaller than thickness " << thickTot
				 << " of all its components **** ERROR ****\n";
    } else if (thickTot < layerThick_[i]) {
      edm::LogWarning("HGCalGeom") << "Thickness of the partition " 
				   << layerThick_[i] << " does not match with "
				   << thickTot << " of the components\n";
    }
  }   // End of loop over blocks
}

double DDHGCalTBModule::rMax(double z) {
  double r(0);
#ifdef EDM_ML_DEBUG
  unsigned int ik(0);
#endif
  for (unsigned int k=0; k<slopeT_.size(); ++k) {
    if (z < zFront_[k]) break;
    r  = rMaxFront_[k] + (z - zFront_[k]) * slopeT_[k];
#ifdef EDM_ML_DEBUG
    ik = k;
#endif
  }
#ifdef EDM_ML_DEBUG
  std::cout << "rMax : " << z << ":" << ik << ":" << r << std::endl;
#endif
  return r;
}

void DDHGCalTBModule::positionSensitive(DDLogicalPart& glog, int type,
					double rin, double rout, 
					DDCompactView& cpv) {
  double ww   = (waferW_+waferGap_);
  double dx   = 0.5*ww;
  double dy   = 3.0*dx*tan(30.0*CLHEP::deg);
  double rr   = 2.0*dx*tan(30.0*CLHEP::deg);
  int    ncol = (int)(2.0*rout/ww) + 1;
  int    nrow = (int)(rout/(ww*tan(30.0*CLHEP::deg))) + 1;
  int    incm(0), inrm(0), kount(0);
  double xc[6], yc[6];
#ifdef EDM_ML_DEBUG
  std::cout << glog.ddname() << " rout " << rout << " Row " << nrow 
	    << " Column " << ncol << std::endl; 
#endif
  for (int nr=-nrow; nr <= nrow; ++nr) {
    int inr = (nr >= 0) ? nr : -nr;
    for (int nc=-ncol; nc <= ncol; ++nc) {
      int inc = (nc >= 0) ? nc : -nc;
      if (inr%2 == inc%2) {
	double xpos = nc*dx;
	double ypos = nr*dy;
        xc[0] = xpos+dx; yc[0] = ypos-0.5*rr;
        xc[1] = xpos+dx; yc[1] = ypos+0.5*rr;
        xc[2] = xpos;    yc[2] = ypos+rr;
        xc[3] = xpos-dx; yc[3] = ypos+0.5*rr;
        xc[4] = xpos+dx; yc[4] = ypos-0.5*rr;
        xc[5] = xpos;    yc[5] = ypos-rr;
        bool cornerAll(true);
        for (int k=0; k<6; ++k) {
          double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
          if (rpos < rin || rpos > rout) cornerAll = false;
        }
	if (cornerAll) {
          double rpos = std::sqrt(xpos*xpos+ypos*ypos);
	  DDTranslation tran(xpos, ypos, 0.0);
	  DDRotation rotation;
	  int copy = inr*100 + inc;
	  if (nc < 0) copy += 10000;
	  if (nr < 0) copy += 100000;
	  DDName name;
	  if (type == 1) {
	    name = (rpos < rMaxFine_) ? 
	      DDName(DDSplit(wafer_[0]).first, DDSplit(wafer_[0]).second) : 
	      DDName(DDSplit(wafer_[1]).first, DDSplit(wafer_[1]).second);
	  } else {
	    name = DDName(DDSplit(covers_[type-2]).first, 
			  DDSplit(covers_[type-2]).second); 
	  }
	  cpv.position(name, glog.ddname(), copy, tran, rotation);
	  if (inc > incm) incm = inc;
	  if (inr > inrm) inrm = inr;
	  kount++;
	  if (copies_.count(copy) == 0 && type == 1)
	    copies_.insert(copy);
#ifdef EDM_ML_DEBUG
	  std::cout << "DDHGCalTBModule: " << name << " number " << copy
		    << " positioned in " << glog.ddname() << " at " << tran 
		    << " with " << rotation << std::endl;
#endif
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "DDHGCalTBModule: # of columns " << incm << " # of rows " 
	    << inrm << " and " << kount << " wafers for " << glog.ddname()
	    << std::endl;
#endif
}
