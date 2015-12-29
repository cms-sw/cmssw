#include "Geometry/HGCalCommonData/interface/HGCalGeomParameters.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDConstant.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "CondFormats/GeometryObjects/interface/HGCalParameters.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

const double k_ScaleFromDDD = 0.1;

HGCalGeomParameters::HGCalGeomParameters() {
#ifdef DebugLog
  std::cout << "HGCalGeomParameters::HGCalGeomParameters() constructor" << std::endl;
#endif
}

HGCalGeomParameters::~HGCalGeomParameters() { 
#ifdef DebugLog
  std::cout << "HGCalGeomParameters::destructed!!!" << std::endl;
#endif
}

void HGCalGeomParameters::loadGeometrySquare(const DDFilteredView& _fv, 
					     HGCalParameters& php,
					     const std::string & sdTag) {
 
  DDFilteredView fv = _fv;
  bool dodet(true), first(true);
  int  zpFirst(0);
  std::vector<HGCalParameters::hgtrform> trforms;
 
  while (dodet) {
    //    DDTranslation    t   = fv.translation();
    const DDSolid & sol  = fv.logicalPart().solid();
    std::string name = sol.name();
    int isd = (name.find(sdTag) == std::string::npos) ? -1 : 1;
    if (isd > 0) {
      std::vector<int> copy = fv.copyNumbers();
      int nsiz = (int)(copy.size());
      int lay  = (nsiz > 0) ? copy[nsiz-1] : -1;
      int sec  = (nsiz > 1) ? copy[nsiz-2] : -1;
      int zp   = (nsiz > 3) ? copy[nsiz-4] : -1;
      if (zp !=1 ) zp = -1;
      if (first) {first = false; zpFirst = zp;}
      const DDTrap & trp = static_cast<DDTrap>(sol);
      HGCalParameters::hgtrap mytr(lay,trp.x1(),trp.x2(),
				   0.5*(trp.y1()+trp.y2()),
				   trp.halfZ(),trp.alpha1());
      int subs = (trp.alpha1()>0 ? 1 : 0);
      if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
	  php.layer_.end()) {
	for (unsigned int k=0; k<php.cellSize_.size(); ++k) {
	  if (lay == (int)(k+1)) {
	    mytr.cellSize = php.cellSize_[k];
	    break;
	  }
	}
	php.modules_.push_back(mytr);
	if (php.layer_.size() == 0) php.nSectors_ = 1;
	php.layer_.push_back(lay);
      } else if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
		 php.layer_.begin()) {
	if (zp == zpFirst) ++(php.nSectors_);
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents( x, y, z ) ;
      const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
					x.Y(), y.Y(), z.Y(),
					x.Z(), y.Z(), z.Z() );
      const CLHEP::HepRotation hr ( rotation );
      const CLHEP::Hep3Vector h3v ( fv.translation().X(),
				    fv.translation().Y(),
				    fv.translation().Z()  );
      HGCalParameters::hgtrform mytrf(zp,lay,sec,subs);
      mytrf.h3v = h3v;
      mytrf.hr  = hr;
      trforms.push_back(mytrf);
    }
    dodet = fv.next();
  }
  if (php.layer_.size() != php.cellSize_.size()) {
    edm::LogError("HGCalGeom") << "HGCalGeomParameters : mismatch in # of bins "
			       << php.layer_.size() << ":" << php.cellSize_.size()
			       << " between geometry and specpar";
    throw cms::Exception("DDException") << "HGCalGeomParameters: mismatch between geometry and specpar";
  }
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (unsigned int k=0; k<php.layer_.size(); ++k) {
      if (php.layer_[k] == (int)(i+1)) {
	php.layerIndex_.push_back(k);
	break;
      }
    }
  }
#ifdef DebugLog
  std::cout << "HGCalGeomParameters finds " << php.layerIndex_.size() 
	    << " modules for "  << sdTag << " with " << php.nSectors_ 
	    << " sectors and " << trforms.size() << " transformation matrices" 
	    << std::endl;
  for (unsigned int i=0; i<layerIndex_.size(); ++i) {
    int k = layerIndex_[i];
    std::cout << "Module[" << i << ":" << k << "] Layer " << php.layer_[k] 
	      << ":" << php.modules_[k].lay << " dx " << php.modules_[k].bl 
	      << ":" << php.modules_[k].tl << " dy " << php.modules_[k].h 
	      << " dz " << php.modules_[k].dz << " alpha " 
	      << php.modules_[k].alpha << " cell " << php.modules_[k].cellSize 
	      << std::endl;
  }
#endif
  int depth(0);
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    bool first(true);
    float dz(0);
    for (unsigned int k=0; k<php.layerGroup_.size(); ++k) {
      if (php.layerGroup_[k] == (int)(i+1)) {
	if (first) {
	  php.depth_.push_back(i+1);
	  php.depthIndex_.push_back(depth);
	  ++depth;
	  php.moduler_.push_back(php.modules_[k]);
	  php.moduler_.back().lay = depth;
	  php.moduler_.back().bl *= k_ScaleFromDDD;
	  php.moduler_.back().tl *= k_ScaleFromDDD;
	  php.moduler_.back().h  *= k_ScaleFromDDD;
	  php.moduler_.back().dz *= k_ScaleFromDDD;
	  php.moduler_.back().cellSize *= (k_ScaleFromDDD*php.cellFactor_[k]);
	  dz    = php.moduler_.back().dz;
	  first = false;
	} else {
	  dz   += (k_ScaleFromDDD*php.modules_[k].dz);
	  php.moduler_.back().dz = dz;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "HGCalGeomParameters has " << php.depthIndex_.size() << " depths"
	    << std::endl;
  for (unsigned int i=0; i<php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    std::cout << "Module[" << i << ":" << k << "]  Depth " << php.depth_[k] 
	      << ":" << php.moduler_[k].lay << " dx " << php.moduler_[k].bl 
	      << ":" << php.moduler_[k].tl << " dy " << php.moduler_[k].h 
	      << " dz " << php.moduler_[k].dz << " alpha " 
	      << php.moduler_[k].alpha << " cell " << php.moduler_[k].cellSize 
	      << std::endl;
  }
#endif
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (unsigned int i1=0; i1<trforms.size(); ++i1) {
      if (!trforms[i1].used && php.layerGroup_[trforms[i1].lay-1] == 
	  (int)(i+1)) {
	php.trform_.push_back(trforms[i1]);
	php.trform_.back().h3v *= k_ScaleFromDDD;
	php.trform_.back().lay  = (i+1);
	trforms[i1].used        = true;
	int nz(1);
	for (unsigned int i2=i1+1; i2<trforms.size(); ++i2) {
	  if (!trforms[i2].used && trforms[i2].zp ==  trforms[i1].zp &&
	      php.layerGroup_[trforms[i2].lay-1] == (int)(i+1) &&
	      trforms[i2].sec == trforms[i1].sec &&
	      trforms[i2].subsec == trforms[i1].subsec) {
	    php.trform_.back().h3v += (k_ScaleFromDDD*trforms[i2].h3v);
	    nz++;
	    trforms[i2].used = true;
	  }
	}
	if (nz > 0) {
	  php.trform_.back().h3v /= nz;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "Obtained " << php.trform_.size() << " transformation matrices"
	    << std::endl;
  for (unsigned int k=0; k<php.trform_.size(); ++k) {
    std::cout << "Matrix[" << k << "] (" << php.trform_[k].zp << "," 
	      << php.trform_[k].sec << "," << php.trform_[k].subsec << ","
	      << php.trform_[k].lay << ") " << " Trnaslation " 
	      << php.trform_[k].h3v << " Rotation " << php.trform_[k].hr;
  }
#endif
}

void HGCalGeomParameters::loadSpecParsSquare(const DDFilteredView& fv,
					     HGCalParameters& php) {

  DDsvalues_type sv(fv.mergedSpecifics());

  //Granularity in x-y plane
  php.nCells_    = 0;
  php.cellSize_  = getDDDArray("Granularity",sv,php.nCells_);
#ifdef DebugLog
  std::cout << "HGCalGeomParameters: " << php.nCells_ <<" entries for cellSize_"
	    << std::endl;
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.cellSize_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((php.nCells_-1)%8 != 7) std::cout << std::endl;
#endif

  //Grouping in the detector plane
  php.cellFactor_  = dbl_to_int(getDDDArray("GroupingXY",sv,php.nCells_));
#ifdef DebugLog
  std::cout << "HGCalGeomParameters: " << php.nCells_ 
	    << " entries for cellFactor_" << std::endl;
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.cellFactor_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((php.nCells_-1)%8 != 7) std::cout << std::endl;
#endif

  //Grouping of layers
  php.layerGroup_  = dbl_to_int(getDDDArray("GroupingZ",sv,php.nCells_));
#ifdef DebugLog
  std::cout << "HGCalGeomParameters: " << php.nCells_ 
	    << " entries for layerGroup_" << std::endl;
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.layerGroup_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((php.nCells_-1)%8 != 7) std::cout << std::endl;
#endif
}

std::vector<double> HGCalGeomParameters::getDDDArray(const std::string & str, 
						     const DDsvalues_type & sv,
						     int & nmin) {
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        edm::LogError("HGCalGeom") << "HGCalGeomParameters : # of " << str 
				   << " bins " << nval << " < " << nmin 
				   << " ==> illegal";
        throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
      }
    } else {
      if (nval < 1 && nmin == 0) {
        edm::LogError("HGCalGeom") << "HGCalGeomParameters : # of " << str
				   << " bins " << nval << " < 2 ==> illegal"
				   << " (nmin=" << nmin << ")";
        throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (nmin >= 0) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: cannot get array "
				 << str;
      throw cms::Exception("DDException") << "HGCalGeomParameters: cannot get array " << str;
    }
    std::vector<double> fvec;
    nmin = 0;
    return fvec;
  }
}
