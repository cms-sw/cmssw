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
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/RegressionTest/interface/DDErrorDetection.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <unordered_set>

//#define EDM_ML_DEBUG

const double k_ScaleFromDDD = 0.1;

HGCalGeomParameters::HGCalGeomParameters() {
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters::HGCalGeomParameters() constructor\n";
#endif
}

HGCalGeomParameters::~HGCalGeomParameters() { 
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters::destructed!!!\n";
#endif
}

void HGCalGeomParameters::loadGeometrySquare(const DDFilteredView& _fv, 
					     HGCalParameters& php,
					     const std::string & sdTag) {
 
  DDFilteredView fv = _fv;
  bool dodet(true), first(true);
  int  zpFirst(0);
  std::vector<HGCalParameters::hgtrform> trforms;
  std::vector<bool>                      trformUse;
 
  while (dodet) {
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
      HGCalParameters::hgtrap mytr;
      mytr.lay = lay;   mytr.bl = trp.x1();   mytr.tl = trp.x2();
      mytr.h   = 0.5*(trp.y1()+trp.y2());     mytr.dz = trp.halfZ();
      mytr.alpha = trp.alpha1();              mytr.cellSize = 0;
      int subs = (trp.alpha1()>0 ? 1 : 0);
      if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
	  php.layer_.end()) {
	for (unsigned int k=0; k<php.cellSize_.size(); ++k) {
	  if (lay == (int)(k+1)) {
	    mytr.cellSize = php.cellSize_[k];
	    break;
	  }
	}
	php.fillModule(mytr,false);
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
      HGCalParameters::hgtrform mytrf;
      mytrf.zp    = zp;
      mytrf.lay   = lay;
      mytrf.sec   = sec;
      mytrf.subsec= subs;
      mytrf.h3v   = h3v;
      mytrf.hr    = hr;
      trforms.push_back(mytrf);
      trformUse.push_back(false);
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
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters finds " << php.layerIndex_.size() 
	    << " modules for " << sdTag << " with " << php.nSectors_ 
	    << " sectors and " << trforms.size() << " transformation matrices" 
	    << std::endl;
  for (unsigned int i=0; i<php.layerIndex_.size(); ++i) {
    int k = php.layerIndex_[i];
    std::cout << "Module[" << i << ":" << k << "] Layer " << php.layer_[k] 
	      << ":" << php.moduleLayS_[k] << " dx " << php.moduleBlS_[k] 
	      << ":" << php.moduleTlS_[k] << " dy "<< php.moduleHS_[k]
	      << " dz " << php.moduleDzS_[k] << " alpha "
	      << php.moduleAlphaS_[k] << " cell " << php.moduleCellS_[k] 
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
	  php.depthLayerF_.push_back(k);
	  ++depth;
	  HGCalParameters::hgtrap mytr = php.getModule(k,false);
	  mytr.lay = depth;
	  mytr.bl *= k_ScaleFromDDD;
	  mytr.tl *= k_ScaleFromDDD;
	  mytr.h  *= k_ScaleFromDDD;
	  mytr.dz *= k_ScaleFromDDD;
	  mytr.cellSize *= (k_ScaleFromDDD*php.cellFactor_[k]);
	  php.fillModule(mytr, true);
	  dz    = mytr.dz;
	  first = false;
	} else {
	  dz   += (k_ScaleFromDDD*php.moduleDzS_[k]);
	  php.moduleDzR_.back() = dz;
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters has " << php.depthIndex_.size() 
	    << " depths" << std::endl;
  for (unsigned int i=0; i<php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    std::cout << "Module[" << i << ":" << k <<"]  First Layer "
	      << php.depthLayerF_[i] << " Depth " << php.depth_[k] << ":" 
	      << php.moduleLayR_[k] << " dx " << php.moduleBlR_[k] << ":"
	      << php.moduleTlR_[k] << " dy " <<php.moduleHR_[k] << " dz " 
	      << php.moduleDzR_[k] << " alpha " << php.moduleAlphaR_[k] 
	      << " cellSize " << php.moduleCellR_[k] << std::endl;
  }
#endif
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (unsigned int i1=0; i1<trforms.size(); ++i1) {
      if (!trformUse[i1] && php.layerGroup_[trforms[i1].lay-1] == 
	  (int)(i+1)) {
	trforms[i1].h3v *= k_ScaleFromDDD;
	trforms[i1].lay  = (i+1);
	trformUse[i1]    = true;
	php.fillTrForm(trforms[i1]);
	int nz(1);
	for (unsigned int i2=i1+1; i2<trforms.size(); ++i2) {
	  if (!trformUse[i2] && trforms[i2].zp ==  trforms[i1].zp &&
	      php.layerGroup_[trforms[i2].lay-1] == (int)(i+1) &&
	      trforms[i2].sec == trforms[i1].sec &&
	      trforms[i2].subsec == trforms[i1].subsec) {
	    php.addTrForm(k_ScaleFromDDD*trforms[i2].h3v);
	    nz++;
	    trformUse[i2] = true;
	  }
	}
	if (nz > 0) {
	  php.scaleTrForm(double(1.0/nz));
	}
      }
    }
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Obtained " << php.trformIndex_.size() 
	    << " transformation matrices"  << std::endl;
  for (unsigned int k=0; k<php.trformIndex_.size(); ++k) {
    std::cout << "Matrix[" << k << "] (" << std::hex << php.trformIndex_[k] 
	      << std::dec << ") Trnaslation (" << php.trformTranX_[k]
	      << ", " << php.trformTranY_[k] << ", " << php.trformTranZ_[k] 
	      << ") Rotation (" << php.trformRotXX_[k] << ", "
	      << php.trformRotYX_[k] << ", " << php.trformRotZX_[k] << ", "
	      << php.trformRotXY_[k] << ", " << php.trformRotYY_[k] << ", "
	      << php.trformRotZY_[k] << ", " << php.trformRotXZ_[k] << ", "
	      << php.trformRotYZ_[k] << ", " << php.trformRotZZ_[k] << ")\n";
  }
#endif
}

void HGCalGeomParameters::loadGeometryHexagon(const DDFilteredView& _fv, 
					      HGCalParameters& php,
					      const std::string & sdTag1,
					      const DDCompactView* cpv,
					      const std::string & sdTag2,
					      const std::string & sdTag3) {
 
  DDFilteredView fv = _fv;
  bool dodet(true);
  std::map<int,HGCalGeomParameters::layerParameters> layers;
  std::vector<HGCalParameters::hgtrform> trforms;
  std::vector<bool>                      trformUse;
  
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    std::string name = sol.name();
    // Layers first
    std::vector<int> copy = fv.copyNumbers();
    int nsiz = (int)(copy.size());
    int lay  = (nsiz > 0) ? copy[nsiz-1] : 0;
    int zp   = (nsiz > 2) ? copy[nsiz-3] : -1;
    if (zp != 1) zp = -1;
    if (lay == 0) {
      edm::LogError("HGCalGeom") << "Funny layer # " << lay << " zp "
				 << zp << " in " << nsiz << " components";
      throw cms::Exception("DDException") << "Funny layer # " << lay;
    } else {
      if (std::find(php.layer_.begin(),php.layer_.end(),lay) == 
	  php.layer_.end()) php.layer_.push_back(lay);
      std::map<int,HGCalGeomParameters::layerParameters>::iterator itr = layers.find(lay);
      if (itr == layers.end()) {
	const DDTubs & tube = static_cast<DDTubs>(sol);
	double rin = k_ScaleFromDDD*tube.rIn();
	double rout= k_ScaleFromDDD*tube.rOut();
	double zp = k_ScaleFromDDD*fv.translation().Z();
	HGCalGeomParameters::layerParameters laypar(rin,rout,zp);
	layers[lay] = laypar;
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents( x, y, z ) ;
      const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
					x.Y(), y.Y(), z.Y(),
					x.Z(), y.Z(), z.Z() );
      const CLHEP::HepRotation hr ( rotation );
      double xx = k_ScaleFromDDD*fv.translation().X();
      if (std::abs(xx) < 0.001) xx = 0;
      double yy = k_ScaleFromDDD*fv.translation().Y();
      if (std::abs(yy) < 0.001) yy = 0;
      const CLHEP::Hep3Vector h3v ( xx, yy, fv.translation().Z() );
      HGCalParameters::hgtrform mytrf;
      mytrf.zp    = zp;
      mytrf.lay   = lay;
      mytrf.sec   = 0;
      mytrf.subsec= 0;
      mytrf.h3v   = h3v;
      mytrf.hr    = hr;
      trforms.push_back(mytrf);
      trformUse.push_back(false);
    }
    dodet = fv.next();
  }

  // Then wafers
  // This assumes layers are build starting from 1 (which on 25 Jan 2016, they were)
  // to ensure that new copy numbers are always added
  // to the end of the list.
  std::unordered_map<int32_t,int32_t> copies;
  HGCalParameters::layer_map   copiesInLayers(layers.size()+1);
  std::vector<int32_t>         wafer2copy;
  std::vector<HGCalGeomParameters::cellParameters> wafers;  
  std::string attribute = "Volume";
  DDValue val1(attribute, sdTag2, 0.0);
  DDSpecificsFilter filter1;
  filter1.setCriteria(val1, DDCompOp::equals);
  DDFilteredView fv1(*cpv);
  fv1.addFilter(filter1);
  bool ok = fv1.firstChild();
  if (!ok) {
    edm::LogError("HGCalGeom") << " Attribute " << val1
			       << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute " << val1
					<< " not found but needed.";
  } else {
    dodet = true;
    std::unordered_set<std::string> names;
    while (dodet) {
      const DDSolid & sol  = fv1.logicalPart().solid();
      std::string name = fv1.logicalPart().name();
      std::vector<int> copy = fv1.copyNumbers();
      int nsiz  = (int)(copy.size());
      int wafer = (nsiz > 0) ? copy[nsiz-1] : 0;
      int layer = (nsiz > 1) ? copy[nsiz-2] : 0;
      if (nsiz < 2) {
	edm::LogError("HGCalGeom") << "Funny wafer # " << wafer << " in "
				   << nsiz << " components";
	throw cms::Exception("DDException") << "Funny wafer # " << wafer;
      } else {          
	std::unordered_map<int32_t,int32_t>::iterator itr = copies.find(wafer);
	std::unordered_map<int32_t,int32_t>::iterator cpy = 
	  copiesInLayers[layer].find(wafer);
	if (itr != copies.end() && cpy == copiesInLayers[layer].end()) {
	  copiesInLayers[layer][wafer] = itr->second;
	}
	if (itr == copies.end()) {            
	  copies[wafer] = wafer2copy.size();
	  copiesInLayers[layer][wafer] = wafer2copy.size();
	  double xx = k_ScaleFromDDD*fv1.translation().X();
	  if (std::abs(xx) < 0.001) xx = 0;
	  double yy = k_ScaleFromDDD*fv1.translation().Y();
	  if (std::abs(yy) < 0.001) yy = 0;
	  wafer2copy.emplace_back(wafer);
	  GlobalPoint p(xx,yy,k_ScaleFromDDD*fv1.translation().Z());
	  HGCalGeomParameters::cellParameters cell(false,wafer,p);
	  wafers.emplace_back(cell);
	  if ( names.count(name) == 0 ) {
	    const DDPolyhedra & polyhedra = static_cast<DDPolyhedra>(sol);
	    std::vector<double> zv = polyhedra.zVec();
	    std::vector<double> rv = polyhedra.rMaxVec();
	    php.waferR_ = rv[0]/std::cos(30.0*CLHEP::deg);
	    double dz   = 0.5*(zv[1]-zv[0]);
	    HGCalParameters::hgtrap mytr;
	    mytr.lay = 1;           mytr.bl = php.waferR_; 
	    mytr.tl = php.waferR_;  mytr.h = php.waferR_; 
	    mytr.dz = dz;           mytr.alpha = 0.0;
	    mytr.cellSize = waferSize_;
	    php.fillModule(mytr,false);
	    names.insert(name);
	  }
	}
      }
      dodet = fv1.next();
    }
  }
  
  // Finally the cells
  std::map<int,int>         wafertype;
  std::map<int,HGCalGeomParameters::cellParameters> cellsf, cellsc;
  DDValue val2(attribute, sdTag3, 0.0);
  DDSpecificsFilter filter2;
  filter2.setCriteria(val2, DDCompOp::equals);
  DDFilteredView fv2(*cpv);
  fv2.addFilter(filter2);
  ok = fv2.firstChild();
  if (!ok) {
    edm::LogError("HGCalGeom") << " Attribute " << val2
			       << " not found but needed.";
    throw cms::Exception("DDException") << "Attribute " << val2
					<< " not found but needed.";
  } else {
    dodet = true;
    while (dodet) {
      const DDSolid & sol  = fv2.logicalPart().solid();
      std::string name = sol.name();
      std::vector<int> copy = fv2.copyNumbers();
      int nsiz = (int)(copy.size());
      int cellx= (nsiz > 0) ? copy[nsiz-1] : 0;
      int wafer= (nsiz > 1) ? copy[nsiz-2] : 0;
      int cell = cellx%1000;
      int type = cellx/1000;
      if (type != 1 && type != 2) {
	edm::LogError("HGCalGeom") << "Funny cell # " << cell << " type " 
				   << type << " in " << nsiz << " components";
	throw cms::Exception("DDException") << "Funny cell # " << cell;
      } else {
	std::map<int,int>::iterator ktr = wafertype.find(wafer);
	if (ktr == wafertype.end()) wafertype[wafer] = type;
	bool newc(false);
	std::map<int,HGCalGeomParameters::cellParameters>::iterator itr;
	double cellsize = php.cellSize_[0];
	if (type == 1) {
	  itr = cellsf.find(cell);
	  newc= (itr == cellsf.end());
	} else {
	  itr = cellsc.find(cell);
	  newc= (itr == cellsc.end());
	  cellsize = php.cellSize_[1];
	}
	if (newc) {
	  bool half = (name.find("Half") != std::string::npos);
	  double xx = k_ScaleFromDDD*fv2.translation().X();
	  double yy = k_ScaleFromDDD*fv2.translation().Y();
	  if (half) {
	    math::XYZPointD p1(-2.0*cellsize/9.0,0,0);
	    math::XYZPointD p2 = fv2.rotation()(p1);
	    xx += (k_ScaleFromDDD*(p2.X()));
	    yy += (k_ScaleFromDDD*(p2.Y()));
#ifdef EDM_ML_DEBUG
	    std::cout << "Type " << type << " Cell " << cellx << " local " 
		      << xx << ":" << yy  << " new " << p1 << ":" << p2 <<"\n";
#endif
	  }
	  HGCalGeomParameters::cellParameters cp(half,wafer,GlobalPoint(xx,yy,0));
	  if (type == 1) {
	    cellsf[cell] = cp;
	  } else {
	    cellsc[cell] = cp;
	  }
	}
      }
      dodet = fv2.next();
    }
  }
  
  if (((cellsf.size()+cellsc.size())==0) || (wafers.size()==0) || 
      (layers.size()==0)) {
    edm::LogError("HGCalGeom") << "HGCalGeomParameters : number of cells "
			       << cellsf.size() << ":" << cellsc.size()
			       << " wafers " << wafers.size() << " layers "
			       << layers.size() << " illegal";
    throw cms::Exception("DDException")
      << "HGCalGeomParameters: mismatch between geometry and specpar: cells "
      << cellsf.size() << ":" << cellsc.size() << " wafers " << wafers.size()
      << " layers " << layers.size();
  }

  for (unsigned int i=0; i<layers.size(); ++i) {
    for (std::map<int,HGCalGeomParameters::layerParameters>::iterator itr = layers.begin();
	 itr != layers.end(); ++itr) {
      if (itr->first == (int)(i+1)) {
	php.layerIndex_.push_back(i);
	php.rMinLayHex_.push_back(itr->second.rmin);
	php.rMaxLayHex_.push_back(itr->second.rmax);
	php.zLayerHex_.push_back(itr->second.zpos);
	break;
      }
    }
  }
  for (unsigned int i=0; i<php.layer_.size(); ++i) {
    for (unsigned int i1=0; i1<trforms.size(); ++i1) {
      if (!trformUse[i1] && php.layerGroup_[trforms[i1].lay-1] == 
	  (int)(i+1)) {
	trforms[i1].h3v *= k_ScaleFromDDD;
	trforms[i1].lay  = (i+1);
	trformUse[i1]    = true;
	php.fillTrForm(trforms[i1]);
	int nz(1);
	for (unsigned int i2=i1+1; i2<trforms.size(); ++i2) {
	  if (!trformUse[i2] && trforms[i2].zp ==  trforms[i1].zp &&
	      php.layerGroup_[trforms[i2].lay-1] == (int)(i+1)) {
	    php.addTrForm(k_ScaleFromDDD*trforms[i2].h3v);
	    nz++;
	    trformUse[i2] = true;
	  }
	}
	if (nz > 0) {
	  php.scaleTrForm(double(1.0/nz));
	}
      }
    }
  }

  double rmin = k_ScaleFromDDD*php.waferR_;
  for (unsigned i = 0; i < wafer2copy.size(); ++i ) {
    php.waferCopy_.push_back(wafer2copy[i]);
    php.waferPosX_.push_back(wafers[i].xyz.x());
    php.waferPosY_.push_back(wafers[i].xyz.y());
    std::map<int,int>::iterator ktr = wafertype.find(wafer2copy[i]);
    int typet = (ktr == wafertype.end()) ? 0 : (ktr->second);
    php.waferTypeT_.push_back(typet);
    double r = wafers[i].xyz.perp();
    int    type(3);
    for (int k=1; k<4; ++k) {
      if ((r+rmin)<=php.boundR_[k]) {
	type = k; break;
      }
    }
    php.waferTypeL_.push_back(type);
  }
  php.copiesInLayers_ = copiesInLayers;
  php.nSectors_ = (int)(php.waferCopy_.size());

  std::vector<HGCalGeomParameters::cellParameters>::const_iterator itrf = wafers.end();
  for (unsigned int i=0; i<cellsf.size(); ++i) {
    std::map<int,HGCalGeomParameters::cellParameters>::iterator itr = cellsf.find(i);
    if (itr == cellsf.end()) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: missing info for"
				 << " fine cell number " << i;
      throw cms::Exception("DDException")
	<< "HGCalGeomParameters: missing info for fine cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int    waf= (itr->second).wafer;
      std::pair<double,double> xy = cellPosition(wafers,itrf,waf,xx,yy);
      php.cellFineX_.push_back(xy.first);
      php.cellFineY_.push_back(xy.second);
      php.cellFineHalf_.push_back((itr->second).half);      
    }
  }
  itrf = wafers.end();
  for (unsigned int i=0; i<cellsc.size(); ++i) {
    std::map<int,HGCalGeomParameters::cellParameters>::iterator itr = cellsc.find(i);
    if (itr == cellsc.end()) {
      edm::LogError("HGCalGeom") << "HGCalGeomParameters: missing info for"
				 << " coarse cell number " << i;
      throw cms::Exception("DDException")
	<< "HGCalGeomParameters: missing info for coarse cell number " << i;
    } else {
      double xx = (itr->second).xyz.x();
      double yy = (itr->second).xyz.y();
      int    waf= (itr->second).wafer;
      std::pair<double,double> xy = cellPosition(wafers,itrf,waf,xx,yy);
      php.cellCoarseX_.push_back(xy.first);
      php.cellCoarseY_.push_back(xy.second);
      php.cellCoarseHalf_.push_back((itr->second).half);   
    }
  }
  int depth(0);
  for (unsigned int i=0; i<php.layerGroup_.size(); ++i) {
    bool first(true);
    for (unsigned int k=0; k<php.layerGroup_.size(); ++k) {
      if (php.layerGroup_[k] == (int)(i+1)) {
	if (first) {
	  php.depth_.push_back(i+1);
	  php.depthIndex_.push_back(depth);
	  php.depthLayerF_.push_back(k);
	  ++depth;
	  first = false;
	}
      }
    }
  }
  HGCalParameters::hgtrap mytr = php.getModule(0, false);
  mytr.bl   *= k_ScaleFromDDD;
  mytr.tl   *= k_ScaleFromDDD;
  mytr.h    *= k_ScaleFromDDD;
  mytr.dz   *= k_ScaleFromDDD;
  double dz  = mytr.dz;
  php.fillModule(mytr, true);
  mytr.dz = 2*dz;
  php.fillModule(mytr, true);
  mytr.dz = 3*dz;
  php.fillModule(mytr, true);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters finds " << php.zLayerHex_.size() 
	    << " layers" << std::endl;
  for (unsigned int i=0; i<php.zLayerHex_.size(); ++i) {
    int k = php.layerIndex_[i];
    std::cout << "Layer[" << i << ":" << k << ":" << php.layer_[k] 
	      << "] with r = " << php.rMinLayHex_[i] << ":" 
	      << php.rMaxLayHex_[i] << " at z = "  << php.zLayerHex_[i] 
	      << std::endl;
  }
  std::cout << "HGCalGeomParameters has " << php.depthIndex_.size() 
	    << " depths" <<std::endl;
  for (unsigned int i=0; i<php.depthIndex_.size(); ++i) {
    int k = php.depthIndex_[i];
    std::cout << "Reco Layer[" << i << ":" << k  << "]  First Layer "  
	      << php.depthLayerF_[i] << " Depth " << php.depth_[k] <<std::endl;
  }
  std::cout << "HGCalGeomParameters finds " << php.nSectors_ << " wafers\n";
  for (unsigned int i=0; i<php.waferCopy_.size(); ++i) 
    std::cout << "Wafer[" << i << ": " << php.waferCopy_[i] << "] type "
	      << php.waferTypeL_[i] << ":" << php.waferTypeT_[i] << " at (" 
	      << php.waferPosX_[i] << "," << php.waferPosY_[i] << ",0)\n";
  std::cout << "HGCalGeomParameters: wafer radius "  << php.waferR_ 
	    << " and dimensions of the wafers:" << std::endl;
  std::cout << "Sim[0] " << php.moduleLayS_[0] << " dx " << php.moduleBlS_[0]
	    << ":" << php.moduleTlS_[0] << " dy " << php.moduleHS_[0] << " dz "
	    << php.moduleDzS_[0] << " alpha " << php.moduleAlphaS_[0] << "\n";
  for (unsigned int k=0; k<php.moduleLayR_.size(); ++k)
    std::cout << "Rec[" << k << "] " << php.moduleLayR_[k] << " dx " 
	      << php.moduleBlR_[k] << ":" << php.moduleTlR_[k] << " dy " 
	      << php.moduleHR_[k]  << " dz " << php.moduleDzR_[k] << " alpha "
	      << php.moduleAlphaR_[k] << std::endl;
  std::cout << "HGCalGeomParameters finds " << php.cellFineX_.size() 
	    << " fine cells in a  wafer" << std::endl;
  for (unsigned int i=0; i<php.cellFineX_.size(); ++i) 
    std::cout << "Fine Cell[" << i << "] at (" << php.cellFineX_[i] << ","
	      << php.cellFineY_[i] << ",0)" << std::endl;
  std::cout << "HGCalGeomParameters finds " << php.cellCoarseX_.size() 
	    << " coarse cells in a wafer" << std::endl;
  for (unsigned int i=0; i<php.cellCoarseX_.size(); ++i) 
    std::cout << "Coarse Cell[" << i << "] at (" << php.cellCoarseX_[i]
	      << "," << php.cellCoarseY_[i] << ",0)" << std::endl;
  std::cout << "Obtained " << php.trformIndex_.size() 
	    << " transformation matrices"  << std::endl;
  for (unsigned int k=0; k<php.trformIndex_.size(); ++k) {
    std::cout << "Matrix[" << k << "] (" << std::hex << php.trformIndex_[k]
	      << std::dec << ") Trnaslation (" << php.trformTranX_[k]
	      << ", " << php.trformTranY_[k] << ", " << php.trformTranZ_[k] 
	      << " Rotation ("  << php.trformRotXX_[k] << ", "
	      << php.trformRotYX_[k] << ", " << php.trformRotZX_[k] << ", "
	      << php.trformRotXY_[k] << ", " << php.trformRotYY_[k] << ", "
	      << php.trformRotZY_[k] << ", " << php.trformRotXZ_[k] << ", "
	      << php.trformRotYZ_[k] << ", " << php.trformRotZZ_[k] << ")\n";
  }
  std::cout << "Dump copiesInLayers for " << php.copiesInLayers_.size()
	    << " layers\n";
  for (unsigned int k=0; k<php.copiesInLayers_.size(); ++k) {
    const auto& theModules = php.copiesInLayers_[k];
    std::cout << "Layer " << k << ":" << theModules.size() << std::endl;
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr=theModules.begin();
	 itr != theModules.end(); ++itr) {
      std::cout << " " << itr->first << ":" << itr->second;
      ++k2;
      if (k2 > 9) { std::cout << std::endl; k2 = 0; }
    }
    if (k2 > 0) std::cout << std::endl;
  }
#endif
}

void HGCalGeomParameters::loadSpecParsSquare(const DDFilteredView& fv,
					     HGCalParameters& php) {

  DDsvalues_type sv(fv.mergedSpecifics());
  //Granularity in x-y plane
  php.nCells_    = 0;
  php.cellSize_  = getDDDArray("Granularity",sv,php.nCells_);
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: " << php.nCells_ 
	    << " entries for cellSize_"  << std::endl;
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.cellSize_[i] << std::endl;
  }
#endif

  //Grouping in the detector plane
  php.cellFactor_  = dbl_to_int(getDDDArray("GroupingXY",sv,php.nCells_));
  int nmin = 1;
  std::vector<double> slp = getDDDArray("Slope",sv,nmin);
  php.slopeMin_    = slp[0];
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: minimum slope " << php.slopeMin_
	    << " and " << php.nCells_ << " entries for cellFactor_\n";
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.cellFactor_[i] << std::endl;
  }
#endif
  
  //Grouping of layers
  php.layerGroup_  = dbl_to_int(getDDDArray("GroupingZ",sv,php.nCells_));
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: " << php.nCells_ 
	    << " entries for layerGroup_" << std::endl;
  for (int i=0; i<php.nCells_; i++) {
    std::cout << " [" << i << "] = " << php.layerGroup_[i] << std::endl;
  }
#endif
}

void HGCalGeomParameters::loadSpecParsHexagon(const DDFilteredView& fv,
					      HGCalParameters& php,
					      const DDCompactView* cpv,
					      const std::string & sdTag1,
					      const std::string & sdTag2) {

  DDsvalues_type sv(fv.mergedSpecifics());
  int nmin(4);
  php.boundR_ = getDDDArray("RadiusBound",sv,nmin);
  for (unsigned int k=0; k<php.boundR_.size(); ++k) 
    php.boundR_[k] *= k_ScaleFromDDD;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: wafer radius ranges for cell grouping " 
	    << php.boundR_[0] << ":" << php.boundR_[1] << ":"
	    << php.boundR_[2] << ":" << php.boundR_[3] << std::endl;
#endif
  nmin = 2;
  php.rLimit_ = getDDDArray("RadiusLimits",sv,nmin);
  for (unsigned int k=0; k<php.rLimit_.size(); ++k) 
    php.rLimit_[k] *= k_ScaleFromDDD;
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: Minimum/maximum R " 
	    << php.rLimit_[0] << ":" << php.rLimit_[1] << "\n";
#endif
  nmin = 0;
  std::vector<int> ndummy = dbl_to_int(getDDDArray("LevelTop",sv,nmin));
  php.levelT_ = ndummy[0];
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: LevelTop " << php.levelT_ << "\n";
#endif

  //Grouping of layers
  nmin = 0;
  php.layerGroup_  = dbl_to_int(getDDDArray("GroupingZFine",sv,nmin));
  php.layerGroupM_ = dbl_to_int(getDDDArray("GroupingZMid",sv,nmin));
  php.layerGroupO_ = dbl_to_int(getDDDArray("GroupingZOut",sv,nmin));
  nmin = 1;
  std::vector<double> slp = getDDDArray("Slope",sv,nmin);
  php.slopeMin_    = slp[0];
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: minimum slope " << php.slopeMin_ 
	    << " and layer groupings for the 3 ranges:"  << std::endl;
  for (int k=0; k<nmin; ++k)
    std::cout << "[" << k << "] " << php.layerGroup_[k] << ":" 
	      << php.layerGroupM_[k] << ":"  << php.layerGroupO_[k] << "\n";
#endif

  //Wafer size
  std::string attribute = "Volume";
  DDValue val1(attribute, sdTag1, 0.0);
  DDSpecificsFilter filter1;
  filter1.setCriteria(val1, DDCompOp::equals);
  DDFilteredView fv1(*cpv);
  fv1.addFilter(filter1);
  if (fv1.firstChild()) {
    DDsvalues_type sv(fv1.mergedSpecifics());
    int nmin(0);
    std::vector<double> dummy = getDDDArray("WaferSize",sv,nmin);
    waferSize_ = dummy[0];
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: Wafer Size: " << waferSize_ << std::endl;
#endif

  //Cell size
  DDValue val2(attribute, sdTag2, 0.0);
  DDSpecificsFilter filter2;
  filter2.setCriteria(val2, DDCompOp::equals);
  DDFilteredView fv2(*cpv);
  fv2.addFilter(filter2);
  if (fv2.firstChild()) {
    DDsvalues_type sv(fv2.mergedSpecifics());
    int nmin(0);
    php.cellSize_ = getDDDArray("CellSize",sv,nmin);
  }
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalGeomParameters: " << php.cellSize_.size() 
	    << " cells of sizes:\n";
  for (unsigned int k=0; k<php.cellSize_.size(); ++k)
    std::cout << " [" << k << "] " << php.cellSize_[k] << "\n";
#endif

}

void HGCalGeomParameters::loadWaferHexagon(HGCalParameters& php) {

  double waferW(k_ScaleFromDDD*waferSize_), rmin(k_ScaleFromDDD*php.waferR_);
  double rin(php.rLimit_[0]), rout(php.rLimit_[1]), rMaxFine(php.boundR_[1]);
#ifdef EDM_ML_DEBUG
  std::cout << "Input waferWidth " << waferW << ":" << rmin
	    << " R Limits: "  << rin << ":" << rout 
	    << " Fine " << rMaxFine << "\n";
#endif
  // Clear the vectors
  php.waferCopy_.clear();
  php.waferTypeL_.clear();
  php.waferTypeT_.clear();
  php.waferPosX_.clear();
  php.waferPosY_.clear();
  double dx   = 0.5*waferW;
  double dy   = 3.0*dx*tan(30.0*CLHEP::deg);
  double rr   = 2.0*dx*tan(30.0*CLHEP::deg);
  int    ncol = (int)(2.0*rout/waferW) + 1;
  int    nrow = (int)(rout/(waferW*tan(30.0*CLHEP::deg))) + 1;
  int    incm(0), inrm(0), kount(0), ntot(0);
  double xc[6], yc[6];
  HGCalParameters::layer_map copiesInLayers(php.layer_.size()+1);
#ifdef EDM_ML_DEBUG
  std::cout << "Row " << nrow << " Column " << ncol << std::endl;
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
        bool cornerOne(false);
	bool cornerAll(true);
	for (int k=0; k<6; ++k) {
	  double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
	  if (rpos >= rin && rpos <= rout) cornerOne = true;
	  else                             cornerAll = false;
	}
	double rpos = std::sqrt(xpos*xpos+ypos*ypos);
	int typet = (rpos < rMaxFine) ? 1 : 2;
	int typel(3);
	for (int k=1; k<4; ++k) {
	  if ((rpos+rmin)<=php.boundR_[k]) {
	    typel = k; break;
	  }
	}
        ++ntot;
        if (cornerOne) {
          int copy = inr*100 + inc;
          if (nc < 0) copy += 10000;
          if (nr < 0) copy += 100000;
          if (inc > incm) incm = inc;
          if (inr > inrm) inrm = inr;
          kount++;
#ifdef EDM_ML_DEBUG
	  std::cout << kount << ":" << ntot << " Copy " << copy
		    << " Type " << typel << ":" << typet
		    << " Location " << cornerOne << ":" << cornerAll
		    << " Position " << xpos << ":" << ypos << "\n";
#endif
	  php.waferCopy_.push_back(copy);
	  php.waferTypeL_.push_back(typel);
	  php.waferTypeT_.push_back(typet);
	  php.waferPosX_.push_back(xpos);
	  php.waferPosY_.push_back(ypos);
	  for (unsigned int il=0; il<php.layer_.size(); ++il) {
	    bool corner(false);
	    cornerAll = true;
	    for (int k=0; k<6; ++k) {
	      double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
	      if (rpos >= php.rMinLayHex_[il] && 
		  rpos <= php.rMaxLayHex_[il]) corner    = true;
	      else                             cornerAll = false;
	    }
	    if (corner) {
	      std::unordered_map<int32_t,int32_t>::iterator cpy = 
		copiesInLayers[php.layer_[il]].find(copy);
	      if (cpy == copiesInLayers[php.layer_[il]].end()) 
		copiesInLayers[php.layer_[il]][copy] = cornerAll ? php.waferCopy_.size() : -1;
	    }
	  }
	}
      }
    }
  }
  php.copiesInLayers_ = copiesInLayers;
  php.nSectors_       = (int)(php.waferCopy_.size());
#ifdef EDM_ML_DEBUG
  std::cout << "HGCalWaferHexagon: # of columns " << incm << " # of rows " 
	    << inrm << " and " << kount << ":" << ntot  << " wafers; R "
	    << rin << ":" << rout << std::endl;
  std::cout << "Dump copiesInLayers for " << php.copiesInLayers_.size()
	    << " layers\n";
  for (unsigned int k=0; k<copiesInLayers.size(); ++k) {
    const auto& theModules = copiesInLayers[k];
    std::cout << "Layer " << k << ":" << theModules.size() << std::endl;
    int k2(0);
    for (std::unordered_map<int, int>::const_iterator itr=theModules.begin();
	 itr != theModules.end(); ++itr) {
      std::cout << " " << itr->first << ":" << itr->second;
      ++k2;
      if (k2 > 9) { std::cout << std::endl; k2 = 0; }
    }
    if (k2 > 0) std::cout << std::endl;
  }
#endif
}

void HGCalGeomParameters::loadCellParsHexagon(const DDCompactView* cpv,
                                              HGCalParameters& php) {

  //Special parameters for cell parameters
  std::string attribute = "OnlyForHGCalNumbering"; 
  std::string value     = "any";
  DDValue val1(attribute, value, 0.0);
  DDSpecificsFilter filter1;
  filter1.setCriteria(val1, DDCompOp::not_equals,
                      DDLogOp::AND, true, // compare strings otherwise doubles
                      true  // use merged-specifics or simple-specifics
                      );
  DDFilteredView fv1(*cpv);
  fv1.addFilter(filter1);
  bool ok = fv1.firstChild();

  if (ok) {
    php.cellFine_   = dbl_to_int(DDVectorGetter::get("waferFine"));
    php.cellCoarse_ = dbl_to_int(DDVectorGetter::get("waferCoarse"));
  }

#ifdef EDM_ML_DEBUG
  std::cout << "HGCalLoadCellPars: " << php.cellFine_.size() 
	    << " rows for fine cells\n";
  for (unsigned int k=0; k<php.cellFine_.size(); ++k)
    std::cout << k << ":" << php.cellFine_[k] << " ";
  std::cout << std::endl;
  std::cout << "HGCalLoadCellPars: " <<php.cellCoarse_.size()
            << " rows for coarse cells\n";
  for (unsigned int k=0; k<php.cellCoarse_.size(); ++k)
    std::cout << k << ":" << php.cellCoarse_[k] << " ";
  std::cout << std::endl;
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
				   << " bins " << nval << " < 1 ==> illegal"
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

std::pair<double,double>
HGCalGeomParameters::cellPosition(const std::vector<HGCalGeomParameters::cellParameters>& wafers, 
				  std::vector<HGCalGeomParameters::cellParameters>::const_iterator& itrf,
				  int wafer, double xx, double yy) {

  if (itrf == wafers.end()) {
    for (std::vector<HGCalGeomParameters::cellParameters>::const_iterator itr = wafers.begin();
	 itr != wafers.end(); ++itr) {
      if (itr->wafer == wafer) {
	itrf = itr;
	break;
      }
    }
  }
  double dx(0), dy(0);
  if (itrf != wafers.end()) {
    dx = (xx - itrf->xyz.x());
    if (std::abs(dx) < 0.001) dx = 0;
    dy = (yy - itrf->xyz.y());
    if (std::abs(dy) < 0.001) dy = 0;
  }
  return std::pair<double,double>(dx,dy);
}
