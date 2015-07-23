#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

const double k_ScaleFromDDD = 0.1;
const double k_horizontalShift = 1.0;

HGCalDDDConstants::HGCalDDDConstants(const DDCompactView& cpv,
				     std::string& nam) {
  initialize(cpv, nam);
#ifdef DebugLog
  std::cout << "HGCalDDDConstants for " << nam << " initialized with " 
	    << layers(false) << ":" << layers(true) << " layers and " 
	    << sectors() << " sectors and maximum of " << maxCells(false)
	    << ":" << maxCells(true) << " cells" << std::endl;
#endif
}

HGCalDDDConstants::~HGCalDDDConstants() {}

std::pair<int,int> HGCalDDDConstants::assignCell(float x, float y, int lay,
						 int subSec, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  float alpha, h, bl, tl;
  
  //set default values
  std::pair<int,int> cellAssignment( (x>0)?1:0, -1 );
  if (reco) {
    h    =  moduler_[i].h;
    bl   =  moduler_[i].bl;
    tl   =  moduler_[i].tl;
    alpha=  moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
    cellAssignment=assignCell(x, y, h, bl, tl, alpha, index.second);
  } else {
    h    =  modules_[i].h;
    bl   =  modules_[i].bl;
    tl   =  modules_[i].tl;
    alpha=  modules_[i].alpha;
    cellAssignment=assignCell(x, y, h, bl, tl, alpha, index.second);
  }
  
  return cellAssignment;
}
  
std::pair<int,int> HGCalDDDConstants::assignCell(float x, float y, float h, 
						 float bl,float tl,float alpha,
						 float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
 
  float x0(x);
  int phiSector = (x0 > 0) ? 1 : 0;
  if      (alpha < 0) {x0 -= 0.5*(tl+bl); phiSector = 0;}
  else if (alpha > 0) {x0 += 0.5*(tl+bl); phiSector = 1;}


  //determine the i-y
  int ky    = floor((y+h)/cellSize);
  if( ky*cellSize> y+h) ky--;
  if(ky<0)              return std::pair<int,int>(-1,-1);
  if( (ky+1)*cellSize < (y+h) ) ky++;
  int max_ky_allowed=floor(2*h/cellSize);
  if(ky>max_ky_allowed-1) return std::pair<int,int>(-1,-1);
  
  //determine the i-x
  //notice we substitute y by the top of the candidate cell to reduce the dead zones
  int kx    = floor(fabs(x0)/cellSize);
  if( kx*cellSize > fabs(x0) ) kx--;
  if(kx<0)                     return std::pair<int,int>(-1,-1);
  if( (kx+1)*cellSize < fabs(x0) ) kx++;
  int max_kx_allowed=floor( ((ky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize) );
  if(kx>max_kx_allowed-1) return std::pair<int,int>(-1,-1);
  
  //count cells summing in rows until required height
  //notice the bottom of the cell must be used
  int icell(0);
  for (int iky=0; iky<ky; ++iky) {
    int cellsInRow( floor( ((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize) ) );
    icell += cellsInRow;
  }
  icell += kx;

  //return result
  return std::pair<int,int>(phiSector,icell);
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, int lay, int subSec,
					       bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  float alpha, h, bl, tl;
  if (reco) {
    h    =  moduler_[i].h;
    bl   =  moduler_[i].bl;
    tl   =  moduler_[i].tl;
    alpha=  moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  modules_[i].h;
    bl   =  modules_[i].bl;
    tl   =  modules_[i].tl;
    alpha=  modules_[i].alpha;
  }
  return findCell(cell, h, bl, tl, alpha, index.second);
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, float h, float bl, 
					       float tl, float alpha, 
					       float cellSize) const {

  //check if cell number is meaningful
  if(cell<0) return std::pair<int,int>(-1,-1);

  //parameterization of the boundary of the trapezoid
  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  int kx(cell), ky(0);
  int kymax( floor((2*h)/cellSize) );
  int testCell(0);
  for (int iky=0; iky<kymax; ++iky) {

    //check if adding all the cells in this row is above the required cell
    //notice the top of the cell is used to maximize space
    int cellsInRow( floor( ((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize) ) );
    if (testCell+cellsInRow > cell) break;
    testCell += cellsInRow;
    ky++;
    kx -= cellsInRow;
  }

  return std::pair<int,int>(kx,ky);
}

bool HGCalDDDConstants::isValid(int lay, int mod, int cell, bool reco) const {

  bool ok = ((lay > 0 && lay <= (int)(layers(reco))) && 
	     (mod > 0 && mod <= sectors()) &&
	     (cell >=0 && cell <= maxCells(lay,reco)));

#ifdef DebugLog
  if (!ok) std::cout << "HGCalDDDConstants: Layer " << lay << ":" 
		     << (lay > 0 && (lay <= (int)(layers(reco)))) << " Module "
		     << mod << ":" << (mod > 0 && mod <= sectors()) << " Cell "
		     << cell << ":" << (cell >=0 && cell <= maxCells(lay,reco))
		     << ":" << maxCells(reco) << std::endl; 
#endif
  return ok;
}

std::pair<float,float> HGCalDDDConstants::locateCell(int cell, int lay, 
						     int subSec, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<float,float>(999999.,999999.);
  std::pair<int,int> kxy = findCell(cell, lay, subSec, reco);
  float alpha, h, bl, tl;
  if (reco) {
    h    =  moduler_[i].h;
    bl   =  moduler_[i].bl;
    tl   =  moduler_[i].tl;
    alpha=  moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  modules_[i].h;
    bl   =  modules_[i].bl;
    tl   =  modules_[i].tl;
    alpha=  modules_[i].alpha;
  }
  float cellSize = index.second;
  float x        = (kxy.first+0.5)*cellSize;
  if      (alpha < 0) x -= 0.5*(tl+bl);
  else if (alpha > 0) x -= 0.5*(tl+bl);
  if (subSec != 1) x = -x;
  float y        = ((kxy.second+0.5)*cellSize-h);
  return std::pair<float,float>(x,y);
}

int HGCalDDDConstants::maxCells(bool reco) const {

  int cells(0);
  for (unsigned int i = 0; i<layers(reco); ++i) {
    int lay = reco ? depth_[i] : layer_[i];
    if (cells < maxCells(lay, reco)) cells = maxCells(lay, reco);
  }
  return cells;
}

int HGCalDDDConstants::maxCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  float h, bl, tl, alpha;
  if (reco) {
    h    =  moduler_[i].h;
    bl   =  moduler_[i].bl;
    tl   =  moduler_[i].tl;
    alpha=  moduler_[i].alpha;
  } else {
    h    =  modules_[i].h;
    bl   =  modules_[i].bl;
    tl   =  modules_[i].tl;
    alpha=  modules_[i].alpha;
  }
  return maxCells(h, bl, tl, alpha, index.second);
}

int HGCalDDDConstants::maxCells(float h, float bl, float tl, float alpha, 
				float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  
  int   ncells(0);
  //always use the top of the cell to maximize space
  int   kymax = floor((2*h)/cellSize);
  for (int iky=0; iky<kymax; ++iky)
    {
      int cellsInRow=floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
      ncells += cellsInRow;
    }

  return ncells;
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  float h     = (reco) ? moduler_[i].h : modules_[i].h;
  int   kymax = floor((2*h)/index.second);
  return kymax;
}

std::pair<int,int> HGCalDDDConstants::newCell(int cell, int layer, int sector, 
					      int subsector, int incrx, 
					      int incry, bool half) const {

  int subSec = half ? subsector : 0;
  std::pair<int,int> kxy = findCell(cell, layer, subSec, true);
  int kx = kxy.first + incrx;
  int ky = kxy.second + incry;
  if (ky < 0 || ky > maxRows(layer, true)) {
    cell = maxCells(true);
    return std::pair<int,int>(cell,sector*subsector);
  } else if (kx < 0) {
    kx        =-kx;
    subsector =-subsector;
  } else if (kx > maxCells(layer, true)) {
    kx       -= maxCells(layer, true);
    sector   += subsector;
    subsector =-subsector;
    if (sector < 1)             sector = nSectors;
    else if (sector > nSectors) sector = 1;
  }
  cell = newCell(kx, ky, layer, subSec);
  return std::pair<int,int>(cell,sector*subsector);
}

std::pair<int,int> HGCalDDDConstants::newCell(int cell, int lay, int subsector,
					      int incrz, bool half) const {
 
  int layer = lay + incrz;
  if (layer <= 0 || layer > (int)(layers(true))) return std::pair<int,int>(cell,0);
  int subSec = half ? subsector : 0;
  std::pair<float,float> xy = locateCell(cell, lay, subSec, true);
  std::pair<int,int>     kcell = assignCell(xy.first, xy.second, layer, subSec,
					    true);
  return std::pair<int,int>(kcell.second,layer);
}
  
int HGCalDDDConstants::newCell(int kx, int ky, int lay, int subSec) const {

  std::pair<int,float> index = getIndex(lay, true);
  int i = index.first;
  if (i < 0) return maxCells(true);
  float alpha    = (subSec == 0) ? modules_[i].alpha : subSec;
  float cellSize = index.second;
  float a        = (alpha==0) ? 
    (2*moduler_[i].h/(moduler_[i].tl-moduler_[i].bl)) :
    (moduler_[i].h/(moduler_[i].tl-moduler_[i].bl));
  float b        = 2*moduler_[i].h*moduler_[i].bl/
    (moduler_[i].tl-moduler_[i].bl);
  int icell(kx);
  for (int iky=0; iky<ky; ++iky)
    icell += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
  return icell;
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i >= 0) {
    float h, bl, tl, alpha;
    if (reco) {
      h    =  moduler_[i].h;
      bl   =  moduler_[i].bl;
      tl   =  moduler_[i].tl;
      alpha=  moduler_[i].alpha;
    } else {
      h    =  modules_[i].h;
      bl   =  modules_[i].bl;
      tl   =  modules_[i].tl;
      alpha=  modules_[i].alpha;
    }
    return numberCells(h, bl, tl, alpha, index.second);
  } else {
    std::vector<int> ncell;
    return ncell;
  }
}

std::vector<int> HGCalDDDConstants::numberCells(float h, float bl, 
						float tl, float alpha, 
						float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  int   kymax = floor((2*h)/cellSize);
  std::vector<int> ncell;
  for (int iky=0; iky<kymax; ++iky)
    ncell.push_back(floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize)));
  return ncell;
}

std::pair<int,int> HGCalDDDConstants::simToReco(int cell, int lay, 
						bool half) const {
  
  std::pair<int,float> index = getIndex(lay, false);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  float h  = modules_[i].h;
  float bl = modules_[i].bl;
  float tl = modules_[i].tl;
  float cellSize = cellFactor_[i]*index.second;
  
  std::pair<int,int> kxy = findCell(cell, h, bl, tl, modules_[i].alpha, index.second);
  int depth   = layerGroup_[i];
  if(depth<0) return std::pair<int,int>(-1,-1);
  int kx      = kxy.first/cellFactor_[i];
  int ky      = kxy.second/cellFactor_[i];

  float a     = (half) ? (h/(tl-bl)) : (2*h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  for (int iky=0; iky<ky; ++iky)
    kx += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
  
#ifdef DebugLog
  std::cout << "simToReco: input " << cell << ":" << lay << ":" << half
	    << " kxy " << kxy.first << ":" << kxy.second << " output "
    	    << kx << ":" << depth << " cell factor=" << cellFactor_[i] << std::endl;
#endif
  return std::pair<int,int>(kx,depth);
}

void HGCalDDDConstants::initialize(const DDCompactView& cpv, std::string name){
  
  nSectors = nCells = 0;
  
  std::string attribute = "Volume"; 
  std::string value     = name;
  DDValue val(attribute, value, 0);
  
  DDSpecificsFilter filter;
  filter.setCriteria(val, DDCompOp::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool ok = fv.firstChild();
  
  if (ok) {
    //Load the SpecPars
    loadSpecPars(fv);
    //Load the Geometry parameters
    loadGeometry(fv, name);
  }
}

void HGCalDDDConstants::loadGeometry(const DDFilteredView& _fv, 
				     const std::string & sdTag) {
 
  DDFilteredView fv = _fv;
  bool dodet(true), first(true);
  int  zpFirst(0);
  std::vector<hgtrform> trforms;
 
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
      HGCalDDDConstants::hgtrap mytr(lay,trp.x1(),trp.x2(),
				     0.5*(trp.y1()+trp.y2()),
				     trp.halfZ(),trp.alpha1());
      int subs = (trp.alpha1()>0 ? 1 : 0);
      if (std::find(layer_.begin(),layer_.end(),lay) == layer_.end()) {
	for (unsigned int k=0; k<cellSize_.size(); ++k) {
	  if (lay == (int)(k+1)) {
	    mytr.cellSize = cellSize_[k];
	    break;
	  }
	}
	modules_.push_back(mytr);
	if (layer_.size() == 0) nSectors = 1;
	layer_.push_back(lay);
      } else if (std::find(layer_.begin(),layer_.end(),lay) == layer_.begin()){
	if (zp == zpFirst) ++nSectors;
      }
      DD3Vector x, y, z;
      fv.rotation().GetComponents( x, y, z ) ;
      const CLHEP::HepRep3x3 rotation ( x.X(), y.X(), z.X(),
					x.Y(), y.Y(), z.Y(),
					x.Z(), y.Z(), z.Z() );
      const CLHEP::HepRotation hr ( rotation );
      const CLHEP::Hep3Vector h3v ( fv.translation().X(),
				    fv.translation().Y(),
				    fv.translation().Z()  ) ;
      HGCalDDDConstants::hgtrform mytrf(zp,lay,sec,subs);
      mytrf.h3v = h3v;
      mytrf.hr  = hr;
      trforms.push_back(mytrf);
    }
    dodet = fv.next();
  }
  if (layer_.size() != cellSize_.size()) {
    edm::LogError("HGCalGeom") << "HGCalDDDConstants : mismatch in # of bins " 
			       << layer_.size() << ":" << cellSize_.size()
			       << " between geometry and specpar";
    throw cms::Exception("DDException") << "HGCalDDDConstants: mismatch between geometry and specpar";
  }
  for (unsigned int i=0; i<layer_.size(); ++i) {
    for (unsigned int k=0; k<layer_.size(); ++k) {
      if (layer_[k] == (int)(i+1)) {
	layerIndex.push_back(k);
	break;
      }
    }
  }
#ifdef DebugLog
  std::cout << "HGCalDDDConstants finds " <<layerIndex.size() <<" modules for "
	    << sdTag << " with " << nSectors << " sectors and " 
	    << trforms.size() << " transformation matrices" << std::endl;
  for (unsigned int i=0; i<layerIndex.size(); ++i) {
    int k = layerIndex[i];
    std::cout << "Module[" << i << ":" << k << "] Layer " << layer_[k] << ":"
	      << modules_[k].lay << " dx " << modules_[k].bl << ":" 
	      << modules_[k].tl << " dy " << modules_[k].h << " dz " 
	      << modules_[k].dz << " alpha " << modules_[k].alpha << " cell " 
	      << modules_[k].cellSize << std::endl;
  }
#endif
  int depth(0);
  for (unsigned int i=0; i<layer_.size(); ++i) {
    bool first(true);
    float dz(0);
    for (unsigned int k=0; k<layerGroup_.size(); ++k) {
      if (layerGroup_[k] == (int)(i+1)) {
	if (first) {
	  depth_.push_back(i+1);
	  depthIndex.push_back(depth);
	  depth++;
	  moduler_.push_back(modules_[k]);
	  moduler_.back().lay = depth;
	  moduler_.back().bl *= k_ScaleFromDDD;
	  moduler_.back().tl *= k_ScaleFromDDD;
	  moduler_.back().h  *= k_ScaleFromDDD;
	  moduler_.back().dz *= k_ScaleFromDDD;
	  moduler_.back().cellSize *= (k_ScaleFromDDD*cellFactor_[k]);
	  dz    = moduler_.back().dz;
	  first = false;
	} else {
	  dz   += (k_ScaleFromDDD*modules_[k].dz);
	  moduler_.back().dz = dz;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "HGCalDDDConstants has " << depthIndex.size() << " depths" 
	    << std::endl;
  for (unsigned int i=0; i<depthIndex.size(); ++i) {
    int k = depthIndex[i];
    std::cout << "Module[" << i << ":" << k << "]  Depth " << depth_[k] 
	      << ":" << moduler_[k].lay << " dx " << moduler_[k].bl << ":" 
	      << moduler_[k].tl << " dy " << moduler_[k].h << " dz " 
	      << moduler_[k].dz << " alpha " << moduler_[k].alpha << " cell " 
	      << moduler_[k].cellSize << std::endl;
  }
#endif
  for (unsigned int i=0; i<layer_.size(); ++i) {
    for (unsigned int i1=0; i1<trforms.size(); ++i1) {
      if (!trforms[i1].used && layerGroup_[trforms[i1].lay-1] == (int)(i+1)) {
	trform_.push_back(trforms[i1]);
	trform_.back().h3v *= k_ScaleFromDDD;
	trform_.back().lay  = (i+1);
	trforms[i1].used    = true;
	int nz(1);
	for (unsigned int i2=i1+1; i2<trforms.size(); ++i2) {
	  if (!trforms[i2].used && trforms[i2].zp ==  trforms[i1].zp &&
	      layerGroup_[trforms[i2].lay-1] == (int)(i+1) &&
	      trforms[i2].sec == trforms[i1].sec &&
	      trforms[i2].subsec == trforms[i1].subsec) {
	    trform_.back().h3v += (k_ScaleFromDDD*trforms[i2].h3v);
	    nz++;
	    trforms[i2].used = true;
	  }
	}
	if (nz > 0) {
	  trform_.back().h3v /= nz;
	}
      }
    }
  }
#ifdef DebugLog
  std::cout << "Obtained " << trform_.size() << " transformation matrices"
	    << std::endl;
  for (unsigned int k=0; k<trform_.size(); ++k) {
    std::cout << "Matrix[" << k << "] (" << trform_[k].zp << "," 
	      << trform_[k].sec << "," << trform_[k].subsec << ","
	      << trform_[k].lay << ") " << " Trnaslation " << trform_[k].h3v 
	      << " Rotation " << trform_[k].hr;
  }
#endif
}

void HGCalDDDConstants::loadSpecPars(const DDFilteredView& fv) {

  DDsvalues_type sv(fv.mergedSpecifics());

  //Granularity in x-y plane
  nCells     = 0;
  cellSize_  = getDDDArray("Granularity",sv,nCells);
#ifdef DebugLog
  std::cout << "HGCalDDDConstants: " << nCells << " entries for cellSize_"
	    << std::endl;
  for (int i=0; i<nCells; i++) {
    std::cout << " [" << i << "] = " << cellSize_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((nCells-1)%8 != 7) std::cout << std::endl;
#endif

  //Grouping in the detector plane
  cellFactor_  = dbl_to_int(getDDDArray("GroupingXY",sv,nCells));
#ifdef DebugLog
  std::cout << "HGCalDDDConstants: " << nCells << " entries for cellFactor_"
	    << std::endl;
  for (int i=0; i<nCells; i++) {
    std::cout << " [" << i << "] = " << cellFactor_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((nCells-1)%8 != 7) std::cout << std::endl;
#endif

  //Grouping of layers
  layerGroup_  = dbl_to_int(getDDDArray("GroupingZ",sv,nCells));
#ifdef DebugLog
  std::cout << "HGCalDDDConstants: " << nCells << " entries for layerGroup_"
	    << std::endl;
  for (int i=0; i<nCells; i++) {
    std::cout << " [" << i << "] = " << layerGroup_[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((nCells-1)%8 != 7) std::cout << std::endl;
#endif
}

std::vector<double> HGCalDDDConstants::getDDDArray(const std::string & str, 
						   const DDsvalues_type & sv,
						   int & nmin) const {
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        edm::LogError("HGCalGeom") << "HGCalDDDConstants : # of " << str 
				   << " bins " << nval << " < " << nmin 
				   << " ==> illegal";
        throw cms::Exception("DDException") << "HGCalDDDConstants: cannot get array " << str;
      }
    } else {
      if (nval < 1 && nmin == 0) {
        edm::LogError("HGCalGeom") << "HGCalDDDConstants : # of " << str
				   << " bins " << nval << " < 2 ==> illegal"
				   << " (nmin=" << nmin << ")";
        throw cms::Exception("DDException") << "HGCalDDDConstants: cannot get array " << str;
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (nmin >= 0) {
      edm::LogError("HGCalGeom") << "HGCalDDDConstants: cannot get array "
				 << str;
      throw cms::Exception("DDException") << "HGCalDDDConstants: cannot get array " << str;
    }
    std::vector<double> fvec;
    nmin = 0;
    return fvec;
  }
}

std::pair<int,float> HGCalDDDConstants::getIndex(int lay, bool reco) const {

  if (lay<1 || lay>(int)(layerIndex.size())) return std::pair<int,float>(-1,0);
  if (reco && lay>(int)(depthIndex.size()))  return std::pair<int,float>(-1,0);
  int   i    = (reco ? depthIndex[lay-1] : layerIndex[lay-1]);
  float cell = (reco ? moduler_[i].cellSize : modules_[i].cellSize);
  return std::pair<int,float>(i,cell);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
