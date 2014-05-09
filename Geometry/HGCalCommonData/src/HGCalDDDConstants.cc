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

HGCalDDDConstants::HGCalDDDConstants() : tobeInitialized(true) {}

HGCalDDDConstants::HGCalDDDConstants(const DDCompactView& cpv,
				     std::string& nam) : tobeInitialized(true){
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
  float alpha = (reco && subSec == 0) ? modules_[i].alpha : subSec;
  return assignCell(x, y, modules_[i].h, modules_[i].bl, modules_[i].tl, alpha,
		    index.second);
}
  
std::pair<int,int> HGCalDDDConstants::assignCell(float x, float y, float h, 
						 float bl,float tl,float alpha,
						 float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  float x0(x);
  if      (alpha < 0) x0 -= 0.5*(tl+bl);
  else if (alpha > 0) x0 += 0.5*(tl+bl);
  int phiSector = (x0 > 0) ? 1 : 0;

  int icell = floor(fabs(x0)/cellSize);
  int ky    = floor((y+h)/cellSize);
  for (int iky=0; iky<ky; ++iky)
    icell += floor((iky*cellSize+b)/(a*cellSize));
  return std::pair<int,int>(phiSector,icell);
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, int lay, int subSec,
					       bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  float alpha = (reco && subSec == 0) ? modules_[i].alpha : subSec;
  return findCell(cell, modules_[i].h, modules_[i].bl, modules_[i].tl, alpha,
		  index.second);
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, float h, float bl, 
					       float tl, float alpha, 
					       float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  int   kymax = floor((2*h)/cellSize);
  int   ky(0), testCell(0);
  for (int iky=0; iky<kymax; ++iky) {
    int deltay(floor((iky*cellSize+b)/(a*cellSize)));
    if (testCell+deltay > cell) break;
    testCell += deltay;
    ky++;
  }
  int kx = (cell-testCell);
  return std::pair<int,int>(kx,ky);
}

void HGCalDDDConstants::initialize(const DDCompactView& cpv, std::string name){

  if (tobeInitialized) {
    tobeInitialized = false;
    nSectors = nCells = 0;

    std::string attribute = "Volume"; 
    std::string value     = name;
    DDValue val(attribute, value, 0);
  
    DDSpecificsFilter filter;
    filter.setCriteria(val, DDSpecificsFilter::equals);
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
}

std::pair<float,float> HGCalDDDConstants::locateCell(int cell, int lay, 
						     int subSec, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<float,float>(999999.,999999.);
  std::pair<int,int> kxy = findCell(cell, lay, subSec, reco);
  float h        = modules_[i].h;
  float cellSize = index.second;
  return std::pair<float,float>((kxy.first+0.5)*cellSize,(kxy.second+0.5)*cellSize-h);
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
  return maxCells(modules_[i].h, modules_[i].bl, modules_[i].tl, 
		  modules_[i].alpha, index.second);
}

int HGCalDDDConstants::maxCells(float h, float bl, float tl, float alpha, 
				float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  int   ncell(0);
  int   kymax = floor((2*h)/cellSize);
  for (int iky=0; iky<kymax; ++iky)
    ncell += floor((iky*cellSize+b)/(a*cellSize));
  return ncell;
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  int   kymax = floor((2*modules_[i].h)/index.second);
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
  if (layer <= 0 || layer < (int)(layers(true))) return std::pair<int,int>(cell,0);
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
    (2*modules_[i].h/(modules_[i].tl-modules_[i].bl)) :
    (modules_[i].h/(modules_[i].tl-modules_[i].bl));
  float b        = 2*modules_[i].h*modules_[i].bl/
    (modules_[i].tl-modules_[i].bl);
  int icell(kx);
  for (int iky=0; iky<ky; ++iky)
    icell += floor((iky*cellSize+b)/(a*cellSize));
  return icell;
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i >= 0) {
    return numberCells(modules_[i].h, modules_[i].bl, modules_[i].tl, 
		       modules_[i].alpha, index.second);
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
    ncell.push_back(floor((iky*cellSize+b)/(a*cellSize)));
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
  float cellSize = modules_[i].cellRec;
  std::pair<int,int> kxy = findCell(cell, h, bl, tl, modules_[i].alpha,
				    index.second);
  int depth   = layerGroup_[i];
  int kx      = kxy.first/cellFactor_[i];
  int ky      = kxy.second/cellFactor_[i];
  float a     = (half) ? (h/(tl-bl)) : (2*h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  for (int iky=0; iky<ky; ++iky)
    kx += floor((iky*cellSize+b)/(a*cellSize));

  return std::pair<int,int>(kx,depth);
}

void HGCalDDDConstants::checkInitialized() const {
  if (tobeInitialized) {
    edm::LogError("HGCalGeom") << "HGCalDDDConstants : to be initialized correctly";
    throw cms::Exception("DDException") << "HGCalDDDConstants: to be initialized";
  }
} 

void HGCalDDDConstants::loadGeometry(const DDFilteredView& _fv, 
				     std::string & sdTag) {
 
  DDFilteredView fv = _fv;
  bool dodet = true;
 
  while (dodet) {
    //    DDTranslation    t   = fv.translation();
    const DDSolid & sol  = fv.logicalPart().solid();
    std::string name = sol.name();
    int isd = (name.find(sdTag) == std::string::npos) ? -1 : 1;
    if (isd > 0) {
      std::vector<int> copy = fv.copyNumbers();
      int nsiz = (int)(copy.size());
      int lay  = (nsiz > 0) ? copy[nsiz-1] : -1;
      const DDTrap & trp = static_cast<DDTrap>(sol);
      HGCalDDDConstants::hgtrap mytr(trp.x1(),trp.x2(),0.5*(trp.y1()+trp.y2()),
				     trp.halfZ(),trp.alpha1());
      if (std::find(layer_.begin(),layer_.end(),lay) == layer_.end()) {
	modules_.push_back(mytr);
	layer_.push_back(lay);
      } else if (std::find(layer_.begin(),layer_.end(),lay) == layer_.begin()){
	++nSectors;
      }
    }
    dodet = fv.next();
  }
  if (nSectors > 0) ++nSectors;
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
	modules_[k].cellSim = cellSize_[i];
	modules_[k].cellRec = cellSize_[i]*cellFactor_[i];
	break;
      }
    }
  }
  for (unsigned int i=0; i<layer_.size(); ++i) {
    for (unsigned int k=0; k<layerGroup_.size(); ++k) {
      if (layerGroup_[k] == (int)(i+1)) {
	depth_.push_back(i+1);
	depthIndex.push_back(layerIndex[k]);
	break;
      }
    }
  }
#ifdef DebugLog
  std::cout << "HGCalDDDConstants finds " <<layerIndex.size() <<" modules for "
	    << sdTag << std::endl;
  for (unsigned int i=0; i<layerIndex.size(); ++i) {
    int k = layerIndex[i];
    std::cout << "Module[" << i << "] Layer " << layer_[k] << " dx "
	      << modules_[k].bl << ":" << modules_[k].tl << " dy " 
	      << modules_[k].h << " dz " << modules_[k].dz << " alpha "
	      << modules_[k].alpha << " cell " << modules_[k].cellSim
	      << ":" << modules_[k].cellRec << std::endl;
  }
  std::cout << "HGCalDDDConstants has " << depth_.size() << " depths" 
	    << std::endl;
  for (unsigned int i=0; i<depth_.size(); ++i) {
    std::cout << " Depth[" << i << "] " << depth_[i] << ":" << depthIndex[i];
    if (i%8 == 7) std::cout << std::endl;
  }
  if ((depth_.size()-1)%8 != 7) std::cout << std::endl;
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

  if (lay<0 || lay>(int)(layerIndex.size())) return std::pair<int,float>(-1,0);
  if (reco && lay>(int)(depthIndex.size()))  return std::pair<int,float>(-1,0);
  int   i = (reco ? depthIndex[lay-1] : layerIndex[lay-1]);
  float cell = (reco ? modules_[i].cellRec : modules_[i].cellSim);
  return std::pair<int,float>(i,cell);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
