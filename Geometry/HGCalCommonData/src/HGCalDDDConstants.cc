#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "DetectorDescription/Base/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define DebugLog

const double k_horizontalShift = 1.0;
const double k_ScaleFromDDD = 0.1;

HGCalDDDConstants::HGCalDDDConstants(const HGCalParameters* hp,
				     const std::string name) : hgpar_(hp), tan30deg_(std::tan(30.0*CLHEP::deg)) {

  if (geomMode() == HGCalGeometryMode::Square) {
    rmax_ = 0;
    edm::LogInfo("HGCalGeom") << "HGCalDDDConstants initialized for " << name
			      << " with " << layers(false) << ":" <<layers(true)
			      << " layers, " << sectors() << " sectors and "
			      << "maximum of " << maxCells(false) << ":" 
			      << maxCells(true) << " cells";
#ifdef DebugLog
    std::cout << "HGCalDDDConstants initialized for " << name << " with " 
	      << layers(false) << ":" << layers(true) << " layers, " 
	      << sectors() << " sectors and maximum of " << maxCells(false)
	      << ":" << maxCells(true) << " cells" << std::endl;
#endif
  } else {
    rmax_ = k_ScaleFromDDD * (hgpar_->waferR_) * std::cos(30.0*CLHEP::deg);
    edm::LogInfo("HGCalGeom") << "HGCalDDDConstants initialized for " << name
			      << " with " << layers(false) << ":" <<layers(true)
			      << " layers, " << wafers() << " wafers and "
			      << "maximum of " << maxCells(false) << ":" 
			      << maxCells(true) << " cells";
#ifdef DebugLog
    std::cout << "HGCalDDDConstants initialized for " << name << " with " 
	      << layers(false) << ":" << layers(true) << " layers, " 
	      << wafers() << " wafers and " << "maximum of " << maxCells(false)
	      << ":" << maxCells(true) << " cells" << std::endl;
#endif
  }
}

HGCalDDDConstants::~HGCalDDDConstants() {}

std::pair<int,int> HGCalDDDConstants::assignCell(float x, float y, int lay,
						 int subSec, bool reco) const {
  std::pair<int,float> index = getIndex(lay, reco);
  std::pair<int,int> cellAssignment(-1,-1);
  int i = index.first;
  if (i < 0) return cellAssignment;
  if (geomMode() == HGCalGeometryMode::Square) {
    float alpha, h, bl, tl;
    getParameterSquare(i,subSec,reco,h,bl,tl,alpha);
    cellAssignment = assignCellSquare(x, y, h, bl, tl, alpha, index.second);
  } else {
    float xx = (reco) ? x : k_ScaleFromDDD*x;
    float yy = (reco) ? y : k_ScaleFromDDD*y;
    cellAssignment = assignCellHexagon(xx,yy);
  }
  return cellAssignment;
}
  
std::pair<int,int> HGCalDDDConstants::assignCellSquare(float x, float y, 
						       float h, float bl,
						       float tl, float alpha,
						       float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
 
  float x0(x);
  int phiSector = (x0 > 0) ? 1 : 0;
  if      (alpha < 0) {x0 -= 0.5*(tl+bl); phiSector = 0;}
  else if (alpha > 0) {x0 += 0.5*(tl+bl); phiSector = 1;}
  
    //determine the i-y
  int ky    = floor((y+h)/cellSize);
  if (ky*cellSize> y+h) ky--;
  if (ky<0)             return std::pair<int,int>(-1,-1);
  if ((ky+1)*cellSize < (y+h) ) ky++;
  int max_ky_allowed=floor(2*h/cellSize);
  if (ky>max_ky_allowed-1) return std::pair<int,int>(-1,-1);
    
  //determine the i-x
  //notice we substitute y by the top of the candidate cell to reduce the dead zones
  int kx    = floor(fabs(x0)/cellSize);
  if (kx*cellSize > fabs(x0) ) kx--;
  if (kx<0)                    return std::pair<int,int>(-1,-1);
  if ((kx+1)*cellSize < fabs(x0)) kx++;
  int max_kx_allowed=floor( ((ky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize) );
  if (kx>max_kx_allowed-1) return std::pair<int,int>(-1,-1);
  
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

std::pair<int,int> HGCalDDDConstants::assignCellHexagon(float x, 
							float y) const {
  double xx(x), yy(y);
  //First the wafer
  int wafer = cellHex(xx, yy, rmax_, hgpar_->waferPos_);
  // Now the cell
  xx -= hgpar_->waferPos_[wafer].x();
  yy -= hgpar_->waferPos_[wafer].y();
  int cell(0);
  if (hgpar_->waferTypeT_[wafer] == 1) 
    cell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[0], hgpar_->cellFine_);
  else
    cell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[1], hgpar_->cellCoarse_);
  return std::pair<int,int>(wafer,cell);
}

double HGCalDDDConstants::cellSizeHex(int type) const {
  int    indx = (type == 1) ? 0 : 1;
  double cell = (0.5*k_ScaleFromDDD*hgpar_->cellSize_[indx]);
  return cell;
}

unsigned int HGCalDDDConstants::layers(bool reco) const {
  return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, int lay, int subSec,
					       bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  if (geomMode() == HGCalGeometryMode::Hexagon) {
    return std::pair<int,int>(-1,-1);
  } else {
    float alpha, h, bl, tl;
    getParameterSquare(i,subSec,reco,h,bl,tl,alpha);
    return findCellSquare(cell, h, bl, tl, alpha, index.second);
  }
}

std::pair<int,int> HGCalDDDConstants::findCellSquare(int cell, float h,
						     float bl, float tl, 
						     float alpha, 
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
    int cellsInRow(floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize)));
    if (testCell+cellsInRow > cell) break;
    testCell += cellsInRow;
    ky++;
    kx -= cellsInRow;
  }

  return std::pair<int,int>(kx,ky);
}

std::vector<HGCalParameters::hgtrap>::const_iterator 
HGCalDDDConstants::getModule(int wafer) const {

  std::vector<HGCalParameters::hgtrap>::const_iterator itr;
  int type = ((wafer>=0)&&(wafer<(int)(hgpar_->waferTypeL_.size()))) ? 
    hgpar_->waferTypeL_[wafer] : 3;
  int indx(1);
  for (itr = hgpar_->moduler_.begin(); itr != hgpar_->moduler_.end();
       ++itr, ++indx) {
    if (type == indx) return itr;
  }
  return hgpar_->moduler_.end();
}

bool HGCalDDDConstants::isValid(int lay, int mod, int cell, bool reco) const {

  bool ok(false);
  int  cellmax(0), modmax(0);
  if (geomMode() == HGCalGeometryMode::Square) {
    cellmax = maxCells(lay,reco);
    modmax  = sectors();
    ok      = ((lay > 0 && lay <= (int)(layers(reco))) && 
	       (mod > 0 && mod <= modmax) &&
	       (cell >=0 && cell <= cellmax));
  } else {
    modmax = modules(lay,reco);
    ok = ((lay > 0 && lay <= (int)(layers(reco))) && 
	  (mod > 0 && mod <= modmax));
    if (ok) {
      cellmax = (hgpar_->waferTypeT_[mod]==1) ? 
	(int)(hgpar_->cellFine_.size()) : (int)(hgpar_->cellCoarse_.size());
      ok = (cell >=0 && cell <=  cellmax);
    }
  }
    
#ifdef DebugLog
  if (!ok) std::cout << "HGCalDDDConstants: Layer " << lay << ":" 
		     << (lay > 0 && (lay <= (int)(layers(reco)))) << " Module "
		     << mod << ":" << (mod > 0 && mod <= modmax) << " Cell "
		     << cell << ":" << (cell >=0 && cell <= cellmax)
		     << ":" << maxCells(reco) << std::endl; 
#endif
  return ok;
}

std::pair<float,float> HGCalDDDConstants::locateCell(int cell, int lay, 
						     int type, bool reco) const {
  // type refers to subsector # for square cell and wafer # for hexagon cell
  float x(999999.), y(999999.);
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<float,float>(x,y);
  if (geomMode() == HGCalGeometryMode::Square) {
    std::pair<int,int> kxy = findCell(cell, lay, type, reco);
    float alpha, h, bl, tl;
    getParameterSquare(i,type,reco,h,bl,tl,alpha);
    float cellSize = index.second;
    x              = (kxy.first+0.5)*cellSize;
    if      (alpha < 0) x -= 0.5*(tl+bl);
    else if (alpha > 0) x -= 0.5*(tl+bl);
    if (type != 1) x = -x;
    y              = ((kxy.second+0.5)*cellSize-h);
  } else {
    x              = hgpar_->waferPos_[type].x();
    y              = hgpar_->waferPos_[type].y();
    if (hgpar_->waferTypeT_[type] == 1) {
      x           += hgpar_->cellFine_[cell].x();
      y           += hgpar_->cellFine_[cell].y();
    } else {
      x           += hgpar_->cellCoarse_[cell].x();
      y           += hgpar_->cellCoarse_[cell].y();
    }
    if (!reco) {
      x           /= k_ScaleFromDDD;
      y           /= k_ScaleFromDDD;
    }
  }
  return std::pair<float,float>(x,y);
}

std::pair<float,float> HGCalDDDConstants::locateCellHex(int cell, int wafer, 
							bool reco) const {
  float x(0), y(0);
  if (hgpar_->waferTypeT_[wafer] == 1) {
    x  = hgpar_->cellFine_[cell].x();
    y  = hgpar_->cellFine_[cell].y();
  } else {
    x  = hgpar_->cellCoarse_[cell].x();
    y  = hgpar_->cellCoarse_[cell].y();
  }
  if (!reco) {
    x /= k_ScaleFromDDD;
    y /= k_ScaleFromDDD;
  }
  return std::pair<float,float>(x,y);
}

int HGCalDDDConstants::maxCells(bool reco) const {

  int cells(0);
  for (unsigned int i = 0; i<layers(reco); ++i) {
    int lay = reco ? hgpar_->depth_[i] : hgpar_->layer_[i];
    if (cells < maxCells(lay, reco)) cells = maxCells(lay, reco);
  }
  return cells;
}

int HGCalDDDConstants::maxCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  if (geomMode() == HGCalGeometryMode::Square) {
    float h, bl, tl, alpha;
    getParameterSquare(i,0,reco,h,bl,tl,alpha);
    return maxCellsSquare(h, bl, tl, alpha, index.second);
  } else {
    unsigned int cells(0);
    for (unsigned int k=0; k<hgpar_->waferPos_.size(); ++k) {
      if (waferInLayer(k,index.first)) {
	unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	  (hgpar_->cellFine_.size()) : (hgpar_->cellCoarse_.size());
	if (cell > cells) cells = cell;
      }
    }
    return (int)(cells);
  }
}

int HGCalDDDConstants::maxCellsSquare(float h, float bl, float tl, 
				      float alpha, float cellSize) const {

  float a     = (alpha==0) ? (2*h/(tl-bl)) : (h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  
  int   ncells(0);
  //always use the top of the cell to maximize space
  int   kymax = floor((2*h)/cellSize);
  for (int iky=0; iky<kymax; ++iky) {
    int cellsInRow=floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
    ncells += cellsInRow;
  }

  return ncells;
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  int kymax(0);
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return kymax;
  if (geomMode() == HGCalGeometryMode::Square) {
    float h = (reco) ? hgpar_->moduler_[i].h : hgpar_->modules_[i].h;
    kymax   = floor((2*h)/index.second);
  } else {
    for (unsigned int k=0; k<hgpar_->waferPos_.size(); ++k) {
      if (waferInLayer(k,i)) {
	int ky = ((hgpar_->waferCopy_[k])/100)%100;
	if (ky > kymax) kymax = ky;
      }
    }
  }
  return kymax;
}

int HGCalDDDConstants::modules(int lay, bool reco) const {
  int nmod(0);
  std::pair<int,float> index = getIndex(lay, reco);
  if (index.first < 0) return nmod;
  for (unsigned int k=0; k<hgpar_->waferPos_.size(); ++k) {
    if (waferInLayer(k,index.first)) ++nmod;
  }
  return nmod;
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
    if (sector < 1)                      sector = hgpar_->nSectors_;
    else if (sector > hgpar_->nSectors_) sector = 1;
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
  float alpha    = (subSec == 0) ? hgpar_->modules_[i].alpha : subSec;
  float cellSize = index.second;
  float a        = (alpha==0) ? 
    (2*hgpar_->moduler_[i].h/(hgpar_->moduler_[i].tl-hgpar_->moduler_[i].bl)) :
    (hgpar_->moduler_[i].h/(hgpar_->moduler_[i].tl-hgpar_->moduler_[i].bl));
  float b        = 2*hgpar_->moduler_[i].h*hgpar_->moduler_[i].bl/
    (hgpar_->moduler_[i].tl-hgpar_->moduler_[i].bl);
  int icell(kx);
  for (int iky=0; iky<ky; ++iky)
    icell += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
  return icell;
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  std::vector<int> ncell;
  if (i >= 0) {
    if (geomMode() == HGCalGeometryMode::Square) {
      float h, bl, tl, alpha;
      getParameterSquare(i,0,reco,h,bl,tl,alpha);
      return numberCellsSquare(h, bl, tl, alpha, index.second);
    } else {
      for (unsigned int k=0; k<hgpar_->waferPos_.size(); ++k) {
	if (waferInLayer(k,i)) {
	  unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	    (hgpar_->cellFine_.size()) : (hgpar_->cellCoarse_.size());
	  ncell.push_back((int)(cell));
	}
      }
    }
  }
  return ncell;
}

std::vector<int> HGCalDDDConstants::numberCellsSquare(float h, float bl, 
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

int HGCalDDDConstants::numberCellsHexagon(int wafer) const {

  int ncell(0);
  if (wafer >= 0 && wafer < (int)(hgpar_->waferTypeT_.size())) {
    if (hgpar_->waferTypeT_[wafer]==1) 
      ncell = (int)(hgpar_->cellFine_.size());
    else 
      ncell = (int)(hgpar_->cellCoarse_.size());
  }
  return ncell;
}

std::pair<int,int> HGCalDDDConstants::simToReco(int cell, int lay, int mod,
						bool half) const {
  
  std::pair<int,float> index = getIndex(lay, false);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  int kx(-1), depth(-1);
  if (geomMode() == HGCalGeometryMode::Square) {
    float h  = hgpar_->modules_[i].h;
    float bl = hgpar_->modules_[i].bl;
    float tl = hgpar_->modules_[i].tl;
    float cellSize = hgpar_->cellFactor_[i]*index.second;
  
    std::pair<int,int> kxy = findCellSquare(cell, h, bl, tl, hgpar_->modules_[i].alpha, index.second);
    depth       = hgpar_->layerGroup_[i];
    if (depth<0) return std::pair<int,int>(-1,-1);
    kx          = kxy.first/hgpar_->cellFactor_[i];
    int ky      = kxy.second/hgpar_->cellFactor_[i];

    float a     = (half) ? (h/(tl-bl)) : (2*h/(tl-bl));
    float b     = 2*h*bl/(tl-bl);
    for (int iky=0; iky<ky; ++iky)
      kx += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
#ifdef DebugLog
    std::cout << "simToReco: input " << cell << ":" << lay << ":" << half
	      << " kxy " << kxy.first << ":" << kxy.second << " output "
	      << kx << ":" << depth << " cell factor=" 
	      << hgpar_->cellFactor_[i] << std::endl;
#endif
  } else {
    kx  = cell;
    int type = hgpar_->waferTypeL_[mod];
    if (type == 1) {
      depth = hgpar_->layerGroup_[i];
    } else if (type == 2) {
      depth = hgpar_->layerGroupM_[i];
    } else {
      depth = hgpar_->layerGroupO_[i];
    }
  }
  return std::pair<int,int>(kx,depth);
}

int HGCalDDDConstants::waferFromCopy(int copy) const {
  int wafer = wafers();
  for (int k=0; k<wafers(); ++k) {
    if (copy == hgpar_->waferCopy_[k]) {
      wafer = k;
      break;
    }
  }
  return wafer;
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay, bool reco) const {

  std::pair<int,float> indx = getIndex(lay, reco);
  if (indx.first < 0) return false;
  return waferInLayer(wafer,indx.first);
}

int HGCalDDDConstants::wafers() const {

  int wafer(0);
  for (unsigned int i = 0; i<layers(true); ++i) {
    int lay = hgpar_->depth_[i];
    wafer += modules(lay, true);
  }
  return wafer;
}

int HGCalDDDConstants::cellHex(double xx, double yy, const double& cellR, 
			       const std::vector<GlobalPoint>& pos) const {
  int num(0);
  const double tol(0.00001);
  double cellY = 2.0*cellR*tan30deg_;
  for (unsigned int k=0; k<pos.size(); ++k) {
    double dx = std::abs(xx - pos[k].x());
    double dy = std::abs(yy - pos[k].y());
    if (dx <= (cellR+tol) && dy <= (cellY+tol)) {
      double xmax = (dy<=0.5*cellY) ? cellR : (cellR-(dy-0.5*cellY)/tan30deg_);
      if (dx <= (xmax+tol)) {
	num = k;
	break;
      }
    }
  }
  return num;
}

std::pair<int,float> HGCalDDDConstants::getIndex(int lay, bool reco) const {

  if (lay<1 || lay>(int)(hgpar_->layerIndex_.size())) return std::pair<int,float>(-1,0);
  if (reco && lay>(int)(hgpar_->depthIndex_.size()))  return std::pair<int,float>(-1,0);
  int   indx(0);
  float cell(0);
  if (geomMode() == HGCalGeometryMode::Square) {
    indx = (reco ? hgpar_->depthIndex_[lay-1] : hgpar_->layerIndex_[lay-1]);
    cell = (reco ? hgpar_->moduler_[indx].cellSize : hgpar_->modules_[indx].cellSize);
  } else {
    indx = (reco ? hgpar_->depthLayerF_[lay-1] : hgpar_->layerIndex_[lay-1]);
    cell = (reco ? hgpar_->moduler_[0].cellSize : hgpar_->modules_[0].cellSize);
  }
  return std::pair<int,float>(indx,cell);
}

void HGCalDDDConstants::getParameterSquare(int lay, int subSec, bool reco, 
					   float& h, float& bl, float& tl,
					   float& alpha) const {
  if (reco) {
    h    =  hgpar_->moduler_[lay].h;
    bl   =  hgpar_->moduler_[lay].bl;
    tl   =  hgpar_->moduler_[lay].tl;
    alpha=  hgpar_->moduler_[lay].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  hgpar_->modules_[lay].h;
    bl   =  hgpar_->modules_[lay].bl;
    tl   =  hgpar_->modules_[lay].tl;
    alpha=  hgpar_->modules_[lay].alpha;
  }
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay) const {

  double rr   = 2*rmax_*tan30deg_;
  double rpos = hgpar_->waferPos_[wafer].perp();
  bool   in   = (rpos-rr >= hgpar_->rMinLayHex_[lay] && 
		 rpos+rr <= hgpar_->rMaxLayHex_[lay]);
  return in;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
