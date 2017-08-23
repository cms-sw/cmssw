#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

constexpr double k_horizontalShift = 1.0;
constexpr double k_ScaleFromDDD = 0.1;

HGCalDDDConstants::HGCalDDDConstants(const HGCalParameters* hp,
				     const std::string& name) : hgpar_(hp) {
  mode_ = hgpar_->mode_;
  if (mode_ == HGCalGeometryMode::Square) {
    rmax_    = 0;
    modHalf_ = sectors()*layers(true);
    for (int simreco = 0; simreco < 2; ++simreco ) {
      tot_layers_[simreco] = layersInit((bool)simreco);
    }

    edm::LogInfo("HGCalGeom") << "HGCalDDDConstants initialized for " << name
			      << " with " << layers(false) << ":" 
			      << layers(true) << " layers, " << sectors() 
			      << " sectors " << 2*modHalf_ << " modules and "
			      << "maximum of " << maxCells(false) << ":" 
			      << maxCells(true) << " cells";
#ifdef EDM_ML_DEBUG
    std::cout << "HGCalDDDConstants initialized for " << name << " with " 
	      << layers(false) << ":" << layers(true) << " layers, " 
	      << sectors() << " sectors and maximum of " << maxCells(false)
	      << ":" << maxCells(true) << " cells" << std::endl;
#endif
  } else {
    rmax_     = k_ScaleFromDDD * (hgpar_->waferR_) * std::cos(30.0*CLHEP::deg);
    hexside_  = 2.0 * rmax_ * tan30deg_;
#ifdef EDM_ML_DEBUG
    std::cout << "rmax_ " << rmax_ << ":" << hexside_ << " CellSize " 
	      << 0.5*k_ScaleFromDDD*hgpar_->cellSize_[0] << ":" 
	      << 0.5*k_ScaleFromDDD*hgpar_->cellSize_[1] << std::endl;
#endif
    // init maps and constants
    modHalf_ = 0;
    for (int simreco = 0; simreco < 2; ++simreco) {
      tot_layers_[simreco] = layersInit((bool)simreco);
      max_modules_layer_[simreco].resize(tot_layers_[simreco]+1);
      for (unsigned int layer=1; layer <= tot_layers_[simreco]; ++layer) {
	max_modules_layer_[simreco][layer] = modulesInit(layer,(bool)simreco);
	if (simreco == 1) {
	  modHalf_ += max_modules_layer_[simreco][layer];
#ifdef EDM_ML_DEBUG
	  std::cout << "Layer " << layer << " with " << max_modules_layer_[simreco][layer] << ":" << modHalf_ << " modules\n";
#endif
	}
      }
    }
    tot_wafers_ = wafers();
    
    edm::LogInfo("HGCalGeom") << "HGCalDDDConstants initialized for " << name
			      << " with " << layers(false) << ":" 
			      << layers(true) << " layers, " << wafers() 
			      << ":" << 2*modHalf_ << " wafers and "
			      << "maximum of " << maxCells(false) << ":" 
			      << maxCells(true) << " cells";
    
#ifdef EDM_ML_DEBUG
    std::cout << "HGCalDDDConstants initialized for " << name << " with " 
	      << layers(false) << ":" << layers(true) << " layers, " 
	      << wafers() << " wafers and " << "maximum of " << maxCells(false)
	      << ":" << maxCells(true) << " cells" << std::endl;
#endif
    
  }  
  
  int wminT(9999999), wmaxT(0), kount1(0), kount2(0);
  for (unsigned int i=0; i<getTrFormN(); ++i) {
    int lay0 = getTrForm(i).lay;
    int wmin(9999999), wmax(0), kount(0);
    for (int wafer=0; wafer<sectors(); ++wafer) {
      if (waferInLayer(wafer,lay0,true)) {
	if (wafer < wmin) wmin = wafer;
	if (wafer > wmax) wmax = wafer;
	++kount;
      }
    }
    if (wminT  > wmin)  wminT = wmin;
    if (wmaxT  < wmax)  wmaxT = wmax;
    if (kount1 < kount) kount1= kount;
    kount2 += kount;
#ifdef EDM_ML_DEBUG
    int lay1 = getIndex(lay0,true).first;
    std::cout << "Index " << i << " Layer " << lay0 << ":" << lay1 
	      << " Wafer " << wmin << ":" << wmax << ":" << kount << std::endl;
#endif
    HGCWaferParam a1{ {wmin,wmax,kount} };
    waferLayer_[lay0] = a1;
  }
  waferMax_ = std::array<int,4>{ {wminT,wmaxT,kount1,kount2} };
#ifdef EDM_ML_DEBUG
  std::cout << "Overall wafer statistics: " << wminT << ":" << wmaxT << ":"
	    << kount1 << ":" << kount2 << std::endl;
#endif
}

HGCalDDDConstants::~HGCalDDDConstants() {}

std::pair<int,int> HGCalDDDConstants::assignCell(float x, float y, int lay,
						 int subSec, bool reco) const {
  std::pair<int,float> index = getIndex(lay, reco);
  std::pair<int,int> cellAssignment(-1,-1);
  int i = index.first;
  if (i < 0) return cellAssignment;
  if (mode_ == HGCalGeometryMode::Square) {
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
  int wafer = cellHex(xx, yy, rmax_, hgpar_->waferPosX_, hgpar_->waferPosY_);
  // Now the cell
  xx -= hgpar_->waferPosX_[wafer];
  yy -= hgpar_->waferPosY_[wafer];
  int cell(0);
  if (hgpar_->waferTypeT_[wafer] == 1) 
    cell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[0], hgpar_->cellFineX_, hgpar_->cellFineY_);
  else
    cell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[1], hgpar_->cellCoarseX_, hgpar_->cellCoarseY_);
  return std::pair<int,int>(wafer,cell);
}

double HGCalDDDConstants::cellSizeHex(int type) const {
  int    indx = (type == 1) ? 0 : 1;
  double cell = (0.5*k_ScaleFromDDD*hgpar_->cellSize_[indx]);
  return cell;
}

std::pair<int,int> HGCalDDDConstants::findCell(int cell, int lay, int subSec,
					       bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<int,int>(-1,-1);
  if (mode_ != HGCalGeometryMode::Square) {
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

HGCalParameters::hgtrap HGCalDDDConstants::getModule(unsigned int indx, 
						     bool hexType, 
						     bool reco) const {

  HGCalParameters::hgtrap mytr;
  if (hexType) {
    unsigned int type = (indx < hgpar_->waferTypeL_.size()) ? 
      hgpar_->waferTypeL_[indx] : 3;
    if (type > 0) --type;
    mytr = hgpar_->getModule(type, true);
  }  else {
    if (reco) mytr = hgpar_->getModule(indx,true);
    else      mytr = hgpar_->getModule(indx,false);
  }
  return mytr;
}

std::vector<HGCalParameters::hgtrap> HGCalDDDConstants::getModules() const {

  std::vector<HGCalParameters::hgtrap> mytrs;
  for (unsigned int k=0; k<hgpar_->moduleLayR_.size(); ++k) 
    mytrs.emplace_back(hgpar_->getModule(k,true));
  return mytrs;
}

std::vector<HGCalParameters::hgtrform> HGCalDDDConstants::getTrForms() const {

  std::vector<HGCalParameters::hgtrform> mytrs;
  for (unsigned int k=0; k<hgpar_->trformIndex_.size(); ++k) 
    mytrs.emplace_back(hgpar_->getTrForm(k));
  return mytrs;
}

bool HGCalDDDConstants::isValid(int lay, int mod, int cell, bool reco) const {

  bool ok(false);
  int  cellmax(0), modmax(0);
  if (mode_ == HGCalGeometryMode::Square) {
    cellmax = maxCells(lay,reco);
    modmax  = sectors();
    ok      = ((lay > 0 && lay <= (int)(layers(reco))) && 
	       (mod > 0 && mod <= modmax) &&
	       (cell >=0 && cell <= cellmax));
  } else {
    int32_t copyNumber = hgpar_->waferCopy_[mod];
    ok = ((lay > 0 && lay <= (int)(layers(reco))));
    if (ok) {
      const int32_t lay_idx = reco ? (lay-1)*3 + 1 : lay;
      const auto& the_modules = hgpar_->copiesInLayers_[lay_idx];
      auto moditr = the_modules.find(copyNumber);
      ok = (moditr != the_modules.end());
#ifdef EDM_ML_DEBUG
      if (!ok) std::cout << "HGCalDDDConstants: Layer " << lay << ":" 
			 << lay_idx << " Copy " << copyNumber << ":" << mod
			 << " Flag " << ok << std::endl;
#endif
      if (ok) {
	if (moditr->second >= 0) {
	  cellmax = (hgpar_->waferTypeT_[mod]==1) ? 
	    (int)(hgpar_->cellFineX_.size()) : (int)(hgpar_->cellCoarseX_.size());
	  ok = (cell >=0 && cell <=  cellmax);
	} else {
	  ok = isValidCell(lay_idx, mod, cell);
	}
      }
    }
  }
    
#ifdef EDM_ML_DEBUG
  if (!ok) std::cout << "HGCalDDDConstants: Layer " << lay << ":" 
		     << (lay > 0 && (lay <= (int)(layers(reco)))) << " Module "
		     << mod << ":" << modmax << ":" 
		     << (mod > 0 && mod <= modmax) << " Cell " << cell << ":"
		     << cellmax << ":" << (cell >=0 && cell <= cellmax)
		     << ":" << maxCells(reco) << std::endl; 
#endif
  return ok;
}

bool HGCalDDDConstants::isValidCell(int lay, int wafer, int cell) const {

  // Calculate the position of the cell
  double x = hgpar_->waferPosX_[wafer];
  double y = hgpar_->waferPosY_[wafer];
  if (hgpar_->waferTypeT_[wafer] == 1) {
    x     += hgpar_->cellFineX_[cell];
    y     += hgpar_->cellFineY_[cell];
  } else {
    x     += hgpar_->cellCoarseX_[cell];
    y     += hgpar_->cellCoarseY_[cell];
  }
  double rr = sqrt(x*x+y*y);
  bool   ok = ((rr >= hgpar_->rMinLayHex_[lay-1]) &
	       (rr <= hgpar_->rMaxLayHex_[lay-1]));
#ifdef EDM_ML_DEBUG
  if (!ok) 
    std::cout << "Input " << lay << ":" << wafer << ":" << cell << " Position "
	      << x << ":" << y << ":" << rr << " Compare Limits " 
	      << hgpar_->rMinLayHex_[lay-1] << ":" <<hgpar_->rMaxLayHex_[lay-1]
	      << " Flag " << ok << std::endl;
#endif
  return ok;
}

unsigned int HGCalDDDConstants::layers(bool reco) const {
  return tot_layers_[(int)reco];
}

unsigned int HGCalDDDConstants::layersInit(bool reco) const {
  return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());
}

std::pair<float,float> HGCalDDDConstants::locateCell(int cell, int lay, 
						     int type, bool reco) const {
  // type refers to subsector # for square cell and wafer # for hexagon cell
  float x(999999.), y(999999.);
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<float,float>(x,y);
  if (mode_ == HGCalGeometryMode::Square) {
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
    x              = hgpar_->waferPosX_[type];
    y              = hgpar_->waferPosY_[type];
    if (hgpar_->waferTypeT_[type] == 1) {
      x           += hgpar_->cellFineX_[cell];
      y           += hgpar_->cellFineY_[cell];
    } else {
      x           += hgpar_->cellCoarseX_[cell];
      y           += hgpar_->cellCoarseY_[cell];
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
    x  = hgpar_->cellFineX_[cell];
    y  = hgpar_->cellFineY_[cell];
  } else {
    x  = hgpar_->cellCoarseX_[cell];
    y  = hgpar_->cellCoarseY_[cell];
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
  if (mode_ == HGCalGeometryMode::Square) {
    float h, bl, tl, alpha;
    getParameterSquare(i,0,reco,h,bl,tl,alpha);
    return maxCellsSquare(h, bl, tl, alpha, index.second);
  } else {
    unsigned int cells(0);
    for (unsigned int k=0; k<hgpar_->waferTypeT_.size(); ++k) {
      if (waferInLayer(k,index.first)) {
	unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	  (hgpar_->cellFineX_.size()) : (hgpar_->cellCoarseX_.size());
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
  if (mode_ == HGCalGeometryMode::Square) {
    float h = (reco) ? hgpar_->moduleHR_[i] : hgpar_->moduleHS_[i];
    kymax   = floor((2*h)/index.second);
  } else {
    for (unsigned int k=0; k<hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayer(k,i)) {
	int ky = ((hgpar_->waferCopy_[k])/100)%100;
	if (ky > kymax) kymax = ky;
      }
    }
  }
  return kymax;
}

int HGCalDDDConstants::modules(int lay, bool reco) const {
  if( getIndex(lay,reco).first < 0 ) return 0;
  return max_modules_layer_[(int)reco][lay];
}

int HGCalDDDConstants::modulesInit(int lay, bool reco) const {
  int nmod(0);
  std::pair<int,float> index = getIndex(lay, reco);
  if (index.first < 0) return nmod;
  for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
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
  float alpha    = (subSec == 0) ? hgpar_->moduleAlphaS_[i] : subSec;
  float cellSize = index.second;
  float a        = (alpha==0) ? 
    (2*hgpar_->moduleHR_[i]/(hgpar_->moduleTlR_[i]-hgpar_->moduleBlR_[i])) :
    (hgpar_->moduleHR_[i]/(hgpar_->moduleTlR_[i]-hgpar_->moduleBlR_[i]));
  float b        = 2*hgpar_->moduleHR_[i]*hgpar_->moduleBlR_[i]/
    (hgpar_->moduleTlR_[i]-hgpar_->moduleBlR_[i]);
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
    if (mode_ == HGCalGeometryMode::Square) {
      float h, bl, tl, alpha;
      getParameterSquare(i,0,reco,h,bl,tl,alpha);
      return numberCellsSquare(h, bl, tl, alpha, index.second);
    } else {
      for (unsigned int k=0; k<hgpar_->waferTypeT_.size(); ++k) {
	if (waferInLayer(k,i)) {
	  unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	    (hgpar_->cellFineX_.size()) : (hgpar_->cellCoarseX_.size());
	  ncell.emplace_back((int)(cell));
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
    ncell.emplace_back(floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize)));
  return ncell;
}

int HGCalDDDConstants::numberCellsHexagon(int wafer) const {

  int ncell(0);
  if (wafer >= 0 && wafer < (int)(hgpar_->waferTypeT_.size())) {
    if (hgpar_->waferTypeT_[wafer]==1) 
      ncell = (int)(hgpar_->cellFineX_.size());
    else 
      ncell = (int)(hgpar_->cellCoarseX_.size());
  }
  return ncell;
}

std::pair<int,int> HGCalDDDConstants::rowColumnWafer(int wafer) const {
  int row(0), col(0);
  if (wafer < (int)(hgpar_->waferCopy_.size())) {
    int copy = hgpar_->waferCopy_[wafer];
    col      = copy%100;
    if ((copy/10000)%10 != 0)  col = -col;
    row      = (copy/100)%100;
    if ((copy/100000)%10 != 0) row = -row;
  }
  return std::pair<int,int>(row,col);
}

std::pair<int,int> HGCalDDDConstants::simToReco(int cell, int lay, int mod,
						bool half) const {
  
  std::pair<int,float> index = getIndex(lay, false);
  int i = index.first;
  if (i < 0) {
    return std::pair<int,int>(-1,-1);
  }
  int kx(-1), depth(-1);
  if (mode_ == HGCalGeometryMode::Square) {
    float h  = hgpar_->moduleHS_[i];
    float bl = hgpar_->moduleBlS_[i];
    float tl = hgpar_->moduleTlS_[i];
    float cellSize = hgpar_->cellFactor_[i]*index.second;
  
    std::pair<int,int> kxy = findCellSquare(cell, h, bl, tl, hgpar_->moduleAlphaS_[i], index.second);
    depth       = hgpar_->layerGroup_[i];
    if (depth<0) return std::pair<int,int>(-1,-1);
    kx          = kxy.first/hgpar_->cellFactor_[i];
    int ky      = kxy.second/hgpar_->cellFactor_[i];

    float a     = (half) ? (h/(tl-bl)) : (2*h/(tl-bl));
    float b     = 2*h*bl/(tl-bl);
    for (int iky=0; iky<ky; ++iky)
      kx += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
#ifdef EDM_ML_DEBUG
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
  const int ncopies = hgpar_->waferCopy_.size();
  int  wafer(ncopies);
#ifdef EDM_ML_DEBUG
  bool ok(false);
#endif
  for (int k=0; k<ncopies; ++k) {
    if (copy == hgpar_->waferCopy_[k]) {
      wafer = k;
#ifdef EDM_ML_DEBUG
      ok = true;
#endif
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  if (!ok) {
    std::cout << "Cannot find " << copy << " in a list of " << ncopies << "\n";
    for (int k=0; k<ncopies; ++k) std::cout << " " << hgpar_->waferCopy_[k];
    std::cout << std::endl;
  }
#endif
  return wafer;
}

void HGCalDDDConstants::waferFromPosition(const double x, const double y,
					  int& wafer, int& icell, 
					  int& celltyp) const {

  double xx(k_ScaleFromDDD*x), yy(k_ScaleFromDDD*y);
  int size_ = (int)(hgpar_->waferCopy_.size());
  wafer     = size_;
  for (int k=0; k<size_; ++k) {
    double dx = std::abs(xx-hgpar_->waferPosX_[k]);
    double dy = std::abs(yy-hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5*hexside_) || (dx <= (2.*rmax_-dy/tan30deg_))) {
	wafer   = k;
	celltyp = hgpar_->waferTypeT_[k];
	xx     -= hgpar_->waferPosX_[k];
	yy     -= hgpar_->waferPosY_[k];
	break;
      }
    }
  }
  if (wafer < size_) {
    if (celltyp == 1) 
      icell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[0], 
		       hgpar_->cellFineX_, hgpar_->cellFineY_);
    else
      icell  = cellHex(xx, yy, 0.5*k_ScaleFromDDD*hgpar_->cellSize_[1],
		       hgpar_->cellCoarseX_, hgpar_->cellCoarseY_);
  }
#ifdef EDM_ML_DEBUG
  std::cout << "Position " << x << ":" << y << " Wafer " << wafer << ":" 
	    << size_ << " XX " << xx << ":" << yy << " Cell " << icell 
	    << " Type " << celltyp << std::endl;
#endif
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay, bool reco) const {

  std::pair<int,float> indx = getIndex(lay, reco);
  if (indx.first < 0) return false;
  return waferInLayer(wafer,indx.first);
}

std::pair<double,double> HGCalDDDConstants::waferPosition(int wafer, 
							  bool reco) const {

  double xx(0), yy(0);
  if (wafer >= 0 && wafer < (int)(hgpar_->waferPosX_.size())) {
    xx = hgpar_->waferPosX_[wafer];
    yy = hgpar_->waferPosY_[wafer];
  }
  if (!reco) {
    xx /= k_ScaleFromDDD;
    yy /= k_ScaleFromDDD;
  }
  std::pair<double,double> xy(xx,yy);
  return xy;
}

double HGCalDDDConstants::waferZ(int lay, bool reco) const {
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  else       return hgpar_->zLayerHex_[i];
}

int HGCalDDDConstants::wafers() const {

  int wafer(0);
  for (unsigned int i = 0; i<layers(true); ++i) {
    int lay = hgpar_->depth_[i];
    wafer += modules(lay, true);
  }
  return wafer;
}

int HGCalDDDConstants::wafers(int layer, int type) const {

  int  wafer(0);
  auto itr = waferLayer_.find(layer);
  if (itr != waferLayer_.end()) {
    unsigned ity = (type > 0 && type <= 2) ? type : 0;
    wafer = (itr->second)[ity];
  }
  return wafer;
}

bool HGCalDDDConstants::isHalfCell(int waferType, int cell) const {
  if( waferType < 1 || cell < 0) return false;
  return waferType == 2 ? hgpar_->cellCoarseHalf_[cell] : hgpar_->cellFineHalf_[cell];
}

int HGCalDDDConstants::cellHex(double xx, double yy, 
                               const double& cellR, 
			       const std::vector<double>& posX,
			       const std::vector<double>& posY) const {
  int num(0);
  const double tol(0.00001);
  double cellY = 2.0*cellR*tan30deg_;
  for (unsigned int k=0; k<posX.size(); ++k) {
    double dx = std::abs(xx - posX[k]);
    double dy = std::abs(yy - posY[k]);
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
  if (mode_ == HGCalGeometryMode::Square) {
    indx = (reco ? hgpar_->depthIndex_[lay-1] : hgpar_->layerIndex_[lay-1]);
    cell = (reco ? hgpar_->moduleCellR_[indx] : hgpar_->moduleCellS_[indx]);
  } else {
    indx = (reco ? hgpar_->depthLayerF_[lay-1] : hgpar_->layerIndex_[lay-1]);
    cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
  }
  return std::pair<int,float>(indx,cell);
}

void HGCalDDDConstants::getParameterSquare(int lay, int subSec, bool reco, 
					   float& h, float& bl, float& tl,
					   float& alpha) const {
  if (reco) {
    h    =  hgpar_->moduleHR_[lay];
    bl   =  hgpar_->moduleBlR_[lay];
    tl   =  hgpar_->moduleTlR_[lay];
    alpha=  hgpar_->moduleAlphaR_[lay];
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  hgpar_->moduleHS_[lay];
    bl   =  hgpar_->moduleBlS_[lay];
    tl   =  hgpar_->moduleTlS_[lay];
    alpha=  hgpar_->moduleAlphaS_[lay];
  }
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay) const {

  const double rr     = 2*rmax_*tan30deg_;
  const double waferX = hgpar_->waferPosX_[wafer];
  const double waferY = hgpar_->waferPosY_[wafer];
  double       xc[6], yc[6];
  xc[0] = waferX+rmax_; yc[0] = waferY-0.5*rr;
  xc[1] = waferX+rmax_; yc[1] = waferY+0.5*rr;
  xc[2] = waferX;       yc[2] = waferY+rr;
  xc[3] = waferX-rmax_; yc[3] = waferY+0.5*rr;
  xc[4] = waferX+rmax_; yc[4] = waferY-0.5*rr;
  xc[5] = waferX;       yc[5] = waferY-rr;
  bool cornerOne(false), cornerAll(true);
  for (int k=0; k<6; ++k) {
    double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    if ((rpos >= hgpar_->rMinLayHex_[lay]) && 
	(rpos <= hgpar_->rMaxLayHex_[lay])) cornerOne = true;
    else                                    cornerAll = false;
  }
  bool   in(false);
  if (hgpar_->mode_ == HGCalGeometryMode::Hexagon)
    in = cornerAll;
  else if (hgpar_->mode_ == HGCalGeometryMode::HexagonFull)
    in = cornerOne;
  return in;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
