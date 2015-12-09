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

const double k_horizontalShift = 1.0;

HGCalDDDConstants::HGCalDDDConstants(const HGCalParameters* hp,
				     const std::string name) : hgpar_(hp) {

  edm::LogInfo("HGCalGeom") << "HGCalDDDConstants initialized for " << name
			    << " with " << layers(false) << ":" << layers(true)
			    << " layers, " << sectors() << " sectors and "
			    << "maximum of " << maxCells(false) << ":" 
			    << maxCells(true) << " cells";
#ifdef DebugLog
    std::cout << "HGCalDDDConstants initialized for " << name << " with " 
	      << layers(false) << ":" << layers(true) << " layers, " 
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
    h    =  hgpar_->moduler_[i].h;
    bl   =  hgpar_->moduler_[i].bl;
    tl   =  hgpar_->moduler_[i].tl;
    alpha=  hgpar_->moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
    cellAssignment=assignCell(x, y, h, bl, tl, alpha, index.second);
  } else {
    h    =  hgpar_->modules_[i].h;
    bl   =  hgpar_->modules_[i].bl;
    tl   =  hgpar_->modules_[i].tl;
    alpha=  hgpar_->modules_[i].alpha;
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
    h    =  hgpar_->moduler_[i].h;
    bl   =  hgpar_->moduler_[i].bl;
    tl   =  hgpar_->moduler_[i].tl;
    alpha=  hgpar_->moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  hgpar_->modules_[i].h;
    bl   =  hgpar_->modules_[i].bl;
    tl   =  hgpar_->modules_[i].tl;
    alpha=  hgpar_->modules_[i].alpha;
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
    int cellsInRow(floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize)));
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
    h    =  hgpar_->moduler_[i].h;
    bl   =  hgpar_->moduler_[i].bl;
    tl   =  hgpar_->moduler_[i].tl;
    alpha=  hgpar_->moduler_[i].alpha;
    if ((subSec>0 && alpha<0) || (subSec<=0 && alpha>0)) alpha = -alpha;
  } else {
    h    =  hgpar_->modules_[i].h;
    bl   =  hgpar_->modules_[i].bl;
    tl   =  hgpar_->modules_[i].tl;
    alpha=  hgpar_->modules_[i].alpha;
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
    int lay = reco ? hgpar_->depth_[i] : hgpar_->layer_[i];
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
    h    =  hgpar_->moduler_[i].h;
    bl   =  hgpar_->moduler_[i].bl;
    tl   =  hgpar_->moduler_[i].tl;
    alpha=  hgpar_->moduler_[i].alpha;
  } else {
    h    =  hgpar_->modules_[i].h;
    bl   =  hgpar_->modules_[i].bl;
    tl   =  hgpar_->modules_[i].tl;
    alpha=  hgpar_->modules_[i].alpha;
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
  for (int iky=0; iky<kymax; ++iky) {
    int cellsInRow=floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
    ncells += cellsInRow;
  }

  return ncells;
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  float h     = (reco) ? hgpar_->moduler_[i].h : hgpar_->modules_[i].h;
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
  if (i >= 0) {
    float h, bl, tl, alpha;
    if (reco) {
      h    =  hgpar_->moduler_[i].h;
      bl   =  hgpar_->moduler_[i].bl;
      tl   =  hgpar_->moduler_[i].tl;
      alpha=  hgpar_->moduler_[i].alpha;
    } else {
      h    =  hgpar_->modules_[i].h;
      bl   =  hgpar_->modules_[i].bl;
      tl   =  hgpar_->modules_[i].tl;
      alpha=  hgpar_->modules_[i].alpha;
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
  float h  = hgpar_->modules_[i].h;
  float bl = hgpar_->modules_[i].bl;
  float tl = hgpar_->modules_[i].tl;
  float cellSize = hgpar_->cellFactor_[i]*index.second;
  
  std::pair<int,int> kxy = findCell(cell, h, bl, tl, hgpar_->modules_[i].alpha, index.second);
  int depth   = hgpar_->layerGroup_[i];
  if(depth<0) return std::pair<int,int>(-1,-1);
  int kx      = kxy.first/hgpar_->cellFactor_[i];
  int ky      = kxy.second/hgpar_->cellFactor_[i];

  float a     = (half) ? (h/(tl-bl)) : (2*h/(tl-bl));
  float b     = 2*h*bl/(tl-bl);
  for (int iky=0; iky<ky; ++iky)
    kx += floor(((iky+1)*cellSize+b+k_horizontalShift*cellSize)/(a*cellSize));
  
#ifdef DebugLog
  std::cout << "simToReco: input " << cell << ":" << lay << ":" << half
	    << " kxy " << kxy.first << ":" << kxy.second << " output "
    	    << kx << ":" << depth << " cell factor=" << hgpar_->cellFactor_[i] 
	    << std::endl;
#endif
  return std::pair<int,int>(kx,depth);
}

std::pair<int,float> HGCalDDDConstants::getIndex(int lay, bool reco) const {

  if (lay<1 || lay>(int)(hgpar_->layerIndex_.size())) return std::pair<int,float>(-1,0);
  if (reco && lay>(int)(hgpar_->depthIndex_.size()))  return std::pair<int,float>(-1,0);
  int   i    = (reco ? hgpar_->depthIndex_[lay-1] : hgpar_->layerIndex_[lay-1]);
  float cell = (reco ? hgpar_->moduler_[i].cellSize : hgpar_->modules_[i].cellSize);
  return std::pair<int,float>(i,cell);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
