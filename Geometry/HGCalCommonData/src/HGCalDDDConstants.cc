#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

//#define EDM_ML_DEBUG

HGCalDDDConstants::HGCalDDDConstants(const HGCalParameters* hp,
				     const std::string& name) : hgpar_(hp),
								sqrt3_(std::sqrt(3.0)) {
  mode_ = hgpar_->mode_;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull) ||
      (mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    rmax_     = (HGCalParameters::k_ScaleFromDDD * (hgpar_->waferR_) *
		 std::cos(30.0*CLHEP::deg));
    hexside_  = 2.0 * rmax_ * tan30deg_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "rmax_ " << rmax_ << ":" << hexside_ 
				  << " CellSize " << 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[0] 
				  << ":" << 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[1];
#endif
  }
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
	edm::LogVerbatim("HGCalGeom") << "Layer " << layer << " with " 
					<< max_modules_layer_[simreco][layer] 
					<< ":" << modHalf_ << " modules";
#endif
      }
    }
  }
  tot_wafers_ = wafers();
    
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants initialized for " 
				<< name << " with " << layers(false) << ":" 
				<< layers(true) << " layers, " << wafers() 
				<< ":" << 2*modHalf_ << " wafers and "
				<< "maximum of " << maxCells(false) << ":" 
				<< maxCells(true) << " cells";

  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull) ||
      (mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int wminT(9999999), wmaxT(-9999999), kount1(0), kount2(0);
    for (unsigned int i=0; i<getTrFormN(); ++i) {
      int lay0 = getTrForm(i).lay;
      int wmin(9999999), wmax(-9999999), kount(0);
      for (int wafer=0; wafer<sectors(); ++wafer) {
	if (waferInLayer(wafer,lay0,true)) {
	  int waferU = (((mode_ == HGCalGeometryMode::Hexagon) ||
			 (mode_ == HGCalGeometryMode::HexagonFull)) ? wafer :
			HGCalWaferIndex::waferU(hgpar_->waferCopy_[wafer]));
	  if (waferU < wmin) wmin = waferU;
	  if (waferU > wmax) wmax = waferU;
	  ++kount;
	}
      }
      if (wminT  > wmin)  wminT = wmin;
      if (wmaxT  < wmax)  wmaxT = wmax;
      if (kount1 < kount) kount1= kount;
      kount2 += kount;
#ifdef EDM_ML_DEBUG
      int lay1 = getIndex(lay0,true).first;
      edm::LogVerbatim("HGCalGeom") << "Index " << i << " Layer " << lay0
				    << ":"  << lay1 << " Wafer " << wmin 
				    << ":" << wmax << ":" << kount;
#endif
      HGCWaferParam a1{ {wmin,wmax,kount} };
      waferLayer_[lay0] = a1;
    }
    waferMax_ = std::array<int,4>{ {wminT,wmaxT,kount1,kount2} };
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Overall wafer statistics: " << wminT 
				  << ":" << wmaxT << ":" << kount1 << ":" 
				  << kount2;
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
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    float xx = (reco) ? x : HGCalParameters::k_ScaleFromDDD*x;
    float yy = (reco) ? y : HGCalParameters::k_ScaleFromDDD*y;

    //First the wafer
    int wafer = cellHex(xx, yy, rmax_, hgpar_->waferPosX_, hgpar_->waferPosY_);
    if (wafer < 0 || wafer >= (int)(hgpar_->waferTypeT_.size())) {
      edm::LogWarning("HGCalGeom") << "Wafer no. out of bound for " << wafer 
				   << ":" << (hgpar_->waferTypeT_).size() 
				   << ":" << (hgpar_->waferPosX_).size()
				   << ":" << (hgpar_->waferPosY_).size() 
				   << " ***** ERROR *****";
    } else {
      // Now the cell
      xx -= hgpar_->waferPosX_[wafer];
      yy -= hgpar_->waferPosY_[wafer];
      int cell(0);
      if (hgpar_->waferTypeT_[wafer] == 1) 
	cell  = cellHex(xx, yy, 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[0], 
			hgpar_->cellFineX_, hgpar_->cellFineY_);
      else
	cell  = cellHex(xx, yy, 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[1], 
			hgpar_->cellCoarseX_, hgpar_->cellCoarseY_);
      cellAssignment = std::pair<int,int>(wafer,cell);
    }
  }
  return cellAssignment;
}

std::array<int,5> HGCalDDDConstants::assignCellHex(float x, float y, int lay,
						   bool reco) const {
  int waferU(0), waferV(0), waferType(-1), cellU(0), cellV(0);
  if ((mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    double xx = (reco) ? HGCalParameters::k_ScaleToDDD*x : x;
    double yy = (reco) ? HGCalParameters::k_ScaleToDDD*y : y;
    double wt(1.0);
    waferFromPosition(xx,yy,lay,waferU,waferV,cellU,cellV,waferType,wt);
  }
  return std::array<int,5>{ {waferU,waferV,waferType,cellU,cellV} };
}

std::array<int,3> HGCalDDDConstants::assignCellTrap(float x, float y,
						    float z, int layer,
						    bool reco) const {
  
  int ieta(-1), iphi(-1), type(-1);
  std::pair<int,float> indx  = getIndex(layer,reco);
  if (indx.first < 0) return std::array<int,3>{ {ieta,iphi,type} };
  double xx    = (z > 0) ? x : -x;
  double zz    = (reco ? hgpar_->zLayerHex_[indx.first] : 
		  HGCalParameters::k_ScaleToDDD*hgpar_->zLayerHex_[indx.first]);
  double r     = std::sqrt(x*x+y*y+zz*zz);
  double theta = (r == 0. ? 0. : std::acos(std::max(std::min(zz/r,1.0),-1.0)));
  double stheta= std::sin(theta);
  double phi   = (r*stheta == 0. ? 0. : std::atan2(y,xx));
  if (phi < 0) phi += (2.0*M_PI);
  double eta   = (std::abs(stheta) == 1.? 0. : -std::log(std::abs(std::tan(0.5*theta))) );
  ieta         = 1 + (int)((std::abs(eta)-hgpar_->etaMinBH_)/indx.second);
  iphi         = 1 + (int)(phi/indx.second);
  type         = scintType(indx.second);
  return std::array<int,3>{ {ieta,iphi,type} };
}

double HGCalDDDConstants::cellSizeHex(int type) const {
  int    indx = (type == 1) ? 0 : 1;
  double cell = (0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[indx]);
  return cell;
}

HGCalParameters::hgtrap HGCalDDDConstants::getModule(unsigned int indx, 
						     bool hexType, 
						     bool reco) const {

  HGCalParameters::hgtrap mytr;
  if (hexType) {
    if (indx >= hgpar_->waferTypeL_.size())
      edm::LogWarning("HGCalGeom") << "Wafer no. out bound for index " << indx
				   << ":" << (hgpar_->waferTypeL_).size() 
				   << ":" << (hgpar_->waferPosX_).size() 
				   << ":" << (hgpar_->waferPosY_).size() 
				   << " ***** ERROR *****";
    unsigned int type = (indx < hgpar_->waferTypeL_.size()) ? 
      hgpar_->waferTypeL_[indx] : 3;
    if (type > 0) --type;
    mytr = hgpar_->getModule(type, true);
  }  else {
    mytr = hgpar_->getModule(indx,reco);
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

  bool ok(false), okMod(false);
  int  cellmax(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    int32_t copyNumber = hgpar_->waferCopy_[mod];
    ok = ((lay > 0 && lay <= (int)(layers(reco))));
    if (ok) {
      const int32_t lay_idx = reco ? (lay-1)*3 + 1 : lay;
      const auto& the_modules = hgpar_->copiesInLayers_[lay_idx];
      auto moditr = the_modules.find(copyNumber);
      ok = okMod = (moditr != the_modules.end());
#ifdef EDM_ML_DEBUG
      if (!ok) edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay 
					     << ":" << lay_idx << " Copy " 
					     << copyNumber << ":" << mod
					     << " Flag " << ok;
#endif
      if (ok) {
	if (moditr->second >= 0) {
	  if (mod >= (int)(hgpar_->waferTypeT_.size()))
	    edm::LogWarning("HGCalGeom") << "Module no. out of bound for " 
					 << mod << " to be compared with " 
					 << (hgpar_->waferTypeT_).size() 
					 << " ***** ERROR *****";
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
  if (!ok) 
    edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay << ":"
				  << (lay > 0 && (lay <= (int)(layers(reco))))
				  << " Module " << mod << ":" << okMod 
				  << " Cell " << cell << ":" << cellmax << ":"
				  << (cell >=0 && cell <= cellmax)
				  << ":" << maxCells(reco); 
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
  bool   ok = ((rr >= hgpar_->rMinLayHex_[lay-1]) &&
	       (rr <= hgpar_->rMaxLayHex_[lay-1]) &&
	       (wafer < (int)(hgpar_->waferPosX_.size())));
#ifdef EDM_ML_DEBUG
  if (!ok) 
    edm::LogVerbatim("HGCalGeom") << "Input " << lay << ":" << wafer << ":" 
				  << cell << " Position " << x << ":" << y 
				  << ":" << rr << " Compare Limits "
				  << hgpar_->rMinLayHex_[lay-1] << ":" 
				  << hgpar_->rMaxLayHex_[lay-1]
				  << " Flag " << ok;
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
  // type refers to wafer # for hexagon cell
  float x(999999.), y(999999.);
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::pair<float,float>(x,y);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
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
      x           *= HGCalParameters::k_ScaleToDDD;
      y           *= HGCalParameters::k_ScaleToDDD;
    }
  }
  return std::pair<float,float>(x,y);
}

std::pair<float,float> HGCalDDDConstants::locateCell(int lay, int waferU,
						     int waferV, int cellU,
						     int cellV, bool reco) const {

  float x(0), y(0);
  std::pair<double,double> xy = waferPosition(waferU, waferV, reco);
  int  indx = HGCalWaferIndex::waferIndex(lay,waferU,waferV);
  auto itr  = hgpar_->typesInLayers_.find(indx);
  int  type = (itr == hgpar_->typesInLayers_.end()) ? 2 : hgpar_->waferTypeL_[itr->second];
  int  kndx = cellV*100 + cellU;
  if (type == 0) {
    auto ktr = hgpar_->cellFineIndex_.find(kndx);
    if (ktr != hgpar_->cellFineIndex_.end()) {
      x = hgpar_->cellFineX_[ktr->second];
      y = hgpar_->cellFineY_[ktr->second];
    }
  } else {
    auto ktr = hgpar_->cellCoarseIndex_.find(kndx);
    if (ktr != hgpar_->cellCoarseIndex_.end()) {
      x = hgpar_->cellCoarseX_[ktr->second];
      y = hgpar_->cellCoarseY_[ktr->second];
    }
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  x += xy.first;
  y += xy.second;
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
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  return std::pair<float,float>(x,y);
}

std::pair<float,float> HGCalDDDConstants::locateCellTrap(int lay, int ieta,
							 int iphi, bool reco) const {
  
  float x(0), y(0);
  std::pair<int,float> indx = getIndex(lay,reco);
  if (indx.first >= 0) {
    ieta        = std::abs(ieta);
    double eta  = hgpar_->etaMinBH_ + (ieta-0.5)*indx.second;
    double phi  = (iphi-0.5)*indx.second;
    double z    = hgpar_->zLayerHex_[indx.first];
    double r    = z*std::tan(2.0*std::atan(std::exp(-eta)));
    x           = r*std::cos(phi);
    y           = r*std::sin(phi);
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
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
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    unsigned int cells(0);
    for (unsigned int k=0; k<hgpar_->waferTypeT_.size(); ++k) {
      if (waferInLayer(k,index.first)) {
	unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	  (hgpar_->cellFineX_.size()) : (hgpar_->cellCoarseX_.size());
	if (cell > cells) cells = cell;
      }
    }
    return (int)(cells);
  } else {
    return 0;
  }
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  int kymax(0);
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return kymax;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
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
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
      if (waferInLayer(k,index.first)) ++nmod;
    }
  } else {
    nmod = 1+hgpar_->lastModule_[index.first]-hgpar_->firstModule_[index.first];
  }
  return nmod;
}

double HGCalDDDConstants::mouseBite(bool reco) const {

  return (reco ? hgpar_->mouseBite_ : HGCalParameters::k_ScaleToDDD*hgpar_->mouseBite_);
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {

  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  std::vector<int> ncell;
  if (i >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
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
    edm::LogWarning("HGCalGeom") << "Wrong Layer # " << lay 
				 << " not in the list ***** ERROR *****";
    return std::pair<int,int>(-1,-1);
  }
  if (mod >= (int)(hgpar_->waferTypeL_).size()) {
    edm::LogWarning("HGCalGeom") << "Invalid Wafer # " << mod << "should be < "
				 << (hgpar_->waferTypeL_).size()
				 << " ***** ERROR *****";
    return std::pair<int,int>(-1,-1);
  }
  int kx(-1), depth(-1);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
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
  bool ok(false);
  for (int k=0; k<ncopies; ++k) {
    if (copy == hgpar_->waferCopy_[k]) {
      wafer = k;
      ok = true;
      break;
    }
  }
  if (!ok) {
    wafer = -1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Cannot find " << copy << " in a list of "
				  << ncopies << " members";
    for (int k=0; k<ncopies; ++k) 
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << hgpar_->waferCopy_[k];
#endif
  }
  return wafer;
}

void HGCalDDDConstants::waferFromPosition(const double x, const double y,
					  int& wafer, int& icell, 
					  int& celltyp) const {
  //Input x, y in Geant4 unit and transformed to conform +z convention
  double xx(HGCalParameters::k_ScaleFromDDD*x), yy(HGCalParameters::k_ScaleFromDDD*y);
  int size_ = (int)(hgpar_->waferCopy_.size());
  wafer     = size_;
  for (int k=0; k<size_; ++k) {
    double dx = std::abs(xx-hgpar_->waferPosX_[k]);
    double dy = std::abs(yy-hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5*hexside_) || (dx*tan30deg_ <= (hexside_-dy))) {
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
      icell  = cellHex(xx, yy, 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[0], 
		       hgpar_->cellFineX_, hgpar_->cellFineY_);
    else
      icell  = cellHex(xx, yy, 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[1],
		       hgpar_->cellCoarseX_, hgpar_->cellCoarseY_);
  } else {
    wafer = -1;
#ifdef EDM_ML_DEBUG
    edm::LogWarning("HGCalGeom") << "Cannot get wafer type corresponding to " 
				 << x << ":" << y << "    " << xx << ":" << yy;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Position " << x << ":" << y << " Wafer "
				<< wafer << ":" << size_ << " XX " << xx << ":"
				<< yy << " Cell " << icell << " Type " << celltyp;
#endif
}

void HGCalDDDConstants::waferFromPosition(const double x, const double y,
					  const int layer, int& waferU,
					  int& waferV, int& cellU, int& cellV,
					  int& celltype, double& wt) const {

  double xx(HGCalParameters::k_ScaleFromDDD*x), yy(HGCalParameters::k_ScaleFromDDD*y);
  waferU = waferV = 1+hgpar_->waferUVMax_;
  for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
    double dx = std::abs(xx-hgpar_->waferPosX_[k]);
    double dy = std::abs(yy-hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5*hexside_) || (dx*tan30deg_ <= (hexside_-dy))) {
	int indx = hgpar_->waferCopy_[k];
	waferU   = HGCalWaferIndex::waferU(indx);
	waferV   = HGCalWaferIndex::waferV(indx);
	indx     = HGCalWaferIndex::waferIndex(layer,waferU,waferV);
	auto itr = hgpar_->typesInLayers_.find(indx);
	celltype = (itr == hgpar_->typesInLayers_.end()) ? 2 : hgpar_->waferTypeL_[itr->second];
	xx      -= hgpar_->waferPosX_[k];
	yy      -= hgpar_->waferPosY_[k];
	break;
      }
    }
  }
  if (std::abs(waferU) <= hgpar_->waferUVMax_) {
    cellHex(xx, yy, celltype, cellU, cellV);
    wt    = ((celltype < 2) ? 
	     (hgpar_->cellThickness_[celltype]/hgpar_->waferThick_) : 1.0);
  } else {
    cellU = cellV = 2*hgpar_->nCellsFine_;
    wt    = 1.0;
  }
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
    xx *= HGCalParameters::k_ScaleToDDD;
    yy *= HGCalParameters::k_ScaleToDDD;
  }
  std::pair<double,double> xy(xx,yy);
  return xy;
}

std::pair<double,double> HGCalDDDConstants::waferPosition(int waferU,
							  int waferV,
							  bool reco) const {

  double xx(0), yy(0);
  int indx = HGCalWaferIndex::waferIndex(0,waferU,waferV);
  auto itr = hgpar_->wafersInLayers_.find(indx);
  if (itr != hgpar_->wafersInLayers_.end()) {
    xx  = hgpar_->waferPosX_[itr->second];
    yy  = hgpar_->waferPosY_[itr->second];
  }
  if (!reco) {
    xx *= HGCalParameters::k_ScaleToDDD;
    yy *= HGCalParameters::k_ScaleToDDD;
  }
  std::pair<double,double> xy(xx,yy);
  return xy;
}

double HGCalDDDConstants::waferZ(int lay, bool reco) const {
  std::pair<int,float> index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return 0;
  else       return (reco ? hgpar_->zLayerHex_[i] : 
		     HGCalParameters::k_ScaleToDDD*hgpar_->zLayerHex_[i]);
}

int HGCalDDDConstants::wafers() const {

  int wafer(0);
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    for (unsigned int i = 0; i<layers(true); ++i) {
      int lay = hgpar_->depth_[i];
      wafer += modules(lay, true);
    }
  } else {
    wafer = (int)(hgpar_->moduleLayR_.size());
  }
  return wafer;
}

int HGCalDDDConstants::wafers(int layer, int type) const {

  int  wafer(0);
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    auto itr = waferLayer_.find(layer);
    if (itr != waferLayer_.end()) {
      unsigned ity = (type > 0 && type <= 2) ? type : 0;
      wafer = (itr->second)[ity];
    }
  } else {
    std::pair<int,float> index = getIndex(layer, true);
    wafer = 1+hgpar_->lastModule_[index.first]-hgpar_->firstModule_[index.first];
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

void HGCalDDDConstants::cellHex(double xloc, double yloc, int cellType, 
				int& cellU, int& cellV) const {
  int N     = (cellType == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  double Rc = 2*rmax_/(3*N);
  double rc = 0.5*Rc*sqrt3_;
  double v0 = ((xloc/Rc -1.0)/1.5);
  int cv0   = (v0 > 0) ? (N + (int)(v0+0.5)) : (N - (int)(-v0+0.5));
  double u0 = (0.5*yloc/rc+0.5*cv0);
  int cu0   = (u0 > 0) ? (N/2 + (int)(u0+0.5)) : (N/2 - (int)(-u0+0.5));
  bool found(false);
  static const int shift[3] = {0,1,-1};
  for (int i1=0; i1<3; ++i1) {
    cellU       = cu0 + shift[i1];
    for (int i2=0; i2<3; ++i2) {
      cellV     = cv0 + shift[i2];
      double xc = (1.5*(cellV-N)+1.0)*Rc;
      double yc = (2*cellU-cellV-N)*rc;
      if (((std::abs(xloc-xc) <= rc) && (std::abs(yloc-yc) <= 0.5*Rc)) ||
	  ((std::abs(yloc-yc) <= Rc) && 
	   (std::abs(xloc-xc) <= sqrt3_*std::abs(yloc-yc-Rc)))) {
	found = true; break;
      }
    }
    if (found) break;
  }
  if (!found) { cellU = cu0; cellV = cv0; }
}

std::pair<int,float> HGCalDDDConstants::getIndex(int lay, bool reco) const {

  int ll = lay - hgpar_->firstLayer_;
  if (ll<0 || ll>=(int)(hgpar_->layerIndex_.size())) return std::pair<int,float>(-1,0);
  int   indx(0);
  float cell(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) { 
    if (reco && ll>=(int)(hgpar_->depthIndex_.size()))  return std::pair<int,float>(-1,0);
    indx = (reco ? hgpar_->depthLayerF_[ll] : hgpar_->layerIndex_[ll]);
    cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
  } else {
    indx = hgpar_->layerIndex_[ll];
    if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	(mode_ == HGCalGeometryMode::Hexagon8Full)) {
      cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
    } else {
      cell = hgpar_->dPhiEtaBH_[ll];
    }
  }
  return std::pair<int,float>(indx,cell);
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay) const {

  const double waferX = hgpar_->waferPosX_[wafer];
  const double waferY = hgpar_->waferPosY_[wafer];
  double xc[HGCalParameters::k_CornerSize], yc[HGCalParameters::k_CornerSize];
  xc[0] = waferX+rmax_; yc[0] = waferY-0.5*hexside_;
  xc[1] = waferX+rmax_; yc[1] = waferY+0.5*hexside_;
  xc[2] = waferX;       yc[2] = waferY+hexside_;
  xc[3] = waferX-rmax_; yc[3] = waferY+0.5*hexside_;
  xc[4] = waferX+rmax_; yc[4] = waferY-0.5*hexside_;
  xc[5] = waferX;       yc[5] = waferY-hexside_;
  bool cornerOne(false), cornerAll(true);
  for (unsigned int k=0; k<HGCalParameters::k_CornerSize; ++k) {
    double rpos = std::sqrt(xc[k]*xc[k]+yc[k]*yc[k]);
    if ((rpos >= hgpar_->rMinLayHex_[lay]) && 
	(rpos <= hgpar_->rMaxLayHex_[lay])) cornerOne = true;
    else                                    cornerAll = false;
  }
  bool in =  (hgpar_->defineFull_) ? cornerOne : cornerAll;
  return in;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
