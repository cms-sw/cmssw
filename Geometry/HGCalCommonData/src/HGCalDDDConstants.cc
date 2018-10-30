#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <algorithm>
#include <functional>
#include <numeric>

//#define EDM_ML_DEBUG

static const int maxType = 2;
static const int minType = 0;

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
  maxWafersPerLayer_ = 0;
  for (int simreco = 0; simreco < 2; ++simreco) {
    tot_layers_[simreco] = layersInit((bool)simreco);
    max_modules_layer_[simreco].resize(tot_layers_[simreco]+1);
    for (unsigned int layer=1; layer <= tot_layers_[simreco]; ++layer) {
      max_modules_layer_[simreco][layer] = modulesInit(layer,(bool)simreco);
      if (simreco == 1) {
	modHalf_ += max_modules_layer_[simreco][layer];
	maxWafersPerLayer_ = std::max(maxWafersPerLayer_,
				      max_modules_layer_[simreco][layer]);
#ifdef EDM_ML_DEBUG
	edm::LogVerbatim("HGCalGeom") << "Layer " << layer << " with " 
					<< max_modules_layer_[simreco][layer] 
					<< ":" << modHalf_ << " modules";
#endif
      }
    }
  }
  tot_wafers_ = wafers();
    
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants initialized for " 
				<< name << " with " << layers(false) << ":" 
				<< layers(true) << " layers, " << wafers() 
				<< ":" << 2*modHalf_ << " wafers with maximum "
				<< maxWafersPerLayer_ << " per layer and "
				<< "maximum of " << maxCells(false) << ":" 
				<< maxCells(true) << " cells";
#endif
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull) ||
      (mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int wminT(9999999), wmaxT(-9999999), kount1(0), kount2(0);
    for (unsigned int i=0; i<getTrFormN(); ++i) {
      int lay0 = getTrForm(i).lay;
      int wmin(9999999), wmax(-9999999), kount(0);
      for (int wafer=0; wafer<sectors(); ++wafer) {
	bool waferIn = waferInLayer(wafer,lay0,true);
	if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	    (mode_ == HGCalGeometryMode::Hexagon8Full)) {
	  int kndx = HGCalWaferIndex::waferIndex(lay0,HGCalWaferIndex::waferU(hgpar_->waferCopy_[wafer]),HGCalWaferIndex::waferV(hgpar_->waferCopy_[wafer]));
	  waferIn_[kndx] = waferIn;
	}
	if (waferIn) {
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
  const auto & index = getIndex(lay, reco);
  if (index.first < 0) return std::make_pair(-1,-1);
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
      return std::make_pair(-1,-1);
    } else {
      // Now the cell
      xx -= hgpar_->waferPosX_[wafer];
      yy -= hgpar_->waferPosY_[wafer];
      if (hgpar_->waferTypeT_[wafer] == 1) 
	return std::make_pair(wafer,cellHex(xx, yy, 
					    0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[0], 
					    hgpar_->cellFineX_, hgpar_->cellFineY_));
      else
	return std::make_pair(wafer,cellHex(xx, yy, 
					    0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[1], 
					    hgpar_->cellCoarseX_, hgpar_->cellCoarseY_));
    }
  } else {
    return std::make_pair(-1,-1);
  }
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
  
  int irad(-1), iphi(-1), type(-1);
  const auto & indx  = getIndex(layer,reco);
  if (indx.first < 0) return std::array<int,3>{ {irad,iphi,type} };
  double xx    = (z > 0) ? x : -x;
  double r     = (reco ? std::sqrt(x*x+y*y) : 
		  HGCalParameters::k_ScaleFromDDD*std::sqrt(x*x+y*y));
  double phi   = (r == 0. ? 0. : std::atan2(y,xx));
  if (phi < 0) phi += (2.0*M_PI);
  type         = hgpar_->scintType(layer);
  auto  ir     = std::lower_bound(hgpar_->radiusLayer_[type].begin(),
				  hgpar_->radiusLayer_[type].end(), r);
  irad   = (int)(ir - hgpar_->radiusLayer_[type].begin());
  irad   = std::min(std::max(irad,hgpar_->iradMinBH_[indx.first]),
		    hgpar_->iradMaxBH_[indx.first]);
  iphi         = 1 + (int)(phi/indx.second);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "assignCellTrap Input " << x << ":" << y
				<< ":" << z << ":" << layer << ":" << reco
				<< " x|r " << xx << ":" << r << " phi " 
				<< phi << " o/p " << irad << ":" << iphi 
				<< ":" << type;
#endif
  return std::array<int,3>{ {irad,iphi,type} };
}

bool HGCalDDDConstants::cellInLayer(int waferU, int waferV, int cellU, 
				    int cellV, int lay, bool reco) const {
  const auto & indx  = getIndex(lay,true);
  if (indx.first >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	(mode_ == HGCalGeometryMode::Hexagon8Full) ||
	(mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      const auto & xy =  ((mode_ == HGCalGeometryMode::Hexagon8) ||
			  (mode_ == HGCalGeometryMode::Hexagon8Full)) ?
	locateCell(lay, waferU, waferV, cellU, cellV, reco, true, false) : 
	locateCell(cellU, lay, waferU, reco);
      double rpos = sqrt(xy.first*xy.first + xy.second*xy.second);
      return ((rpos >= hgpar_->rMinLayHex_[indx.first]) && 
	      (rpos <= hgpar_->rMaxLayHex_[indx.first]));
    } else {
      return true;
    }
  } else {
    return false;
  }
}

double HGCalDDDConstants::cellThickness(int layer, int waferU, 
					int waferV) const {

  double thick(-1);
  if ((mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer,waferU,waferV));
    int type = ((itr == hgpar_->typesInLayers_.end() ? maxType : 
		 hgpar_->waferTypeL_[itr->second]));
    thick    = 10000.0*hgpar_->cellThickness_[type];
  } else if ((mode_ == HGCalGeometryMode::Hexagon) ||
	     (mode_ == HGCalGeometryMode::HexagonFull)) {
    int type = (((waferU>=0)&&(waferU<(int)(hgpar_->waferTypeL_.size()))) ? 
		hgpar_->waferTypeL_[waferU] : minType);
    thick    = 100.0*type;
  }
  return thick;
}

double HGCalDDDConstants::cellSizeHex(int type) const {
  int    indx = (((mode_ == HGCalGeometryMode::Hexagon8) ||
		  (mode_ == HGCalGeometryMode::Hexagon8Full)) ?
		 ((type >= 1) ? 1 : 0) : ((type == 1) ? 1 : 0)); 
  double cell = ((mode_ == HGCalGeometryMode::Trapezoid) ? 
		 0.5*hgpar_->cellSize_[indx] :
		 0.5*HGCalParameters::k_ScaleFromDDD*hgpar_->cellSize_[indx]);
  return cell;
}

HGCalDDDConstants::CellType HGCalDDDConstants::cellType(int type, int cellU,
							int cellV) const {
  // type=0: in the middle; 1..6: the edges clocwise from bottom left;
  //     =11..16: the corners clockwise from bottom
  int N = (type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  if (cellU == 0) {
    if      (cellV == 0)         return HGCalDDDConstants::CellType::BottomLeftCorner;
    else if (cellV-cellU == N-1) return HGCalDDDConstants::CellType::BottomCorner;
    else                         return HGCalDDDConstants::CellType::BottomLeftEdge;
  } else if (cellV == 0) {
    if (cellU-cellV == N)        return HGCalDDDConstants::CellType::TopLeftCorner;
    else                         return HGCalDDDConstants::CellType::LeftEdge;
  } else if (cellU-cellV == N) {
    if (cellU == 2*N-1)          return HGCalDDDConstants::CellType::TopCorner;
    else                         return HGCalDDDConstants::CellType::TopLeftEdge;
  } else if (cellU == 2*N-1) {
    if (cellV == 2*N-1)          return HGCalDDDConstants::CellType::TopRightCorner;
    else                         return HGCalDDDConstants::CellType::TopRightEdge;
  } else if (cellV == 2*N-1) {
    if (cellV-cellU == N-1)      return HGCalDDDConstants::CellType::BottomRightCorner;
    else                         return HGCalDDDConstants::CellType::RightEdge;
  } else if (cellV-cellU == N-1) {
    return HGCalDDDConstants::CellType::BottomRightEdge;
  } else if ((cellU > 2*N-1) || (cellV > 2*N-1) || (cellV >= (cellU+N)) ||
	     (cellU > (cellV+N))) {
    return HGCalDDDConstants::CellType::UndefinedType;
  } else {
    return HGCalDDDConstants::CellType::CentralType;
  }
}

double HGCalDDDConstants::distFromEdgeHex(double x, double y, double z) const {

  //Assming the point is within a hexagonal plane of the wafer, calculate
  //the shortest distance from the edge
  if (z < 0) x = -x;
  double dist(0);
  //Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalParameters::k_ScaleFromDDD*x;
  double yy = HGCalParameters::k_ScaleFromDDD*y;
  int sizew = (int)(hgpar_->waferPosX_.size());
  int wafer = sizew;
  //Transform to the local coordinate frame of the wafer first
  for (int k=0; k<sizew; ++k) {
    double dx = std::abs(xx-hgpar_->waferPosX_[k]);
    double dy = std::abs(yy-hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_ &&
	((dy <= 0.5*hexside_) || (dx*tan30deg_ <= (hexside_-dy)))) {
      wafer   = k;
      xx     -= hgpar_->waferPosX_[k];
      yy     -= hgpar_->waferPosY_[k];
      break;
    }
  }
  //Look at only one quarter (both x,y are positive)
  if (wafer < sizew) {
    if (std::abs(yy) < 0.5*hexside_) {
      dist = rmax_ - std::abs(xx);
    } else {
      dist = 0.5*((rmax_-std::abs(xx))-sqrt3_*(std::abs(yy)-0.5*hexside_));
    }
  } else {
    dist = 0;
  }
  dist *= HGCalParameters::k_ScaleToDDD;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DistFromEdgeHex: Local " << xx << ":"
				<< yy << " wafer " << wafer << " flag " 
				<< (wafer < sizew) << " Distance " << rmax_ 
				<< ":" << (rmax_-std::abs(xx)) << ":" 
				<< (std::abs(yy)-0.5*hexside_) << ":"
				<< 0.5*hexside_ << ":" << dist;
#endif
  return dist;
}

double HGCalDDDConstants::distFromEdgeTrap(double x, double y, double z) const {

  //Assming the point is within the eta-phi plane of the scintillator tile,
  //calculate the shortest distance from the edge
  int lay      = getLayer(z,false);
  double xx    = (z < 0) ? -x : x;
  int indx     = layerIndex(lay,false);
  double r     = HGCalParameters::k_ScaleFromDDD*std::sqrt(x*x+y*y);
  double phi   = (r == 0. ? 0. : std::atan2(y,xx));
  if (phi < 0) phi += (2.0*M_PI);
  int    type  = hgpar_->scintType(lay);
  double cell  = hgpar_->scintCellSize(lay);
  //Compare with the center of the tile find distances along R and also phi
  //Take the smaller value
  auto  ir     = std::lower_bound(hgpar_->radiusLayer_[type].begin(),
				  hgpar_->radiusLayer_[type].end(), r);
  int irad     = (int)(ir-hgpar_->radiusLayer_[type].begin());
  if      (irad < hgpar_->iradMinBH_[indx]) irad = hgpar_->iradMinBH_[indx];
  else if (irad > hgpar_->iradMaxBH_[indx]) irad = hgpar_->iradMaxBH_[indx];
  int    iphi  = 1 + (int)(phi/cell);
  double dphi  = std::max(0.0,(0.5*cell-std::abs(phi-(iphi-0.5)*cell)));
  double dist  = std::min((r-hgpar_->radiusLayer_[type][irad-1]),
			  (hgpar_->radiusLayer_[type][irad]-r));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DistFromEdgeTrap: Global " << x << ":"
				<< y << ":" << z << " Layer " << lay 
				<< " Index " << indx << ":" << type << " xx " 
				<< xx << " R " << r << ":" << irad << ":" 
				<< hgpar_->radiusLayer_[type][irad-1] << ":"
				<< hgpar_->radiusLayer_[type][irad] 
				<< " Phi " << phi << ":" << iphi << ":" 
				<< (iphi-0.5)*cell << " cell " << cell 
				<< " Dphi " << dphi << " Dist " << dist
				<< ":" << r*dphi;
#endif
  return HGCalParameters::k_ScaleToDDD*std::min(r*dphi,dist);
}

int HGCalDDDConstants::getLayer(double z, bool reco) const {

  //Get the layer # from the gloabl z coordinate
  unsigned int k  = 0;
  double       zz = (reco ? std::abs(z) : 
		     HGCalParameters::k_ScaleFromDDD*std::abs(z));
  const auto& zLayerHex = hgpar_->zLayerHex_;
  std::find_if(zLayerHex.begin()+1,zLayerHex.end(),[&k,&zz,&zLayerHex](double zLayer){ ++k; return zz < 0.5*(zLayerHex[k-1]+zLayerHex[k]);});
  int lay = k;
  if (((mode_ == HGCalGeometryMode::Hexagon) ||
       (mode_ == HGCalGeometryMode::HexagonFull)) & reco) {
    int indx = layerIndex(lay,false);
    if (indx >= 0) lay = hgpar_->layerGroup_[indx];
  } else {
    lay += (hgpar_->firstLayer_ - 1);
  }
  return lay;
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
    mytr = hgpar_->getModule(type, reco);
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

int HGCalDDDConstants::getTypeTrap(int layer) const {
  //Get the module type for scinitllator
  if (mode_ == HGCalGeometryMode::Trapezoid) {
    return hgpar_->scintType(layer);
  } else {
    return -1;
  }
}

int HGCalDDDConstants::getTypeHex(int layer, int waferU, int waferV) const {
  //Get the module type for a silicon wafer
  if ((mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer,waferU,waferV));
    return ((itr == hgpar_->typesInLayers_.end() ? 2 : 
	     hgpar_->waferTypeL_[itr->second]));
  } else {
    return -1;
  }
}

bool HGCalDDDConstants::isHalfCell(int waferType, int cell) const {
  if (waferType < 1 || cell < 0) return false;
  return waferType == 2 ? hgpar_->cellCoarseHalf_[cell] : hgpar_->cellFineHalf_[cell];
}

bool HGCalDDDConstants::isValidHex(int lay, int mod, int cell, 
				   bool reco) const {
  //Check validity for a layer|wafer|cell of pre-TDR version
  bool result(false), resultMod(false);
  int  cellmax(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    int32_t copyNumber = hgpar_->waferCopy_[mod];
    result = ((lay > 0 && lay <= (int)(layers(reco))));
    if (result) {
      const int32_t lay_idx = reco ? (lay-1)*3 + 1 : lay;
      const auto& the_modules = hgpar_->copiesInLayers_[lay_idx];
      auto moditr = the_modules.find(copyNumber);
      result = resultMod = (moditr != the_modules.end());
#ifdef EDM_ML_DEBUG
      if (!result) 
	edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay 
				      << ":" << lay_idx << " Copy " 
				      << copyNumber << ":" << mod
				      << " Flag " << result;
#endif
      if (result) {
	if (moditr->second >= 0) {
	  if (mod >= (int)(hgpar_->waferTypeT_.size()))
	    edm::LogWarning("HGCalGeom") << "Module no. out of bound for " 
					 << mod << " to be compared with " 
					 << (hgpar_->waferTypeT_).size() 
					 << " ***** ERROR *****";
	  cellmax = (hgpar_->waferTypeT_[mod]==1) ? 
	    (int)(hgpar_->cellFineX_.size()) : (int)(hgpar_->cellCoarseX_.size());
	  result = (cell >=0 && cell <=  cellmax);
	} else {
	  result = isValidCell(lay_idx, mod, cell);
	}
      }
    }
  }
    
#ifdef EDM_ML_DEBUG
  if (!result) 
    edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay << ":"
				  << (lay > 0 && (lay <= (int)(layers(reco))))
				  << " Module " << mod << ":" << resultMod 
				  << " Cell " << cell << ":" << cellmax << ":"
				  << (cell >=0 && cell <= cellmax)
				  << ":" << maxCells(reco); 
#endif
  return result;
}

bool HGCalDDDConstants::isValidHex8(int layer, int modU, int modV, int cellU, 
				    int cellV) const {
  //Check validity for a layer|wafer|cell of post-TDR version
  int indx  = HGCalWaferIndex::waferIndex(layer,modU,modV);
  auto itr  = hgpar_->typesInLayers_.find(indx);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:WaferType "
				<< layer << ":" << modU << ":" << modV << ":"
				<< indx << " Test " 
				<< (itr == hgpar_->typesInLayers_.end());
#endif
  if (itr == hgpar_->typesInLayers_.end()) return false;
  auto jtr  = waferIn_.find(indx);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:WaferIn "
				<< jtr->first << ":" << jtr->second;
#endif
  if (!(jtr->second))                      return false;
  int  N = ((hgpar_->waferTypeL_[itr->second] == 0) ? hgpar_->nCellsFine_ :
	    hgpar_->nCellsCoarse_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:Cell "
				<< cellU << ":" << cellV << ":" << N 
				<< " Tests " << (cellU >= 0) << ":"
				<< (cellU < 2*N) << ":" << (cellV >= 0) << ":"
				<< (cellV < 2*N) << ":" << ((cellV-cellU) < N)
				<< ":" << ((cellU-cellV) <= N);
#endif
  if ((cellU >= 0) && (cellU < 2*N) && (cellV >= 0) && (cellV < 2*N)) {
    return (((cellV-cellU) < N) && ((cellU-cellV) <= N));
  } else {
    return false;
  }
}

bool HGCalDDDConstants::isValidTrap(int layer, int irad, int iphi) const {
  //Check validity for a layer|eta|phi of scintillator
  const auto & indx  = getIndex(layer,true);
  if (indx.first < 0) return false;
  return ((irad >= hgpar_->iradMinBH_[indx.first]) &&
	  (irad <= hgpar_->iradMaxBH_[indx.first]) &&
	  (iphi > 0) && (iphi <= hgpar_->scintCells(layer)));
}

unsigned int HGCalDDDConstants::layers(bool reco) const {
  return tot_layers_[(int)reco];
}

int HGCalDDDConstants::layerIndex(int lay, bool reco) const {
  int ll = lay - hgpar_->firstLayer_;
  if (ll<0 || ll>=(int)(hgpar_->layerIndex_.size())) return -1;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) { 
    if (reco && ll>=(int)(hgpar_->depthIndex_.size())) return -1;
    return (reco ? hgpar_->depthLayerF_[ll] : hgpar_->layerIndex_[ll]);
  } else {
    return (hgpar_->layerIndex_[ll]);
  }
}

unsigned int HGCalDDDConstants::layersInit(bool reco) const {
  return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());
}

std::pair<float,float> HGCalDDDConstants::locateCell(int cell, int lay, 
						     int type, bool reco) const {
  // type refers to wafer # for hexagon cell
  float x(999999.), y(999999.);
  const auto & index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return std::make_pair(x,y);
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
  return std::make_pair(x,y);
}

std::pair<float,float> HGCalDDDConstants::locateCell(int lay, int waferU,
						     int waferV, int cellU,
						     int cellV, bool reco,
						     bool all, bool
#ifdef EDM_ML_DEBUG
						     debug
#endif
						     ) const {

  float x(0), y(0);
  int  indx = HGCalWaferIndex::waferIndex(lay,waferU,waferV);
  auto itr  = hgpar_->typesInLayers_.find(indx);
  int  type = ((itr == hgpar_->typesInLayers_.end()) ? 2 : 
	       hgpar_->waferTypeL_[itr->second]);
#ifdef EDM_ML_DEBUG
  if (debug) 
    edm::LogVerbatim("HGCalGeom") << "LocateCell " << lay << ":" << waferU
				  << ":" << waferV << ":" << indx << ":"
				  << (itr == hgpar_->typesInLayers_.end())
				  << ":" << type;
#endif
  int  kndx = cellV*100 + cellU;
  if (type == 0) {
    auto ktr = hgpar_->cellFineIndex_.find(kndx);
    if (ktr != hgpar_->cellFineIndex_.end()) {
      x = hgpar_->cellFineX_[ktr->second];
      y = hgpar_->cellFineY_[ktr->second];
    }
#ifdef EDM_ML_DEBUG
    if (debug) 
      edm::LogVerbatim("HGCalGeom") << "Fine " << cellU << ":" << cellV << ":"
				    << kndx << ":" << x << ":" << y << ":"
				    << (ktr != hgpar_->cellFineIndex_.end());
#endif
  } else {
    auto ktr = hgpar_->cellCoarseIndex_.find(kndx);
    if (ktr != hgpar_->cellCoarseIndex_.end()) {
      x = hgpar_->cellCoarseX_[ktr->second];
      y = hgpar_->cellCoarseY_[ktr->second];
    }
#ifdef EDM_ML_DEBUG
    if (debug) 
      edm::LogVerbatim("HGCalGeom") << "Coarse " << cellU << ":" << cellV <<":"
				    << kndx << ":" << x << ":" << y << ":"
				    << (ktr != hgpar_->cellCoarseIndex_.end());
#endif
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  if (all) {
    const auto & xy = waferPosition(waferU, waferV, reco);
    x += xy.first;
    y += xy.second;
#ifdef EDM_ML_DEBUG
    if (debug) 
      edm::LogVerbatim("HGCalGeom") << "With wafer " << x << ":" << y << ":"
				    << xy.first << ":" << xy.second;
#endif
  }
  return std::make_pair(x,y);
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
  return std::make_pair(x,y);
}

std::pair<float,float> HGCalDDDConstants::locateCellTrap(int lay, int irad,
							 int iphi, bool reco) const {
  
  float x(0), y(0);
  const auto & indx = getIndex(lay,reco);
  if (indx.first >= 0) {
    int    ir   = std::abs(irad);
    int    type = hgpar_->scintType(lay);
    double phi  = (iphi-0.5)*indx.second;
    double z    = hgpar_->zLayerHex_[indx.first];
    double r    = 0.5*(hgpar_->radiusLayer_[type][ir-1] + 
		       hgpar_->radiusLayer_[type][ir]);
    std::pair<double,double> range = rangeR(z,true);
    r           = std::max(range.first, std::min(r,range.second));
    x           = r*std::cos(phi);
    y           = r*std::sin(phi);
    if (irad < 0) x =-x;
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(x,y);
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

  const auto & index = getIndex(lay, reco);
  if (index.first < 0) return 0;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    unsigned int cells(0);
    for (unsigned int k=0; k<hgpar_->waferTypeT_.size(); ++k) {
      if (waferInLayerTest(k,index.first,hgpar_->defineFull_)) {
	unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	  (hgpar_->cellFineX_.size()) : (hgpar_->cellCoarseX_.size());
	if (cell > cells) cells = cell;
      }
    }
    return (int)(cells);
  } else if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	     (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int cells(0);
    for (unsigned int k=0; k<hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayerTest(k,index.first,hgpar_->defineFull_)) { 
	auto itr   = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(lay,HGCalWaferIndex::waferU(hgpar_->waferCopy_[k]),HGCalWaferIndex::waferV(hgpar_->waferCopy_[k])));
	int  type  = ((itr == hgpar_->typesInLayers_.end()) ? 2 : 
		      hgpar_->waferTypeL_[itr->second]);
	int N      = (type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
	cells      = std::max(cells,3*N*N);
      }
    }
    return cells;
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    return hgpar_->scintCells(index.first+hgpar_->firstLayer_);
  } else {
    return 0;
  }
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {

  int kymax(0);
  const auto & index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0) return kymax;
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) {
    for (unsigned int k=0; k<hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayerTest(k,i,hgpar_->defineFull_)) {
	int ky = ((hgpar_->waferCopy_[k])/100)%100;
	if (ky > kymax) kymax = ky;
      }
    }
  } else if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	     (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    kymax = 1+2*hgpar_->waferUVMaxLayer_[index.first];
  }
  return kymax;
}

int HGCalDDDConstants::modifyUV(int uv, int type1, int type2) const {
  // Modify u/v for transition of type1 to type2
  return (((type1==type2) || (type1*type2 !=0)) ? uv : 
	  ((type1==0) ? (2*uv+1)/3 : (3*uv)/2));
}

int HGCalDDDConstants::modules(int lay, bool reco) const {
  if (getIndex(lay,reco).first < 0) return 0;
  else                              return max_modules_layer_[(int)reco][lay];
}

int HGCalDDDConstants::modulesInit(int lay, bool reco) const {
  int nmod(0);
  const auto & index = getIndex(lay, reco);
  if (index.first < 0) return nmod;
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
      if (waferInLayerTest(k,index.first,hgpar_->defineFull_)) ++nmod;
    }
  } else {
    nmod = 1+hgpar_->lastModule_[index.first]-hgpar_->firstModule_[index.first];
  }
  return nmod;
}

double HGCalDDDConstants::mouseBite(bool reco) const {

  return (reco ? hgpar_->mouseBite_ : HGCalParameters::k_ScaleToDDD*hgpar_->mouseBite_);
}

int HGCalDDDConstants::numberCells(bool reco) const {

  int cells(0);
  unsigned int nlayer = (reco) ? hgpar_->depth_.size() : hgpar_->layer_.size();
  for (unsigned k=0; k<nlayer; ++k) {
    std::vector<int> ncells = numberCells(((reco) ? hgpar_->depth_[k] : hgpar_->layer_[k]), reco);
    cells = std::accumulate(ncells.begin(),ncells.end(),cells);
  }
  return cells;
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {

  const auto & index = getIndex(lay, reco);
  int i = index.first;
  std::vector<int> ncell;
  if (i >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon) ||
	(mode_ == HGCalGeometryMode::HexagonFull)) {
      for (unsigned int k=0; k<hgpar_->waferTypeT_.size(); ++k) {
	if (waferInLayerTest(k,i,hgpar_->defineFull_)) {
	  unsigned int cell = (hgpar_->waferTypeT_[k]==1) ? 
	    (hgpar_->cellFineX_.size()) : (hgpar_->cellCoarseX_.size());
	  ncell.emplace_back((int)(cell));
	}
      }
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      int nphi = hgpar_->scintCells(lay);
      for (int k=hgpar_->firstModule_[i]; k<=hgpar_->lastModule_[i]; ++k)
	ncell.emplace_back(nphi);
    } else {
      for (unsigned int k=0; k<hgpar_->waferCopy_.size(); ++k) {
	if (waferInLayerTest(k,index.first,hgpar_->defineFull_)) { 
	  int cell   = numberCellsHexagon(lay,
					  HGCalWaferIndex::waferU(hgpar_->waferCopy_[k]),
					  HGCalWaferIndex::waferV(hgpar_->waferCopy_[k]),
					  true);
	  ncell.emplace_back(cell);
	}
      }
    }
  }
  return ncell;
}

int HGCalDDDConstants::numberCellsHexagon(int wafer) const {

  if (wafer >= 0 && wafer < (int)(hgpar_->waferTypeT_.size())) {
    if (hgpar_->waferTypeT_[wafer]==1) 
      return (int)(hgpar_->cellFineX_.size());
    else 
      return (int)(hgpar_->cellCoarseX_.size());
  } else {
    return 0;
  }
}

int HGCalDDDConstants::numberCellsHexagon(int lay, int waferU, int waferV,
					  bool flag) const {
  auto itr   = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(lay,waferU,waferV));
  int  type  = ((itr == hgpar_->typesInLayers_.end()) ? 2 : 
		hgpar_->waferTypeL_[itr->second]);
  int N      = (type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  if (flag) return (3*N*N);
  else      return N;
}

std::pair<double,double> HGCalDDDConstants::rangeR(double z, bool reco) const {
  double rmin(0), rmax(0), zz(0);
  if (hgpar_->detectorType_ > 0) {
    zz   = (reco ? std::abs(z) : HGCalParameters::k_ScaleFromDDD*std::abs(z));
    if (hgpar_->detectorType_ <= 2) {
      rmin = HGCalGeomTools::radius(zz,hgpar_->zFrontMin_,
				    hgpar_->rMinFront_, hgpar_->slopeMin_);
    } else {
      rmin = HGCalGeomTools::radius(zz, hgpar_->firstLayer_,
				    hgpar_->firstMixedLayer_,
				    hgpar_->zLayerHex_,
				    hgpar_->radiusMixBoundary_);
    }
    if ((hgpar_->detectorType_ == 2) && 
	(zz >= hgpar_->zLayerHex_[hgpar_->firstMixedLayer_ - 1])) {
      rmax = HGCalGeomTools::radius(zz, hgpar_->firstLayer_,
				    hgpar_->firstMixedLayer_,
				    hgpar_->zLayerHex_,
				    hgpar_->radiusMixBoundary_);
    } else {
      rmax = HGCalGeomTools::radius(zz, hgpar_->zFrontTop_,
				    hgpar_->rMaxFront_, hgpar_->slopeTop_);
    }
  }
  if (!reco) {
    rmin *= HGCalParameters::k_ScaleToDDD;
    rmax *= HGCalParameters::k_ScaleToDDD;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo:rangeR: " << z << ":" << zz
				<< " R " << rmin << ":" << rmax;
#endif
  return std::pair<double,double>(rmin,rmax);
}

std::pair<double,double> HGCalDDDConstants::rangeZ(bool reco) const {
  double zmin = (hgpar_->zLayerHex_[0] - hgpar_->waferThick_);
  double zmax = (hgpar_->zLayerHex_[hgpar_->zLayerHex_.size()-1] +
		 hgpar_->waferThick_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DDHGCalEEAlgo:rangeZ: " << zmin << ":" 
				<< zmax << ":" << hgpar_->waferThick_;
#endif
  if (!reco) {
    zmin *= HGCalParameters::k_ScaleToDDD;
    zmax *= HGCalParameters::k_ScaleToDDD;
  }
  return std::pair<double,double>(zmin,zmax);
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
  return std::make_pair(row,col);
}

std::pair<int,int> HGCalDDDConstants::simToReco(int cell, int lay, int mod,
						bool half) const {
  
  if ((mode_ != HGCalGeometryMode::Hexagon) &&
      (mode_ != HGCalGeometryMode::HexagonFull)) {
    return std::make_pair(cell,lay);
  } else {
    const auto & index = getIndex(lay, false);
    int i = index.first;
    if (i < 0) {
      edm::LogWarning("HGCalGeom") << "Wrong Layer # " << lay 
				   << " not in the list ***** ERROR *****";
      return std::make_pair(-1,-1);
    }
    if (mod >= (int)(hgpar_->waferTypeL_).size()) {
      edm::LogWarning("HGCalGeom") << "Invalid Wafer # " << mod 
				   << "should be < "
				   << (hgpar_->waferTypeL_).size()
				   << " ***** ERROR *****";
      return std::make_pair(-1,-1);
    }
    int depth(-1);
    int kx   = cell;
    int type = hgpar_->waferTypeL_[mod];
    if (type == 1) {
      depth = hgpar_->layerGroup_[i];
    } else if (type == 2) {
      depth = hgpar_->layerGroupM_[i];
    } else {
      depth = hgpar_->layerGroupO_[i];
    }    
    return std::make_pair(kx,depth);
  }
}

int HGCalDDDConstants::waferFromCopy(int copy) const {
  const int ncopies = hgpar_->waferCopy_.size();
  int  wafer(ncopies);
  bool result(false);
  for (int k=0; k<ncopies; ++k) {
    if (copy == hgpar_->waferCopy_[k]) {
      wafer  = k;
      result = true;
      break;
    }
  }
  if (!result) {
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
  //Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalParameters::k_ScaleFromDDD*x;
  double yy = HGCalParameters::k_ScaleFromDDD*y;
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
					  int& celltype, double& wt, bool 
#ifdef EDM_ML_DEBUG
					  debug
#endif
					  ) const {

  double xx(HGCalParameters::k_ScaleFromDDD*x);
  double yy(HGCalParameters::k_ScaleFromDDD*y);
  waferU = waferV = 1+hgpar_->waferUVMax_;
  for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
    double dx = std::abs(xx-hgpar_->waferPosX_[k]);
    double dy = std::abs(yy-hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5*hexside_) || (dx*tan30deg_ <= (hexside_-dy))) {
	waferU   = HGCalWaferIndex::waferU(hgpar_->waferCopy_[k]);
	waferV   = HGCalWaferIndex::waferV(hgpar_->waferCopy_[k]);
	auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer,waferU,waferV));
	celltype = ((itr == hgpar_->typesInLayers_.end()) ? 2 : 
		    hgpar_->waferTypeL_[itr->second]);
#ifdef EDM_ML_DEBUG
	if (debug) 
	  edm::LogVerbatim("HGCalGeom") << "WaferFromPosition:: Input " << xx
					<< ":" << yy << " compared with "
					<< hgpar_->waferPosX_[k] << ":"
					<< hgpar_->waferPosY_[k]
					<< " difference " << dx << ":" << dy
					<< ":" << dx*tan30deg_ << ":" 
					<< (hexside_-dy) << " comparator " 
					<< rmax_ << ":"	<< hexside_ <<" wafer "
					<< waferU << ":" << waferV << ":"
					<< celltype;
#endif
	xx      -= hgpar_->waferPosX_[k];
	yy      -= hgpar_->waferPosY_[k];
	break;
      }
    }
  }
  if (std::abs(waferU) <= hgpar_->waferUVMax_) {
    cellHex(xx, yy, celltype, cellU, cellV
#ifdef EDM_ML_DEBUG
	    , debug
#endif
	    );
    wt    = ((celltype < 2) ? 
	     (hgpar_->cellThickness_[celltype]/hgpar_->waferThick_) : 1.0);
  } else {
    cellU = cellV = 2*hgpar_->nCellsFine_;
    wt    = 1.0;
    celltype =-1;
  }
#ifdef EDM_ML_DEBUG
  if (celltype < 0) {
    double x1(HGCalParameters::k_ScaleFromDDD*x);
    double y1(HGCalParameters::k_ScaleFromDDD*y);
    edm::LogVerbatim("HGCalGeom") << "waferFromPosition: Bad type for X " 
				  << x << ":" << x1 << ":" << xx << " Y " << y
				  << ":" << y1 << ":" << yy << " Wafer " 
				  << waferU << ":" << waferV << " Cell "
				  << cellU << ":" << cellV;
    for (unsigned int k=0; k<hgpar_->waferPosX_.size(); ++k) {
      double dx = std::abs(x1-hgpar_->waferPosX_[k]);
      double dy = std::abs(y1-hgpar_->waferPosY_[k]);
      edm::LogVerbatim("HGCalGeom") << "Wafer [" << k << "] Position (" 
				    << hgpar_->waferPosX_[k] << ", "
				    << hgpar_->waferPosY_[k] << ") difference "
				    << dx << ":" << dy << ":" << dx*tan30deg_
				    << ":" << hexside_-dy << " Paramerers " 
				    << rmax_ << ":" << hexside_;
    }
  }
#endif
}
  
bool HGCalDDDConstants::waferInLayer(int wafer, int lay, bool reco) const {

  const auto & indx = getIndex(lay, reco);
  if (indx.first < 0) return false;
  return waferInLayerTest(wafer,indx.first,hgpar_->defineFull_);
}
  
bool HGCalDDDConstants::waferFullInLayer(int wafer, int lay, bool reco) const {
  const auto & indx = getIndex(lay, reco);
  if (indx.first < 0) return false;
  return waferInLayerTest(wafer,indx.first,false);
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
  return std::make_pair(xx,yy);
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
  return std::make_pair(xx,yy);
}

int HGCalDDDConstants::waferType(DetId const& id) const {

  int type(1);
  if ((mode_ == HGCalGeometryMode::Hexagon8) ||
      (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    type = ((id.det() != DetId::Forward) ? (1 + HGCSiliconDetId(id).type()) :
	    (1 + HFNoseDetId(id).type()));
  } else if ((mode_ == HGCalGeometryMode::Hexagon) ||
	     (mode_ == HGCalGeometryMode::HexagonFull)) {
    type = waferTypeL(HGCalDetId(id).wafer());
  }
  return type;
}

double HGCalDDDConstants::waferZ(int lay, bool reco) const {
  const auto & index = getIndex(lay, reco);
  if (index.first < 0) return 0;
  else                 return (reco ? hgpar_->zLayerHex_[index.first] : 
			       HGCalParameters::k_ScaleToDDD*hgpar_->zLayerHex_[index.first]);
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
    const auto & index = getIndex(layer, true);
    wafer = 1+hgpar_->lastModule_[index.first]-hgpar_->firstModule_[index.first];
  }
  return wafer;
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
				int& cellU, int& cellV, bool 
#ifdef EDM_ML_DEBUG
				debug
#endif
				) const {
  int N     = (cellType == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  double Rc = 2*rmax_/(3*N);
  double rc = 0.5*Rc*sqrt3_;
  double v0 = ((xloc/Rc -1.0)/1.5);
  int cv0   = (v0 > 0) ? (N + (int)(v0+0.5)) : (N - (int)(-v0+0.5));
  double u0 = (0.5*yloc/rc+0.5*cv0);
  int cu0   = (u0 > 0) ? (N/2 + (int)(u0+0.5)) : (N/2 - (int)(-u0+0.5));
  cu0       = std::max(0,std::min(cu0,2*N-1));
  cv0       = std::max(0,std::min(cv0,2*N-1));
  if (cv0-cu0 >= N) cv0 = cu0+N-1;
#ifdef EDM_ML_DEBUG
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellHex: input " << xloc << ":"
				  << yloc << ":" << cellType << " parameter "
				  << rc << ":" << Rc << " u0 " << u0 << ":"
				  << cu0 << " v0 " << v0 << ":" << cv0;
#endif
  bool found(false);
  static const int shift[3] = {0,1,-1};
  for (int i1=0; i1<3; ++i1) {
    cellU       = cu0 + shift[i1];
    for (int i2=0; i2<3; ++i2) {
      cellV     = cv0 + shift[i2];
      if (((cellV-cellU) < N) && ((cellU-cellV) <= N) && (cellU >= 0) && 
	  (cellV >= 0) && (cellU < 2*N) && (cellV < 2*N)) {
	double xc = (1.5*(cellV-N)+1.0)*Rc;
	double yc = (2*cellU-cellV-N)*rc;
	if ((std::abs(yloc-yc) <= rc) && (std::abs(xloc-xc) <= Rc) &&
	    ((std::abs(xloc-xc) <= 0.5*Rc) ||
	     (std::abs(yloc-yc) <= sqrt3_*(Rc-std::abs(xloc-xc))))) {
#ifdef EDM_ML_DEBUG
	  if (debug)
	    edm::LogVerbatim("HGCalGeom") << "cellHex: local " << xc << ":" 
					  << yc	<< " difference " 
					  << std::abs(xloc-xc) << ":"
					  << std::abs(yloc-yc) << ":"
					  << sqrt3_*(Rc-std::abs(yloc-yc))
					  << " comparator " << rc << ":" << Rc
					  << " (u,v) = (" << cellU << ","
					  << cellV << ")";
#endif
	  found = true; break;
	}
      }
    }
    if (found) break;
  }
  if (!found) { cellU = cu0; cellV = cv0; }
}

std::pair<int,float> HGCalDDDConstants::getIndex(int lay, bool reco) const {

  int indx = layerIndex(lay,reco);
  if (indx<0) return std::make_pair(-1,0);
  float cell(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) ||
      (mode_ == HGCalGeometryMode::HexagonFull)) { 
    cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
  } else {
    if ((mode_ == HGCalGeometryMode::Hexagon8) ||
	(mode_ == HGCalGeometryMode::Hexagon8Full)) {
      cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
    } else {
      cell = hgpar_->scintCellSize(lay);
    }
  }
  return std::make_pair(indx,cell);
}

bool HGCalDDDConstants::isValidCell(int lay, int wafer, int cell) const {

  // Calculate the position of the cell
  // Works for options HGCalHexagon/HGCalHexagonFull
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
  bool   result = ((rr >= hgpar_->rMinLayHex_[lay-1]) &&
		   (rr <= hgpar_->rMaxLayHex_[lay-1]) &&
		   (wafer < (int)(hgpar_->waferPosX_.size())));
#ifdef EDM_ML_DEBUG
  if (!result) 
    edm::LogVerbatim("HGCalGeom") << "Input " << lay << ":" << wafer << ":" 
				  << cell << " Position " << x << ":" << y 
				  << ":" << rr << " Compare Limits "
				  << hgpar_->rMinLayHex_[lay-1] << ":" 
				  << hgpar_->rMaxLayHex_[lay-1]
				  << " Flag " << result;
#endif
  return result;
}

bool HGCalDDDConstants::waferInLayerTest(int wafer, int lay, bool full) const {

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
  bool in =  full ? cornerOne : cornerAll;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "WaferInLayerTest: Layer " << lay
				<< " wafer " << wafer << " R-limits "
				<< hgpar_->rMinLayHex_[lay] << ":"
				<< hgpar_->rMaxLayHex_[lay] << " Corners "
				<< cornerOne << ":" << cornerAll << " In "
				<< in;
#endif
  return in;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
