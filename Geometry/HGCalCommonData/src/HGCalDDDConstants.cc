#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferType.h"

#include <algorithm>
#include <functional>
#include <numeric>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

HGCalDDDConstants::HGCalDDDConstants(const HGCalParameters* hp, const std::string& name)
    : hgpar_(hp), sqrt3_(std::sqrt(3.0)) {
  mode_ = hgpar_->mode_;
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull) ||
      (mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    rmax_ = (HGCalParameters::k_ScaleFromDDD * (hgpar_->waferR_) * std::cos(30._deg));
    hexside_ = 2.0 * rmax_ * tan30deg_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "rmax_ " << rmax_ << ":" << hexside_ << " CellSize "
                                  << 0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[0] << ":"
                                  << 0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[1];
#endif
  }
  // init maps and constants
  modHalf_ = 0;
  maxWafersPerLayer_ = 0;
  for (int simreco = 0; simreco < 2; ++simreco) {
    tot_layers_[simreco] = layersInit((bool)simreco);
    max_modules_layer_[simreco].resize(tot_layers_[simreco] + 1);
    for (unsigned int layer = 1; layer <= tot_layers_[simreco]; ++layer) {
      max_modules_layer_[simreco][layer] = modulesInit(layer, (bool)simreco);
      if (simreco == 1) {
        modHalf_ += max_modules_layer_[simreco][layer];
        maxWafersPerLayer_ = std::max(maxWafersPerLayer_, max_modules_layer_[simreco][layer]);
#ifdef EDM_ML_DEBUG
        edm::LogVerbatim("HGCalGeom") << "Layer " << layer << " with " << max_modules_layer_[simreco][layer] << ":"
                                      << modHalf_ << " modules";
#endif
      }
    }
  }
  tot_wafers_ = wafers();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants initialized for " << name << " with " << layers(false) << ":"
                                << layers(true) << " layers, " << wafers() << ":" << 2 * modHalf_
                                << " wafers with maximum " << maxWafersPerLayer_ << " per layer and "
                                << "maximum of " << maxCells(false) << ":" << maxCells(true) << " cells";
#endif
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull) ||
      (mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int wminT(9999999), wmaxT(-9999999), kount1(0), kount2(0);
    for (unsigned int i = 0; i < getTrFormN(); ++i) {
      int lay0 = getTrForm(i).lay;
      int wmin(9999999), wmax(-9999999), kount(0);
      for (int wafer = 0; wafer < sectors(); ++wafer) {
        bool waferIn = waferInLayer(wafer, lay0, true);
        if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
          int kndx = HGCalWaferIndex::waferIndex(lay0,
                                                 HGCalWaferIndex::waferU(hgpar_->waferCopy_[wafer]),
                                                 HGCalWaferIndex::waferV(hgpar_->waferCopy_[wafer]));
          waferIn_[kndx] = waferIn;
        }
        if (waferIn) {
          int waferU = (((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull))
                            ? wafer
                            : HGCalWaferIndex::waferU(hgpar_->waferCopy_[wafer]));
          if (waferU < wmin)
            wmin = waferU;
          if (waferU > wmax)
            wmax = waferU;
          ++kount;
        }
      }
      if (wminT > wmin)
        wminT = wmin;
      if (wmaxT < wmax)
        wmaxT = wmax;
      if (kount1 < kount)
        kount1 = kount;
      kount2 += kount;
#ifdef EDM_ML_DEBUG
      int lay1 = getIndex(lay0, true).first;
      edm::LogVerbatim("HGCalGeom") << "Index " << i << " Layer " << lay0 << ":" << lay1 << " Wafer " << wmin << ":"
                                    << wmax << ":" << kount;
#endif
      HGCWaferParam a1{{wmin, wmax, kount}};
      waferLayer_[lay0] = a1;
    }
    waferMax_ = std::array<int, 4>{{wminT, wmaxT, kount1, kount2}};
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Overall wafer statistics: " << wminT << ":" << wmaxT << ":" << kount1 << ":"
                                  << kount2;
#endif
  }
}

HGCalDDDConstants::~HGCalDDDConstants() {}

std::pair<int, int> HGCalDDDConstants::assignCell(float x, float y, int lay, int subSec, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return std::make_pair(-1, -1);
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    float xx = (reco) ? x : HGCalParameters::k_ScaleFromDDD * x;
    float yy = (reco) ? y : HGCalParameters::k_ScaleFromDDD * y;

    // First the wafer
    int wafer = cellHex(xx, yy, rmax_, hgpar_->waferPosX_, hgpar_->waferPosY_);
    if (wafer < 0 || wafer >= (int)(hgpar_->waferTypeT_.size())) {
      edm::LogWarning("HGCalGeom") << "Wafer no. out of bound for " << wafer << ":" << (hgpar_->waferTypeT_).size()
                                   << ":" << (hgpar_->waferPosX_).size() << ":" << (hgpar_->waferPosY_).size()
                                   << " ***** ERROR *****";
      return std::make_pair(-1, -1);
    } else {
      // Now the cell
      xx -= hgpar_->waferPosX_[wafer];
      yy -= hgpar_->waferPosY_[wafer];
      if (hgpar_->waferTypeT_[wafer] == 1)
        return std::make_pair(wafer,
                              cellHex(xx,
                                      yy,
                                      0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[0],
                                      hgpar_->cellFineX_,
                                      hgpar_->cellFineY_));
      else
        return std::make_pair(wafer,
                              cellHex(xx,
                                      yy,
                                      0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[1],
                                      hgpar_->cellCoarseX_,
                                      hgpar_->cellCoarseY_));
    }
  } else {
    return std::make_pair(-1, -1);
  }
}

std::array<int, 5> HGCalDDDConstants::assignCellHex(float x, float y, int lay, bool reco) const {
  int waferU(0), waferV(0), waferType(-1), cellU(0), cellV(0);
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    double xx = (reco) ? HGCalParameters::k_ScaleToDDD * x : x;
    double yy = (reco) ? HGCalParameters::k_ScaleToDDD * y : y;
    double wt(1.0);
    waferFromPosition(xx, yy, lay, waferU, waferV, cellU, cellV, waferType, wt);
  }
  return std::array<int, 5>{{waferU, waferV, waferType, cellU, cellV}};
}

std::array<int, 3> HGCalDDDConstants::assignCellTrap(float x, float y, float z, int layer, bool reco) const {
  int irad(-1), iphi(-1), type(-1);
  const auto& indx = getIndex(layer, reco);
  if (indx.first < 0)
    return std::array<int, 3>{{irad, iphi, type}};
  double xx = (z > 0) ? x : -x;
  double r = (reco ? std::sqrt(x * x + y * y) : HGCalParameters::k_ScaleFromDDD * std::sqrt(x * x + y * y));
  double phi = (r == 0. ? 0. : std::atan2(y, xx));
  if (phi < 0)
    phi += (2.0 * M_PI);
  type = hgpar_->scintType(layer);
  auto ir = std::lower_bound(hgpar_->radiusLayer_[type].begin(), hgpar_->radiusLayer_[type].end(), r);
  irad = (int)(ir - hgpar_->radiusLayer_[type].begin());
  irad = std::clamp(irad, hgpar_->iradMinBH_[indx.first], hgpar_->iradMaxBH_[indx.first]);
  iphi = 1 + (int)(phi / indx.second);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "assignCellTrap Input " << x << ":" << y << ":" << z << ":" << layer << ":" << reco
                                << " x|r " << xx << ":" << r << " phi " << phi << " o/p " << irad << ":" << iphi << ":"
                                << type;
#endif
  return std::array<int, 3>{{irad, iphi, type}};
}

std::pair<double, double> HGCalDDDConstants::cellEtaPhiTrap(int type, int irad) const {
  double dr(0), df(0);
  if (mode_ == HGCalGeometryMode::Trapezoid) {
    double r = 0.5 * ((hgpar_->radiusLayer_[type][irad - 1] + hgpar_->radiusLayer_[type][irad]));
    dr = (hgpar_->radiusLayer_[type][irad] - hgpar_->radiusLayer_[type][irad - 1]);
    df = r * hgpar_->cellSize_[type];
  }
  return std::make_pair(dr, df);
}

bool HGCalDDDConstants::cellInLayer(int waferU, int waferV, int cellU, int cellV, int lay, bool reco) const {
  const auto& indx = getIndex(lay, true);
  if (indx.first >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full) ||
        (mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
      const auto& xy = (((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full))
                            ? locateCell(lay, waferU, waferV, cellU, cellV, reco, true, false)
                            : locateCell(cellU, lay, waferU, reco));
      double rpos = sqrt(xy.first * xy.first + xy.second * xy.second);
      return ((rpos >= hgpar_->rMinLayHex_[indx.first]) && (rpos <= hgpar_->rMaxLayHex_[indx.first]));
    } else {
      return true;
    }
  } else {
    return false;
  }
}

double HGCalDDDConstants::cellThickness(int layer, int waferU, int waferV) const {
  double thick(-1);
  int type = waferType(layer, waferU, waferV);
  if (type >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
      thick = 10000.0 * hgpar_->cellThickness_[type];  // cm to micron
    } else if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
      thick = 100.0 * (type + 1);  // type = 1,2,3 for 100,200,300 micron
    }
  }
  return thick;
}

double HGCalDDDConstants::cellSizeHex(int type) const {
  int indx =
      (((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) ? ((type >= 1) ? 1 : 0)
                                                                                              : ((type == 1) ? 1 : 0));
  double cell =
      ((mode_ == HGCalGeometryMode::Trapezoid) ? 0.5 * hgpar_->cellSize_[indx]
                                               : 0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[indx]);
  return cell;
}

HGCalDDDConstants::CellType HGCalDDDConstants::cellType(int type, int cellU, int cellV) const {
  // type=0: in the middle; 1..6: the edges clocwise from bottom left;
  //     =11..16: the corners clockwise from bottom
  int N = (type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  if (cellU == 0) {
    if (cellV == 0)
      return HGCalDDDConstants::CellType::BottomLeftCorner;
    else if (cellV - cellU == N - 1)
      return HGCalDDDConstants::CellType::BottomCorner;
    else
      return HGCalDDDConstants::CellType::BottomLeftEdge;
  } else if (cellV == 0) {
    if (cellU - cellV == N)
      return HGCalDDDConstants::CellType::TopLeftCorner;
    else
      return HGCalDDDConstants::CellType::LeftEdge;
  } else if (cellU - cellV == N) {
    if (cellU == 2 * N - 1)
      return HGCalDDDConstants::CellType::TopCorner;
    else
      return HGCalDDDConstants::CellType::TopLeftEdge;
  } else if (cellU == 2 * N - 1) {
    if (cellV == 2 * N - 1)
      return HGCalDDDConstants::CellType::TopRightCorner;
    else
      return HGCalDDDConstants::CellType::TopRightEdge;
  } else if (cellV == 2 * N - 1) {
    if (cellV - cellU == N - 1)
      return HGCalDDDConstants::CellType::BottomRightCorner;
    else
      return HGCalDDDConstants::CellType::RightEdge;
  } else if (cellV - cellU == N - 1) {
    return HGCalDDDConstants::CellType::BottomRightEdge;
  } else if ((cellU > 2 * N - 1) || (cellV > 2 * N - 1) || (cellV >= (cellU + N)) || (cellU > (cellV + N))) {
    return HGCalDDDConstants::CellType::UndefinedType;
  } else {
    return HGCalDDDConstants::CellType::CentralType;
  }
}

double HGCalDDDConstants::distFromEdgeHex(double x, double y, double z) const {
  // Assming the point is within a hexagonal plane of the wafer, calculate
  // the shortest distance from the edge
  if (z < 0)
    x = -x;
  double dist(0);
  // Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalParameters::k_ScaleFromDDD * x;
  double yy = HGCalParameters::k_ScaleFromDDD * y;
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int ll = layerIndex(getLayer(z, false), false);
    xx -= hgpar_->xLayerHex_[ll];
    yy -= hgpar_->yLayerHex_[ll];
  }
  int sizew = (int)(hgpar_->waferPosX_.size());
  int wafer = sizew;
  // Transform to the local coordinate frame of the wafer first
  for (int k = 0; k < sizew; ++k) {
    double dx = std::abs(xx - hgpar_->waferPosX_[k]);
    double dy = std::abs(yy - hgpar_->waferPosY_[k]);
    if ((dx <= rmax_) && (dy <= hexside_) && ((dy <= 0.5 * hexside_) || (dx * tan30deg_ <= (hexside_ - dy)))) {
      wafer = k;
      xx -= hgpar_->waferPosX_[k];
      yy -= hgpar_->waferPosY_[k];
      break;
    }
  }
  // Look at only one quarter (both x,y are positive)
  if (wafer < sizew) {
    if (std::abs(yy) < 0.5 * hexside_) {
      dist = rmax_ - std::abs(xx);
    } else {
      dist = 0.5 * ((rmax_ - std::abs(xx)) - sqrt3_ * (std::abs(yy) - 0.5 * hexside_));
    }
  } else {
    dist = 0;
  }
  dist *= HGCalParameters::k_ScaleToDDD;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DistFromEdgeHex: Local " << xx << ":" << yy << " wafer " << wafer << " flag "
                                << (wafer < sizew) << " Distance " << rmax_ << ":" << (rmax_ - std::abs(xx)) << ":"
                                << (std::abs(yy) - 0.5 * hexside_) << ":" << 0.5 * hexside_ << ":" << dist;
#endif
  return dist;
}

double HGCalDDDConstants::distFromEdgeTrap(double x, double y, double z) const {
  // Assming the point is within the eta-phi plane of the scintillator tile,
  // calculate the shortest distance from the edge
  int lay = getLayer(z, false);
  double xx = (z < 0) ? -x : x;
  int indx = layerIndex(lay, false);
  double r = HGCalParameters::k_ScaleFromDDD * std::sqrt(x * x + y * y);
  double phi = (r == 0. ? 0. : std::atan2(y, xx));
  if (phi < 0)
    phi += (2.0 * M_PI);
  int type = hgpar_->scintType(lay);
  double cell = hgpar_->scintCellSize(lay);
  // Compare with the center of the tile find distances along R and also phi
  // Take the smaller value
  auto ir = std::lower_bound(hgpar_->radiusLayer_[type].begin(), hgpar_->radiusLayer_[type].end(), r);
  int irad = (int)(ir - hgpar_->radiusLayer_[type].begin());
  irad = std::clamp(irad, hgpar_->iradMinBH_[indx], hgpar_->iradMaxBH_[indx]);
  int iphi = 1 + (int)(phi / cell);
  double dphi = std::max(0.0, (0.5 * cell - std::abs(phi - (iphi - 0.5) * cell)));
  double dist = std::min((r - hgpar_->radiusLayer_[type][irad - 1]), (hgpar_->radiusLayer_[type][irad] - r));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DistFromEdgeTrap: Global " << x << ":" << y << ":" << z << " Layer " << lay
                                << " Index " << indx << ":" << type << " xx " << xx << " R " << r << ":" << irad << ":"
                                << hgpar_->radiusLayer_[type][irad - 1] << ":" << hgpar_->radiusLayer_[type][irad]
                                << " Phi " << phi << ":" << iphi << ":" << (iphi - 0.5) * cell << " cell " << cell
                                << " Dphi " << dphi << " Dist " << dist << ":" << r * dphi;
#endif
  return HGCalParameters::k_ScaleToDDD * std::min(r * dphi, dist);
}

int HGCalDDDConstants::getLayer(double z, bool reco) const {
  // Get the layer # from the gloabl z coordinate
  unsigned int k = 0;
  double zz = (reco ? std::abs(z) : HGCalParameters::k_ScaleFromDDD * std::abs(z));
  const auto& zLayerHex = hgpar_->zLayerHex_;
  auto itr = std::find_if(zLayerHex.begin() + 1, zLayerHex.end(), [&k, &zz, &zLayerHex](double zLayer) {
    ++k;
    return zz < 0.5 * (zLayerHex[k - 1] + zLayerHex[k]);
  });
  int lay = (itr == zLayerHex.end()) ? static_cast<int>(zLayerHex.size()) : k;
  if (((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) && reco) {
    int indx = layerIndex(lay, false);
    if (indx >= 0)
      lay = hgpar_->layerGroupO_[indx];
  } else {
    lay += (hgpar_->firstLayer_ - 1);
  }
  return lay;
}

HGCalParameters::hgtrap HGCalDDDConstants::getModule(unsigned int indx, bool hexType, bool reco) const {
  HGCalParameters::hgtrap mytr;
  if (hexType) {
    if (indx >= hgpar_->waferTypeL_.size())
      edm::LogWarning("HGCalGeom") << "Wafer no. out bound for index " << indx << ":" << (hgpar_->waferTypeL_).size()
                                   << ":" << (hgpar_->waferPosX_).size() << ":" << (hgpar_->waferPosY_).size()
                                   << " ***** ERROR *****";
    unsigned int type =
        ((indx < hgpar_->waferTypeL_.size()) ? hgpar_->waferTypeL_[indx] - 1 : HGCSiliconDetId::HGCalCoarseThick);
    mytr = hgpar_->getModule(type, reco);
  } else {
    mytr = hgpar_->getModule(indx, reco);
  }
  return mytr;
}

std::vector<HGCalParameters::hgtrap> HGCalDDDConstants::getModules() const {
  std::vector<HGCalParameters::hgtrap> mytrs;
  for (unsigned int k = 0; k < hgpar_->moduleLayR_.size(); ++k)
    mytrs.emplace_back(hgpar_->getModule(k, true));
  return mytrs;
}

int HGCalDDDConstants::getPhiBins(int lay) const {
  return ((mode_ == HGCalGeometryMode::Trapezoid) ? hgpar_->scintCells(lay) : 0);
}

std::pair<int, int> HGCalDDDConstants::getREtaRange(int lay) const {
  int irmin(0), irmax(0);
  if (mode_ == HGCalGeometryMode::Trapezoid) {
    int indx = layerIndex(lay, false);
    if ((indx >= 0) && (indx < static_cast<int>(hgpar_->iradMinBH_.size()))) {
      irmin = hgpar_->iradMinBH_[indx];
      irmax = hgpar_->iradMaxBH_[indx];
    }
  }
  return std::make_pair(irmin, irmax);
}

std::vector<HGCalParameters::hgtrform> HGCalDDDConstants::getTrForms() const {
  std::vector<HGCalParameters::hgtrform> mytrs;
  for (unsigned int k = 0; k < hgpar_->trformIndex_.size(); ++k)
    mytrs.emplace_back(hgpar_->getTrForm(k));
  return mytrs;
}

int HGCalDDDConstants::getTypeTrap(int layer) const {
  // Get the module type for scinitllator
  if (mode_ == HGCalGeometryMode::Trapezoid) {
    return hgpar_->scintType(layer);
  } else {
    return -1;
  }
}

int HGCalDDDConstants::getTypeHex(int layer, int waferU, int waferV) const {
  // Get the module type for a silicon wafer
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer, waferU, waferV));
    return ((itr == hgpar_->typesInLayers_.end() ? 2 : hgpar_->waferTypeL_[itr->second]));
  } else {
    return -1;
  }
}

bool HGCalDDDConstants::isHalfCell(int waferType, int cell) const {
  if (waferType < 1 || cell < 0)
    return false;
  return waferType == 2 ? hgpar_->cellCoarseHalf_[cell] : hgpar_->cellFineHalf_[cell];
}

bool HGCalDDDConstants::isValidHex(int lay, int mod, int cell, bool reco) const {
  // Check validity for a layer|wafer|cell of pre-TDR version
  bool result(false), resultMod(false);
  int cellmax(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    int32_t copyNumber = hgpar_->waferCopy_[mod];
    result = ((lay > 0 && lay <= (int)(layers(reco))));
    if (result) {
      const int32_t lay_idx = reco ? (lay - 1) * 3 + 1 : lay;
      const auto& the_modules = hgpar_->copiesInLayers_[lay_idx];
      auto moditr = the_modules.find(copyNumber);
      result = resultMod = (moditr != the_modules.end());
#ifdef EDM_ML_DEBUG
      if (!result)
        edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay << ":" << lay_idx << " Copy " << copyNumber
                                      << ":" << mod << " Flag " << result;
#endif
      if (result) {
        if (moditr->second >= 0) {
          if (mod >= (int)(hgpar_->waferTypeT_.size()))
            edm::LogWarning("HGCalGeom") << "Module no. out of bound for " << mod << " to be compared with "
                                         << (hgpar_->waferTypeT_).size() << " ***** ERROR *****";
          cellmax = ((hgpar_->waferTypeT_[mod] - 1 == HGCSiliconDetId::HGCalFine) ? (int)(hgpar_->cellFineX_.size())
                                                                                  : (int)(hgpar_->cellCoarseX_.size()));
          result = (cell >= 0 && cell <= cellmax);
        } else {
          result = isValidCell(lay_idx, mod, cell);
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  if (!result)
    edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants: Layer " << lay << ":"
                                  << (lay > 0 && (lay <= (int)(layers(reco)))) << " Module " << mod << ":" << resultMod
                                  << " Cell " << cell << ":" << cellmax << ":" << (cell >= 0 && cell <= cellmax) << ":"
                                  << maxCells(reco);
#endif
  return result;
}

bool HGCalDDDConstants::isValidHex8(int layer, int modU, int modV, int cellU, int cellV) const {
  // Check validity for a layer|wafer|cell of post-TDR version
  int indx = HGCalWaferIndex::waferIndex(layer, modU, modV);
  auto itr = hgpar_->typesInLayers_.find(indx);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:WaferType " << layer << ":" << modU << ":" << modV
                                << ":" << indx << " Test " << (itr != hgpar_->typesInLayers_.end());
#endif
  if (itr == hgpar_->typesInLayers_.end())
    return false;
  auto jtr = waferIn_.find(indx);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:WaferIn " << jtr->first << ":" << jtr->second;
#endif
  if (!(jtr->second))
    return false;

  int N = ((hgpar_->waferTypeL_[itr->second] == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants::isValidHex8:Cell " << cellU << ":" << cellV << ":" << N
                                << " Tests " << (cellU >= 0) << ":" << (cellU < 2 * N) << ":" << (cellV >= 0) << ":"
                                << (cellV < 2 * N) << ":" << ((cellV - cellU) < N) << ":" << ((cellU - cellV) <= N);
#endif
  if ((cellU < 0) || (cellU >= 2 * N) || (cellV < 0) || (cellV >= 2 * N))
    return false;
  if (((cellV - cellU) >= N) || ((cellU - cellV) > N))
    return false;

  auto ktr = hgpar_->waferTypes_.find(indx);
  if (ktr != hgpar_->waferTypes_.end())
    return false;

  //  edm::LogVerbatim("HGCalGeom") << "Corners " << (ktr->second).first << ":" << waferVirtual(layer,modU,modV);
  int type = ((itr == hgpar_->typesInLayers_.end()) ? 2 : hgpar_->waferTypeL_[itr->second]);
  return isValidCell8(layer, modU, modV, cellU, cellV, type);
}

bool HGCalDDDConstants::isValidTrap(int layer, int irad, int iphi) const {
  // Check validity for a layer|eta|phi of scintillator
  const auto& indx = getIndex(layer, true);
  if (indx.first < 0)
    return false;
  return ((irad >= hgpar_->iradMinBH_[indx.first]) && (irad <= hgpar_->iradMaxBH_[indx.first]) && (iphi > 0) &&
          (iphi <= hgpar_->scintCells(layer)));
}

int HGCalDDDConstants::lastLayer(bool reco) const { return (hgpar_->firstLayer_ + tot_layers_[(int)reco] - 1); }

unsigned int HGCalDDDConstants::layers(bool reco) const { return tot_layers_[(int)reco]; }

int HGCalDDDConstants::layerIndex(int lay, bool reco) const {
  int ll = lay - hgpar_->firstLayer_;
  if (ll < 0 || ll >= (int)(hgpar_->layerIndex_.size()))
    return -1;
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    if (reco && ll >= (int)(hgpar_->depthIndex_.size()))
      return -1;
    return (reco ? hgpar_->depthLayerF_[ll] : hgpar_->layerIndex_[ll]);
  } else {
    return (hgpar_->layerIndex_[ll]);
  }
}

unsigned int HGCalDDDConstants::layersInit(bool reco) const {
  return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());
}

std::pair<float, float> HGCalDDDConstants::locateCell(int cell, int lay, int type, bool reco) const {
  // type refers to wafer # for hexagon cell
  float x(999999.), y(999999.);
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0)
    return std::make_pair(x, y);
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    x = hgpar_->waferPosX_[type];
    y = hgpar_->waferPosY_[type];
#ifdef EDM_ML_DEBUG
    float x0(x), y0(y);
#endif
    if (hgpar_->waferTypeT_[type] - 1 == HGCSiliconDetId::HGCalFine) {
      x += hgpar_->cellFineX_[cell];
      y += hgpar_->cellFineY_[cell];
    } else {
      x += hgpar_->cellCoarseX_[cell];
      y += hgpar_->cellCoarseY_[cell];
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "LocateCell (Wafer) " << x0 << ":" << y0 << " Final " << x << ":" << y;
#endif
    if (!reco) {
      x *= HGCalParameters::k_ScaleToDDD;
      y *= HGCalParameters::k_ScaleToDDD;
    }
  }
  return std::make_pair(x, y);
}

std::pair<float, float> HGCalDDDConstants::locateCell(
    int lay, int waferU, int waferV, int cellU, int cellV, bool reco, bool all, bool debug) const {
  float x(0), y(0);
  int indx = HGCalWaferIndex::waferIndex(lay, waferU, waferV);
  auto itr = hgpar_->typesInLayers_.find(indx);
  int type = ((itr == hgpar_->typesInLayers_.end()) ? 2 : hgpar_->waferTypeL_[itr->second]);
#ifdef EDM_ML_DEBUG
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "LocateCell " << lay << ":" << waferU << ":" << waferV << ":" << indx << ":"
                                  << (itr == hgpar_->typesInLayers_.end()) << ":" << type;
#endif
  int kndx = cellV * 100 + cellU;
  if (type == 0) {
    auto ktr = hgpar_->cellFineIndex_.find(kndx);
    if (ktr != hgpar_->cellFineIndex_.end()) {
      x = hgpar_->cellFineX_[ktr->second];
      y = hgpar_->cellFineY_[ktr->second];
    }
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "Fine " << cellU << ":" << cellV << ":" << kndx << ":" << x << ":" << y << ":"
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
      edm::LogVerbatim("HGCalGeom") << "Coarse " << cellU << ":" << cellV << ":" << kndx << ":" << x << ":" << y << ":"
                                    << (ktr != hgpar_->cellCoarseIndex_.end());
#endif
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  if (all) {
    const auto& xy = waferPosition(lay, waferU, waferV, reco, debug);
    x += xy.first;
    y += xy.second;
#ifdef EDM_ML_DEBUG
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "With wafer " << x << ":" << y << ":" << xy.first << ":" << xy.second;
#endif
  }
  return std::make_pair(x, y);
}

std::pair<float, float> HGCalDDDConstants::locateCellHex(int cell, int wafer, bool reco) const {
  float x(0), y(0);
  if (hgpar_->waferTypeT_[wafer] - 1 == HGCSiliconDetId::HGCalFine) {
    x = hgpar_->cellFineX_[cell];
    y = hgpar_->cellFineY_[cell];
  } else {
    x = hgpar_->cellCoarseX_[cell];
    y = hgpar_->cellCoarseY_[cell];
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(x, y);
}

std::pair<float, float> HGCalDDDConstants::locateCellTrap(int lay, int irad, int iphi, bool reco) const {
  float x(0), y(0);
  const auto& indx = getIndex(lay, reco);
  if (indx.first >= 0) {
    int ir = std::abs(irad);
    int type = hgpar_->scintType(lay);
    double phi = (iphi - 0.5) * indx.second;
    double z = hgpar_->zLayerHex_[indx.first];
    double r = 0.5 * (hgpar_->radiusLayer_[type][ir - 1] + hgpar_->radiusLayer_[type][ir]);
    std::pair<double, double> range = rangeR(z, true);
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "locateCellTrap:: Input " << lay << ":" << irad << ":" << iphi << ":" << reco
                                  << " IR " << ir << ":" << hgpar_->iradMinBH_[indx.first] << ":"
                                  << hgpar_->iradMaxBH_[indx.first] << " Type " << type << " Z " << indx.first << ":"
                                  << z << " phi " << phi << " R " << r << ":" << range.first << ":" << range.second;
#endif
    r = std::max(range.first, std::min(r, range.second));
    x = r * std::cos(phi);
    y = r * std::sin(phi);
    if (irad < 0)
      x = -x;
  }
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(x, y);
}

bool HGCalDDDConstants::maskCell(const DetId& detId, int corners) const {
  bool mask(false);
  if (corners > 2 && corners <= (int)(HGCalParameters::k_CornerSize)) {
    if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
      int N(0), layer(0), waferU(0), waferV(0), u(0), v(0);
      if (detId.det() == DetId::Forward) {
        HFNoseDetId id(detId);
        N = getUVMax(id.type());
        layer = id.layer();
        waferU = id.waferU();
        waferV = id.waferV();
        u = id.cellU();
        v = id.cellV();
      } else {
        HGCSiliconDetId id(detId);
        N = getUVMax(id.type());
        layer = id.layer();
        waferU = id.waferU();
        waferV = id.waferV();
        u = id.cellU();
        v = id.cellV();
      }
      int wl = HGCalWaferIndex::waferIndex(layer, waferU, waferV);
      auto itr = hgpar_->waferTypes_.find(wl);
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << "MaskCell: Layer " << layer << " Wafer " << waferU << ":" << waferV << " Index "
                                    << wl << ":" << (itr != hgpar_->waferTypes_.end());
#endif
      if (itr != hgpar_->waferTypes_.end()) {
        if ((itr->second).second <= HGCalWaferMask::k_OffsetRotation)
          mask = HGCalWaferMask::maskCell(u, v, N, (itr->second).first, (itr->second).second, corners);
        else
          mask = !(HGCalWaferMask::goodCell(
              u, v, N, (itr->second).first, ((itr->second).second - HGCalWaferMask::k_OffsetRotation)));
      }
    }
  }
  return mask;
}

int HGCalDDDConstants::maxCells(bool reco) const {
  int cells(0);
  for (unsigned int i = 0; i < layers(reco); ++i) {
    int lay = reco ? hgpar_->depth_[i] : hgpar_->layer_[i];
    if (cells < maxCells(lay, reco))
      cells = maxCells(lay, reco);
  }
  return cells;
}

int HGCalDDDConstants::maxCells(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return 0;
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    unsigned int cells(0);
    for (unsigned int k = 0; k < hgpar_->waferTypeT_.size(); ++k) {
      if (waferInLayerTest(k, index.first, hgpar_->defineFull_)) {
        unsigned int cell = (hgpar_->waferTypeT_[k] - 1 == HGCSiliconDetId::HGCalFine) ? (hgpar_->cellFineX_.size())
                                                                                       : (hgpar_->cellCoarseX_.size());
        if (cell > cells)
          cells = cell;
      }
    }
    return (int)(cells);
  } else if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int cells(0);
    for (unsigned int k = 0; k < hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayerTest(k, index.first, hgpar_->defineFull_)) {
        auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(
            lay, HGCalWaferIndex::waferU(hgpar_->waferCopy_[k]), HGCalWaferIndex::waferV(hgpar_->waferCopy_[k])));
        int type = ((itr == hgpar_->typesInLayers_.end()) ? HGCSiliconDetId::HGCalCoarseThick
                                                          : hgpar_->waferTypeL_[itr->second]);
        int N = (type == HGCSiliconDetId::HGCalFine) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
        cells = std::max(cells, 3 * N * N);
      }
    }
    return cells;
  } else if (mode_ == HGCalGeometryMode::Trapezoid) {
    return hgpar_->scintCells(index.first + hgpar_->firstLayer_);
  } else {
    return 0;
  }
}

int HGCalDDDConstants::maxRows(int lay, bool reco) const {
  int kymax(0);
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0)
    return kymax;
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    for (unsigned int k = 0; k < hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayerTest(k, i, hgpar_->defineFull_)) {
        int ky = ((hgpar_->waferCopy_[k]) / 100) % 100;
        if (ky > kymax)
          kymax = ky;
      }
    }
  } else if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    kymax = 1 + 2 * hgpar_->waferUVMaxLayer_[index.first];
  }
  return kymax;
}

int HGCalDDDConstants::modifyUV(int uv, int type1, int type2) const {
  // Modify u/v for transition of type1 to type2
  return (((type1 == type2) || (type1 * type2 != 0)) ? uv : ((type1 == 0) ? (2 * uv + 1) / 3 : (3 * uv) / 2));
}

int HGCalDDDConstants::modules(int lay, bool reco) const {
  if (getIndex(lay, reco).first < 0)
    return 0;
  else
    return max_modules_layer_[(int)reco][lay];
}

int HGCalDDDConstants::modulesInit(int lay, bool reco) const {
  int nmod(0);
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return nmod;
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    for (unsigned int k = 0; k < hgpar_->waferPosX_.size(); ++k) {
      if (waferInLayerTest(k, index.first, hgpar_->defineFull_))
        ++nmod;
    }
  } else {
    nmod = 1 + hgpar_->lastModule_[index.first] - hgpar_->firstModule_[index.first];
  }
  return nmod;
}

double HGCalDDDConstants::mouseBite(bool reco) const {
  return (reco ? hgpar_->mouseBite_ : HGCalParameters::k_ScaleToDDD * hgpar_->mouseBite_);
}

int HGCalDDDConstants::numberCells(bool reco) const {
  int cells(0);
  unsigned int nlayer = (reco) ? hgpar_->depth_.size() : hgpar_->layer_.size();
  for (unsigned k = 0; k < nlayer; ++k) {
    std::vector<int> ncells = numberCells(((reco) ? hgpar_->depth_[k] : hgpar_->layer_[k]), reco);
    cells = std::accumulate(ncells.begin(), ncells.end(), cells);
  }
  return cells;
}

std::vector<int> HGCalDDDConstants::numberCells(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  std::vector<int> ncell;
  if (i >= 0) {
    if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
      for (unsigned int k = 0; k < hgpar_->waferTypeT_.size(); ++k) {
        if (waferInLayerTest(k, i, hgpar_->defineFull_)) {
          unsigned int cell = (hgpar_->waferTypeT_[k] - 1 == HGCSiliconDetId::HGCalFine)
                                  ? (hgpar_->cellFineX_.size())
                                  : (hgpar_->cellCoarseX_.size());
          ncell.emplace_back((int)(cell));
        }
      }
    } else if (mode_ == HGCalGeometryMode::Trapezoid) {
      int nphi = hgpar_->scintCells(lay);
      for (int k = hgpar_->firstModule_[i]; k <= hgpar_->lastModule_[i]; ++k)
        ncell.emplace_back(nphi);
    } else {
      for (unsigned int k = 0; k < hgpar_->waferCopy_.size(); ++k) {
        if (waferInLayerTest(k, index.first, hgpar_->defineFull_)) {
          int cell = numberCellsHexagon(lay,
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
    if (hgpar_->waferTypeT_[wafer] - 1 == HGCSiliconDetId::HGCalFine)
      return (int)(hgpar_->cellFineX_.size());
    else
      return (int)(hgpar_->cellCoarseX_.size());
  } else {
    return 0;
  }
}

int HGCalDDDConstants::numberCellsHexagon(int lay, int waferU, int waferV, bool flag) const {
  auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(lay, waferU, waferV));
  int type =
      ((itr == hgpar_->typesInLayers_.end()) ? HGCSiliconDetId::HGCalCoarseThick : hgpar_->waferTypeL_[itr->second]);
  int N = (type == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  if (flag)
    return (3 * N * N);
  else
    return N;
}

std::pair<double, double> HGCalDDDConstants::rangeR(double z, bool reco) const {
  double rmin(0), rmax(0), zz(0);
  if (hgpar_->detectorType_ > 0) {
    zz = (reco ? std::abs(z) : HGCalParameters::k_ScaleFromDDD * std::abs(z));
    if (hgpar_->detectorType_ <= 2) {
      rmin = HGCalGeomTools::radius(zz, hgpar_->zFrontMin_, hgpar_->rMinFront_, hgpar_->slopeMin_);
    } else {
      rmin = HGCalGeomTools::radius(
          zz, hgpar_->firstLayer_, hgpar_->firstMixedLayer_, hgpar_->zLayerHex_, hgpar_->radiusMixBoundary_);
    }
    if ((hgpar_->detectorType_ == 2) && (zz >= hgpar_->zLayerHex_[hgpar_->firstMixedLayer_ - 1])) {
      rmax = HGCalGeomTools::radius(
          zz, hgpar_->firstLayer_, hgpar_->firstMixedLayer_, hgpar_->zLayerHex_, hgpar_->radiusMixBoundary_);
    } else {
      rmax = HGCalGeomTools::radius(zz, hgpar_->zFrontTop_, hgpar_->rMaxFront_, hgpar_->slopeTop_);
    }
  }
  if (!reco) {
    rmin *= HGCalParameters::k_ScaleToDDD;
    rmax *= HGCalParameters::k_ScaleToDDD;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants:rangeR: " << z << ":" << zz << " R " << rmin << ":" << rmax;
#endif
  return std::make_pair(rmin, rmax);
}

std::pair<double, double> HGCalDDDConstants::rangeZ(bool reco) const {
  double zmin = (hgpar_->zLayerHex_[0] - hgpar_->waferThick_);
  double zmax = (hgpar_->zLayerHex_[hgpar_->zLayerHex_.size() - 1] + hgpar_->waferThick_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalDDDConstants:rangeZ: " << zmin << ":" << zmax << ":" << hgpar_->waferThick_;
#endif
  if (!reco) {
    zmin *= HGCalParameters::k_ScaleToDDD;
    zmax *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(zmin, zmax);
}

std::pair<int, int> HGCalDDDConstants::rowColumnWafer(int wafer) const {
  int row(0), col(0);
  if (wafer < (int)(hgpar_->waferCopy_.size())) {
    int copy = hgpar_->waferCopy_[wafer];
    col = copy % 100;
    if ((copy / 10000) % 10 != 0)
      col = -col;
    row = (copy / 100) % 100;
    if ((copy / 100000) % 10 != 0)
      row = -row;
  }
  return std::make_pair(row, col);
}

std::pair<int, int> HGCalDDDConstants::simToReco(int cell, int lay, int mod, bool half) const {
  if ((mode_ != HGCalGeometryMode::Hexagon) && (mode_ != HGCalGeometryMode::HexagonFull)) {
    return std::make_pair(cell, lay);
  } else {
    const auto& index = getIndex(lay, false);
    int i = index.first;
    if (i < 0) {
      edm::LogWarning("HGCalGeom") << "Wrong Layer # " << lay << " not in the list ***** ERROR *****";
      return std::make_pair(-1, -1);
    }
    if (mod >= (int)(hgpar_->waferTypeL_).size()) {
      edm::LogWarning("HGCalGeom") << "Invalid Wafer # " << mod << "should be < " << (hgpar_->waferTypeL_).size()
                                   << " ***** ERROR *****";
      return std::make_pair(-1, -1);
    }
    int depth(-1);
    int kx = cell;
    int type = hgpar_->waferTypeL_[mod];
    if (type == 1) {
      depth = hgpar_->layerGroup_[i];
    } else if (type == 2) {
      depth = hgpar_->layerGroupM_[i];
    } else {
      depth = hgpar_->layerGroupO_[i];
    }
    return std::make_pair(kx, depth);
  }
}

int HGCalDDDConstants::waferFromCopy(int copy) const {
  const int ncopies = hgpar_->waferCopy_.size();
  int wafer(ncopies);
  bool result(false);
  for (int k = 0; k < ncopies; ++k) {
    if (copy == hgpar_->waferCopy_[k]) {
      wafer = k;
      result = true;
      break;
    }
  }
  if (!result) {
    wafer = -1;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Cannot find " << copy << " in a list of " << ncopies << " members";
    for (int k = 0; k < ncopies; ++k)
      edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << hgpar_->waferCopy_[k];
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "WaferFromCopy " << copy << ":" << wafer << ":" << result;
#endif
  return wafer;
}

void HGCalDDDConstants::waferFromPosition(const double x, const double y, int& wafer, int& icell, int& celltyp) const {
  // Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalParameters::k_ScaleFromDDD * x;
  double yy = HGCalParameters::k_ScaleFromDDD * y;
  int size_ = (int)(hgpar_->waferCopy_.size());
  wafer = size_;
  for (int k = 0; k < size_; ++k) {
    double dx = std::abs(xx - hgpar_->waferPosX_[k]);
    double dy = std::abs(yy - hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5 * hexside_) || (dx * tan30deg_ <= (hexside_ - dy))) {
        wafer = k;
        celltyp = hgpar_->waferTypeT_[k];
        xx -= hgpar_->waferPosX_[k];
        yy -= hgpar_->waferPosY_[k];
        break;
      }
    }
  }
  if (wafer < size_) {
    if (celltyp - 1 == HGCSiliconDetId::HGCalFine)
      icell = cellHex(
          xx, yy, 0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[0], hgpar_->cellFineX_, hgpar_->cellFineY_);
    else
      icell = cellHex(xx,
                      yy,
                      0.5 * HGCalParameters::k_ScaleFromDDD * hgpar_->cellSize_[1],
                      hgpar_->cellCoarseX_,
                      hgpar_->cellCoarseY_);
  } else {
    wafer = -1;
#ifdef EDM_ML_DEBUG
    edm::LogWarning("HGCalGeom") << "Cannot get wafer type corresponding to " << x << ":" << y << "    " << xx << ":"
                                 << yy;
#endif
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Position " << x << ":" << y << " Wafer " << wafer << ":" << size_ << " XX " << xx
                                << ":" << yy << " Cell " << icell << " Type " << celltyp;
#endif
}

void HGCalDDDConstants::waferFromPosition(const double x,
                                          const double y,
                                          const int layer,
                                          int& waferU,
                                          int& waferV,
                                          int& cellU,
                                          int& cellV,
                                          int& celltype,
                                          double& wt,
                                          bool debug) const {
  int ll = layer - hgpar_->firstLayer_;
  double xx = HGCalParameters::k_ScaleFromDDD * x - hgpar_->xLayerHex_[ll];
  double yy = HGCalParameters::k_ScaleFromDDD * y - hgpar_->yLayerHex_[ll];
  waferU = waferV = 1 + hgpar_->waferUVMax_;
  for (unsigned int k = 0; k < hgpar_->waferPosX_.size(); ++k) {
    double dx = std::abs(xx - hgpar_->waferPosX_[k]);
    double dy = std::abs(yy - hgpar_->waferPosY_[k]);
    if (dx <= rmax_ && dy <= hexside_) {
      if ((dy <= 0.5 * hexside_) || (dx * tan30deg_ <= (hexside_ - dy))) {
        waferU = HGCalWaferIndex::waferU(hgpar_->waferCopy_[k]);
        waferV = HGCalWaferIndex::waferV(hgpar_->waferCopy_[k]);
        auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer, waferU, waferV));
        celltype = ((itr == hgpar_->typesInLayers_.end()) ? HGCSiliconDetId::HGCalCoarseThick
                                                          : hgpar_->waferTypeL_[itr->second]);
#ifdef EDM_ML_DEBUG
        if (debug)
          edm::LogVerbatim("HGCalGeom") << "WaferFromPosition:: Input " << layer << ":" << ll << ":"
                                        << hgpar_->firstLayer_ << ":" << x << ":" << y << ":" << hgpar_->xLayerHex_[ll]
                                        << ":" << hgpar_->yLayerHex_[ll] << ":" << xx << ":" << yy << " compared with "
                                        << hgpar_->waferPosX_[k] << ":" << hgpar_->waferPosY_[k] << " difference " << dx
                                        << ":" << dy << ":" << dx * tan30deg_ << ":" << (hexside_ - dy)
                                        << " comparator " << rmax_ << ":" << hexside_ << " wafer " << waferU << ":"
                                        << waferV << ":" << celltype;
#endif
        xx -= hgpar_->waferPosX_[k];
        yy -= hgpar_->waferPosY_[k];
        break;
      }
    }
  }
  if (std::abs(waferU) <= hgpar_->waferUVMax_) {
    cellHex(xx, yy, celltype, cellU, cellV, debug);
    wt = ((celltype < 2) ? (hgpar_->cellThickness_[celltype] / hgpar_->waferThick_) : 1.0);
  } else {
    cellU = cellV = 2 * hgpar_->nCellsFine_;
    wt = 1.0;
    celltype = -1;
  }
#ifdef EDM_ML_DEBUG
  if (celltype < 0) {
    double x1(HGCalParameters::k_ScaleFromDDD * x);
    double y1(HGCalParameters::k_ScaleFromDDD * y);
    edm::LogVerbatim("HGCalGeom") << "waferFromPosition: Bad type for X " << x << ":" << x1 << ":" << xx << " Y " << y
                                  << ":" << y1 << ":" << yy << " Wafer " << waferU << ":" << waferV << " Cell " << cellU
                                  << ":" << cellV;
    for (unsigned int k = 0; k < hgpar_->waferPosX_.size(); ++k) {
      double dx = std::abs(x1 - hgpar_->waferPosX_[k]);
      double dy = std::abs(y1 - hgpar_->waferPosY_[k]);
      edm::LogVerbatim("HGCalGeom") << "Wafer [" << k << "] Position (" << hgpar_->waferPosX_[k] << ", "
                                    << hgpar_->waferPosY_[k] << ") difference " << dx << ":" << dy << ":"
                                    << dx * tan30deg_ << ":" << hexside_ - dy << " Paramerers " << rmax_ << ":"
                                    << hexside_;
    }
  }
#endif
}

bool HGCalDDDConstants::waferInLayer(int wafer, int lay, bool reco) const {
  const auto& indx = getIndex(lay, reco);
  if (indx.first < 0)
    return false;
  return waferInLayerTest(wafer, indx.first, hgpar_->defineFull_);
}

bool HGCalDDDConstants::waferFullInLayer(int wafer, int lay, bool reco) const {
  const auto& indx = getIndex(lay, reco);
  if (indx.first < 0)
    return false;
  return waferInLayerTest(wafer, indx.first, false);
}

std::pair<double, double> HGCalDDDConstants::waferPosition(int wafer, bool reco) const {
  double xx(0), yy(0);
  if (wafer >= 0 && wafer < (int)(hgpar_->waferPosX_.size())) {
    xx = hgpar_->waferPosX_[wafer];
    yy = hgpar_->waferPosY_[wafer];
  }
  if (!reco) {
    xx *= HGCalParameters::k_ScaleToDDD;
    yy *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(xx, yy);
}

std::pair<double, double> HGCalDDDConstants::waferPosition(
    int lay, int waferU, int waferV, bool reco, bool debug) const {
  int ll = lay - hgpar_->firstLayer_;
  double x = hgpar_->xLayerHex_[ll];
  double y = hgpar_->yLayerHex_[ll];
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "Layer " << lay << ":" << ll << " Shift " << hgpar_->xLayerHex_[ll] << ":"
                                  << hgpar_->yLayerHex_[ll];
  if (!reco) {
    x *= HGCalParameters::k_ScaleToDDD;
    y *= HGCalParameters::k_ScaleToDDD;
  }

  const auto& xy = waferPosition(waferU, waferV, reco);
  x += xy.first;
  y += xy.second;
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "With wafer " << x << ":" << y << ":" << xy.first << ":" << xy.second;
  return std::make_pair(x, y);
}

int HGCalDDDConstants::waferType(DetId const& id) const {
  int type(1);
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    type = ((id.det() != DetId::Forward) ? HGCSiliconDetId(id).type() : HFNoseDetId(id).type());
  } else if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    type = waferTypeL(HGCalDetId(id).wafer()) - 1;
  }
  return type;
}

int HGCalDDDConstants::waferType(int layer, int waferU, int waferV) const {
  int type(HGCSiliconDetId::HGCalCoarseThick);
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    auto itr = hgpar_->typesInLayers_.find(HGCalWaferIndex::waferIndex(layer, waferU, waferV));
    if (itr != hgpar_->typesInLayers_.end())
      type = hgpar_->waferTypeL_[itr->second];
  } else if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    if ((waferU >= 0) && (waferU < (int)(hgpar_->waferTypeL_.size())))
      type = (hgpar_->waferTypeL_[waferU] - 1);
  }
  return type;
}

bool HGCalDDDConstants::waferVirtual(int layer, int waferU, int waferV) const {
  bool type(false);
  if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
    int wl = HGCalWaferIndex::waferIndex(layer, waferU, waferV, false);
    type = (hgpar_->waferTypes_.find(wl) != hgpar_->waferTypes_.end());
  } else if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    int wl = HGCalWaferIndex::waferIndex(layer, waferU, 0, true);
    type = (hgpar_->waferTypes_.find(wl) != hgpar_->waferTypes_.end());
  }
  return type;
}

double HGCalDDDConstants::waferZ(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return 0;
  else
    return (reco ? hgpar_->zLayerHex_[index.first] : HGCalParameters::k_ScaleToDDD * hgpar_->zLayerHex_[index.first]);
}

int HGCalDDDConstants::wafers() const {
  int wafer(0);
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    for (unsigned int i = 0; i < layers(true); ++i) {
      int lay = hgpar_->depth_[i];
      wafer += modules(lay, true);
    }
  } else {
    wafer = (int)(hgpar_->moduleLayR_.size());
  }
  return wafer;
}

int HGCalDDDConstants::wafers(int layer, int type) const {
  int wafer(0);
  if (mode_ != HGCalGeometryMode::Trapezoid) {
    auto itr = waferLayer_.find(layer);
    if (itr != waferLayer_.end()) {
      unsigned ity = (type > 0 && type <= 2) ? type : 0;
      wafer = (itr->second)[ity];
    }
  } else {
    const auto& index = getIndex(layer, true);
    wafer = 1 + hgpar_->lastModule_[index.first] - hgpar_->firstModule_[index.first];
  }
  return wafer;
}

int HGCalDDDConstants::cellHex(
    double xx, double yy, const double& cellR, const std::vector<double>& posX, const std::vector<double>& posY) const {
  int num(0);
  const double tol(0.00001);
  double cellY = 2.0 * cellR * tan30deg_;
  for (unsigned int k = 0; k < posX.size(); ++k) {
    double dx = std::abs(xx - posX[k]);
    double dy = std::abs(yy - posY[k]);
    if (dx <= (cellR + tol) && dy <= (cellY + tol)) {
      double xmax = (dy <= 0.5 * cellY) ? cellR : (cellR - (dy - 0.5 * cellY) / tan30deg_);
      if (dx <= (xmax + tol)) {
        num = k;
        break;
      }
    }
  }
  return num;
}

void HGCalDDDConstants::cellHex(double xloc, double yloc, int cellType, int& cellU, int& cellV, bool debug) const {
  int N = (cellType == 0) ? hgpar_->nCellsFine_ : hgpar_->nCellsCoarse_;
  double Rc = 2 * rmax_ / (3 * N);
  double rc = 0.5 * Rc * sqrt3_;
  double v0 = ((xloc / Rc - 1.0) / 1.5);
  int cv0 = (v0 > 0) ? (N + (int)(v0 + 0.5)) : (N - (int)(-v0 + 0.5));
  double u0 = (0.5 * yloc / rc + 0.5 * cv0);
  int cu0 = (u0 > 0) ? (N / 2 + (int)(u0 + 0.5)) : (N / 2 - (int)(-u0 + 0.5));
  cu0 = std::max(0, std::min(cu0, 2 * N - 1));
  cv0 = std::max(0, std::min(cv0, 2 * N - 1));
  if (cv0 - cu0 >= N)
    cv0 = cu0 + N - 1;
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellHex: input " << xloc << ":" << yloc << ":" << cellType << " parameter " << rc
                                  << ":" << Rc << " u0 " << u0 << ":" << cu0 << " v0 " << v0 << ":" << cv0;
  bool found(false);
  static const int shift[3] = {0, 1, -1};
  for (int i1 = 0; i1 < 3; ++i1) {
    cellU = cu0 + shift[i1];
    for (int i2 = 0; i2 < 3; ++i2) {
      cellV = cv0 + shift[i2];
      if (((cellV - cellU) < N) && ((cellU - cellV) <= N) && (cellU >= 0) && (cellV >= 0) && (cellU < 2 * N) &&
          (cellV < 2 * N)) {
        double xc = (1.5 * (cellV - N) + 1.0) * Rc;
        double yc = (2 * cellU - cellV - N) * rc;
        if ((std::abs(yloc - yc) <= rc) && (std::abs(xloc - xc) <= Rc) &&
            ((std::abs(xloc - xc) <= 0.5 * Rc) || (std::abs(yloc - yc) <= sqrt3_ * (Rc - std::abs(xloc - xc))))) {
          if (debug)
            edm::LogVerbatim("HGCalGeom")
                << "cellHex: local " << xc << ":" << yc << " difference " << std::abs(xloc - xc) << ":"
                << std::abs(yloc - yc) << ":" << sqrt3_ * (Rc - std::abs(yloc - yc)) << " comparator " << rc << ":"
                << Rc << " (u,v) = (" << cellU << "," << cellV << ")";
          found = true;
          break;
        }
      }
    }
    if (found)
      break;
  }
  if (!found) {
    cellU = cu0;
    cellV = cv0;
  }
}

std::pair<int, float> HGCalDDDConstants::getIndex(int lay, bool reco) const {
  int indx = layerIndex(lay, reco);
  if (indx < 0)
    return std::make_pair(-1, 0);
  float cell(0);
  if ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) {
    cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
  } else {
    if ((mode_ == HGCalGeometryMode::Hexagon8) || (mode_ == HGCalGeometryMode::Hexagon8Full)) {
      cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
    } else {
      cell = hgpar_->scintCellSize(lay);
    }
  }
  return std::make_pair(indx, cell);
}

bool HGCalDDDConstants::isValidCell(int lay, int wafer, int cell) const {
  // Calculate the position of the cell
  // Works for options HGCalHexagon/HGCalHexagonFull
  double x = hgpar_->waferPosX_[wafer];
  double y = hgpar_->waferPosY_[wafer];
  if (hgpar_->waferTypeT_[wafer] - 1 == HGCSiliconDetId::HGCalFine) {
    x += hgpar_->cellFineX_[cell];
    y += hgpar_->cellFineY_[cell];
  } else {
    x += hgpar_->cellCoarseX_[cell];
    y += hgpar_->cellCoarseY_[cell];
  }
  double rr = sqrt(x * x + y * y);
  bool result = ((rr >= hgpar_->rMinLayHex_[lay - 1]) && (rr <= hgpar_->rMaxLayHex_[lay - 1]) &&
                 (wafer < (int)(hgpar_->waferPosX_.size())));
#ifdef EDM_ML_DEBUG
  if (!result)
    edm::LogVerbatim("HGCalGeom") << "Input " << lay << ":" << wafer << ":" << cell << " Position " << x << ":" << y
                                  << ":" << rr << " Compare Limits " << hgpar_->rMinLayHex_[lay - 1] << ":"
                                  << hgpar_->rMaxLayHex_[lay - 1] << " Flag " << result;
#endif
  return result;
}

bool HGCalDDDConstants::isValidCell8(int lay, int waferU, int waferV, int cellU, int cellV, int type) const {
  float x(0), y(0);
  int kndx = cellV * 100 + cellU;
  if (type == 0) {
    auto ktr = hgpar_->cellFineIndex_.find(kndx);
    if (ktr != hgpar_->cellFineIndex_.end()) {
      x = hgpar_->cellFineX_[ktr->second];
      y = hgpar_->cellFineY_[ktr->second];
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Fine " << cellU << ":" << cellV << ":" << kndx << ":" << x << ":" << y << ":"
                                  << (ktr != hgpar_->cellFineIndex_.end());
#endif
  } else {
    auto ktr = hgpar_->cellCoarseIndex_.find(kndx);
    if (ktr != hgpar_->cellCoarseIndex_.end()) {
      x = hgpar_->cellCoarseX_[ktr->second];
      y = hgpar_->cellCoarseY_[ktr->second];
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "Coarse " << cellU << ":" << cellV << ":" << kndx << ":" << x << ":" << y << ":"
                                  << (ktr != hgpar_->cellCoarseIndex_.end());
#endif
  }
  const auto& xy = waferPosition(lay, waferU, waferV, true, false);
  x += xy.first;
  y += xy.second;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "With wafer (" << waferU << "," << waferV << ") " << x << ":" << y;
#endif
  double rr = sqrt(x * x + y * y);
  int ll = lay - hgpar_->firstLayer_;
  bool result = ((rr >= hgpar_->rMinLayHex_[ll]) && (rr <= hgpar_->rMaxLayHex_[ll]));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Input " << lay << ":" << ll << ":" << waferU << ":" << waferV << ":" << cellU << ":"
                                << cellV << " Position " << x << ":" << y << ":" << rr << " Compare Limits "
                                << hgpar_->rMinLayHex_[ll] << ":" << hgpar_->rMaxLayHex_[ll] << " Flag " << result;
#endif
  return result;
}

bool HGCalDDDConstants::waferInLayerTest(int wafer, int lay, bool full) const {
  bool flag = ((mode_ == HGCalGeometryMode::Hexagon) || (mode_ == HGCalGeometryMode::HexagonFull)) ? true : false;
  std::pair<int, int> corner = HGCalGeomTools::waferCorner(hgpar_->waferPosX_[wafer],
                                                           hgpar_->waferPosY_[wafer],
                                                           rmax_,
                                                           hexside_,
                                                           hgpar_->rMinLayHex_[lay],
                                                           hgpar_->rMaxLayHex_[lay],
                                                           flag);
  bool in = (full ? (corner.first > 0) : (corner.first == (int)(HGCalParameters::k_CornerSize)));
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "WaferInLayerTest: Layer " << lay << " wafer " << wafer << " R-limits "
                                << hgpar_->rMinLayHex_[lay] << ":" << hgpar_->rMaxLayHex_[lay] << " Corners "
                                << corner.first << ":" << corner.second << " In " << in;
#endif
  return in;
}

std::pair<double, double> HGCalDDDConstants::waferPosition(int waferU, int waferV, bool reco) const {
  double xx(0), yy(0);
  int indx = HGCalWaferIndex::waferIndex(0, waferU, waferV);
  auto itr = hgpar_->wafersInLayers_.find(indx);
  if (itr != hgpar_->wafersInLayers_.end()) {
    xx = hgpar_->waferPosX_[itr->second];
    yy = hgpar_->waferPosY_[itr->second];
  }
  if (!reco) {
    xx *= HGCalParameters::k_ScaleToDDD;
    yy *= HGCalParameters::k_ScaleToDDD;
  }
  return std::make_pair(xx, yy);
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalDDDConstants);
