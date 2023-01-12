#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"

#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBGeomParameters.h"

#include <algorithm>
#include <bitset>
#include <iterator>
#include <functional>
#include <numeric>

//#define EDM_ML_DEBUG
using namespace geant_units::operators;

HGCalTBDDDConstants::HGCalTBDDDConstants(const HGCalTBParameters* hp, const std::string& name)
    : hgpar_(hp), sqrt3_(std::sqrt(3.0)), mode_(hgpar_->mode_) {
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Mode " << mode_;
#endif
  if (waferHexagon6()) {
    rmax_ = (HGCalTBParameters::k_ScaleFromDDD * (hgpar_->waferR_) * std::cos(30._deg));
    rmaxT_ = rmax_ + 0.5 * hgpar_->sensorSeparation_;
    hexside_ = 2.0 * rmax_ * tan30deg_;
    hexsideT_ = 2.0 * rmaxT_ * tan30deg_;
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "rmax_ " << rmax_ << ":" << rmaxT_ << ":" << hexside_ << ":" << hexsideT_
                                  << " CellSize " << 0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[0]
                                  << ":" << 0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[1];
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
                                      << modHalf_ << " modules in RECO";
      } else {
        edm::LogVerbatim("HGCalGeom") << "Layer " << layer << " with " << max_modules_layer_[simreco][layer]
                                      << " modules in SIM";
#endif
      }
    }
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("HGCalGeom") << "SimReco " << simreco << " with " << tot_layers_[simreco] << " Layers";
#endif
  }
  tot_wafers_ = wafers();

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants initialized for " << name << " with " << layers(false) << ":"
                                << layers(true) << " layers, " << wafers() << ":" << 2 * modHalf_
                                << " wafers with maximum " << maxWafersPerLayer_ << " per layer and "
                                << "maximum of " << maxCells(false) << ":" << maxCells(true) << " cells";
#endif
  if (waferHexagon6()) {
    int wminT(9999999), wmaxT(-9999999), kount1(0), kount2(0);
    for (unsigned int i = 0; i < getTrFormN(); ++i) {
      int lay0 = getTrForm(i).lay;
      int wmin(9999999), wmax(-9999999), kount(0);
      for (int wafer = 0; wafer < sectors(); ++wafer) {
        bool waferIn = waferInLayer(wafer, lay0, true);
        if (waferIn) {
          if (wafer < wmin)
            wmin = wafer;
          if (wafer > wmax)
            wmax = wafer;
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

std::pair<int, int> HGCalTBDDDConstants::assignCell(float x, float y, int lay, int subSec, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return std::make_pair(-1, -1);
  if (waferHexagon6()) {
    float xx = (reco) ? x : HGCalTBParameters::k_ScaleFromDDD * x;
    float yy = (reco) ? y : HGCalTBParameters::k_ScaleFromDDD * y;

    // First the wafer
    int wafer = cellHex(xx, yy, rmax_, hgpar_->waferPosX_, hgpar_->waferPosY_);
    if (wafer < 0 || wafer >= static_cast<int>(hgpar_->waferTypeT_.size())) {
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
                                      0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[0],
                                      hgpar_->cellFineX_,
                                      hgpar_->cellFineY_));
      else
        return std::make_pair(wafer,
                              cellHex(xx,
                                      yy,
                                      0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[1],
                                      hgpar_->cellCoarseX_,
                                      hgpar_->cellCoarseY_));
    }
  } else {
    return std::make_pair(-1, -1);
  }
}

double HGCalTBDDDConstants::cellSizeHex(int type) const {
  int indx = (type == 1) ? 1 : 0;
  double cell = 0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[indx];
  return cell;
}

double HGCalTBDDDConstants::cellThickness(int layer, int wafer) const {
  double thick(-1);
  int type = waferType(layer, wafer);
  if (type >= 0)
    thick = 100.0 * (type + 1);  // type = 1,2,3 for 100,200,300 micron
  return thick;
}

double HGCalTBDDDConstants::distFromEdgeHex(double x, double y, double z) const {
  // Assming the point is within a hexagonal plane of the wafer, calculate
  // the shortest distance from the edge
  if (z < 0)
    x = -x;
  double dist(0);
  // Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalTBParameters::k_ScaleFromDDD * x;
  double yy = HGCalTBParameters::k_ScaleFromDDD * y;
  int sizew = static_cast<int>(hgpar_->waferPosX_.size());
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
  dist *= HGCalTBParameters::k_ScaleToDDD;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "DistFromEdgeHex: Local " << xx << ":" << yy << " wafer " << wafer << " flag "
                                << (wafer < sizew) << " Distance " << rmax_ << ":" << (rmax_ - std::abs(xx)) << ":"
                                << (std::abs(yy) - 0.5 * hexside_) << ":" << 0.5 * hexside_ << ":" << dist;
#endif
  return dist;
}

int HGCalTBDDDConstants::getLayer(double z, bool reco) const {
  // Get the layer # from the gloabl z coordinate
  unsigned int k = 0;
  double zz = (reco ? std::abs(z) : HGCalTBParameters::k_ScaleFromDDD * std::abs(z));
  const auto& zLayerHex = hgpar_->zLayerHex_;
  auto itr = std::find_if(zLayerHex.begin() + 1, zLayerHex.end(), [&k, &zz, &zLayerHex](double zLayer) {
    ++k;
    return zz < 0.5 * (zLayerHex[k - 1] + zLayerHex[k]);
  });
  int lay = (itr == zLayerHex.end()) ? static_cast<int>(zLayerHex.size()) : k;
  if (waferHexagon6() && reco) {
    int indx = layerIndex(lay, false);
    if (indx >= 0)
      lay = hgpar_->layerGroupO_[indx];
  } else {
    lay += (hgpar_->firstLayer_ - 1);
  }
  return lay;
}

HGCalTBParameters::hgtrap HGCalTBDDDConstants::getModule(unsigned int indx, bool hexType, bool reco) const {
  HGCalTBParameters::hgtrap mytr;
  if (hexType) {
    if (indx >= hgpar_->waferTypeL_.size())
      edm::LogWarning("HGCalGeom") << "Wafer no. out bound for index " << indx << ":" << (hgpar_->waferTypeL_).size()
                                   << ":" << (hgpar_->waferPosX_).size() << ":" << (hgpar_->waferPosY_).size()
                                   << " ***** ERROR *****";
    unsigned int type =
        ((indx < hgpar_->waferTypeL_.size()) ? hgpar_->waferTypeL_[indx] - 1 : HGCalTBParameters::HGCalCoarseThick);
    mytr = hgpar_->getModule(type, reco);
  } else {
    mytr = hgpar_->getModule(indx, reco);
  }
  return mytr;
}

std::vector<HGCalTBParameters::hgtrap> HGCalTBDDDConstants::getModules() const {
  std::vector<HGCalTBParameters::hgtrap> mytrs;
  for (unsigned int k = 0; k < hgpar_->moduleLayR_.size(); ++k)
    mytrs.emplace_back(hgpar_->getModule(k, true));
  return mytrs;
}

std::vector<HGCalTBParameters::hgtrform> HGCalTBDDDConstants::getTrForms() const {
  std::vector<HGCalTBParameters::hgtrform> mytrs;
  for (unsigned int k = 0; k < hgpar_->trformIndex_.size(); ++k)
    mytrs.emplace_back(hgpar_->getTrForm(k));
  return mytrs;
}

bool HGCalTBDDDConstants::isHalfCell(int waferType, int cell) const {
  if (waferType < 1 || cell < 0)
    return false;
  return waferType == 2 ? hgpar_->cellCoarseHalf_[cell] : hgpar_->cellFineHalf_[cell];
}

bool HGCalTBDDDConstants::isValidHex(int lay, int mod, int cell, bool reco) const {
  // Check validity for a layer|wafer|cell of pre-TDR version
  bool result(false), resultMod(false);
  int cellmax(0);
  if (waferHexagon6()) {
    int32_t copyNumber = hgpar_->waferCopy_[mod];
    result = ((lay > 0 && lay <= static_cast<int>(layers(reco))));
    if (result) {
      const int32_t lay_idx = reco ? (lay - 1) * 3 + 1 : lay;
      const auto& the_modules = hgpar_->copiesInLayers_[lay_idx];
      auto moditr = the_modules.find(copyNumber);
      result = resultMod = (moditr != the_modules.end());
#ifdef EDM_ML_DEBUG
      if (!result)
        edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants: Layer " << lay << ":" << lay_idx << " Copy "
                                      << copyNumber << ":" << mod << " Flag " << result;
#endif
      if (result) {
        if (moditr->second >= 0) {
          if (mod >= static_cast<int>(hgpar_->waferTypeT_.size()))
            edm::LogWarning("HGCalGeom") << "Module no. out of bound for " << mod << " to be compared with "
                                         << (hgpar_->waferTypeT_).size() << " ***** ERROR *****";
          cellmax = ((hgpar_->waferTypeT_[mod] - 1 == HGCalTBParameters::HGCalFine)
                         ? static_cast<int>(hgpar_->cellFineX_.size())
                         : static_cast<int>(hgpar_->cellCoarseX_.size()));
          result = (cell >= 0 && cell <= cellmax);
        } else {
          result = isValidCell(lay_idx, mod, cell);
        }
      }
    }
  }

#ifdef EDM_ML_DEBUG
  if (!result)
    edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants: Layer " << lay << ":"
                                  << (lay > 0 && (lay <= static_cast<int>(layers(reco)))) << " Module " << mod << ":"
                                  << resultMod << " Cell " << cell << ":" << cellmax << ":"
                                  << (cell >= 0 && cell <= cellmax) << ":" << maxCells(reco);
#endif
  return result;
}

int HGCalTBDDDConstants::lastLayer(bool reco) const {
  return (hgpar_->firstLayer_ + tot_layers_[static_cast<int>(reco)] - 1);
}

int HGCalTBDDDConstants::layerIndex(int lay, bool reco) const {
  int ll = lay - hgpar_->firstLayer_;
  if (ll < 0 || ll >= static_cast<int>(hgpar_->layerIndex_.size()))
    return -1;
  if (waferHexagon6()) {
    if (reco && ll >= static_cast<int>(hgpar_->depthIndex_.size()))
      return -1;
    return (reco ? hgpar_->depthLayerF_[ll] : hgpar_->layerIndex_[ll]);
  }
  return -1;
}

unsigned int HGCalTBDDDConstants::layers(bool reco) const { return tot_layers_[static_cast<int>(reco)]; }

unsigned int HGCalTBDDDConstants::layersInit(bool reco) const {
  return (reco ? hgpar_->depthIndex_.size() : hgpar_->layerIndex_.size());
}

std::pair<float, float> HGCalTBDDDConstants::locateCell(int cell, int lay, int type, bool reco) const {
  // type refers to wafer # for hexagon cell
  float x(999999.), y(999999.);
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  if (i < 0)
    return std::make_pair(x, y);
  if (waferHexagon6()) {
    x = hgpar_->waferPosX_[type];
    y = hgpar_->waferPosY_[type];
#ifdef EDM_ML_DEBUG
    float x0(x), y0(y);
#endif
    if (hgpar_->waferTypeT_[type] - 1 == HGCalTBParameters::HGCalFine) {
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
      x *= HGCalTBParameters::k_ScaleToDDD;
      y *= HGCalTBParameters::k_ScaleToDDD;
    }
  }
  return std::make_pair(x, y);
}

std::pair<float, float> HGCalTBDDDConstants::locateCellHex(int cell, int wafer, bool reco) const {
  float x(0), y(0);
  if (hgpar_->waferTypeT_[wafer] - 1 == HGCalTBParameters::HGCalFine) {
    x = hgpar_->cellFineX_[cell];
    y = hgpar_->cellFineY_[cell];
  } else {
    x = hgpar_->cellCoarseX_[cell];
    y = hgpar_->cellCoarseY_[cell];
  }
  if (!reco) {
    x *= HGCalTBParameters::k_ScaleToDDD;
    y *= HGCalTBParameters::k_ScaleToDDD;
  }
  return std::make_pair(x, y);
}

int HGCalTBDDDConstants::maxCells(bool reco) const {
  int cells(0);
  for (unsigned int i = 0; i < layers(reco); ++i) {
    int lay = reco ? hgpar_->depth_[i] : hgpar_->layer_[i];
    if (cells < maxCells(lay, reco))
      cells = maxCells(lay, reco);
  }
  return cells;
}

int HGCalTBDDDConstants::maxCells(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if ((index.first < 0) || (!waferHexagon6()))
    return 0;
  unsigned int cells(0);
  for (unsigned int k = 0; k < hgpar_->waferTypeT_.size(); ++k) {
    if (waferInLayerTest(k, index.first)) {
      unsigned int cell = (hgpar_->waferTypeT_[k] - 1 == HGCalTBParameters::HGCalFine) ? (hgpar_->cellFineX_.size())
                                                                                       : (hgpar_->cellCoarseX_.size());
      if (cell > cells)
        cells = cell;
    }
  }
  return static_cast<int>(cells);
}

int HGCalTBDDDConstants::maxRows(int lay, bool reco) const {
  int kymax(0);
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  if ((i >= 0) && waferHexagon6()) {
    for (unsigned int k = 0; k < hgpar_->waferCopy_.size(); ++k) {
      if (waferInLayerTest(k, i)) {
        int ky = ((hgpar_->waferCopy_[k]) / 100) % 100;
        if (ky > kymax)
          kymax = ky;
      }
    }
  }
  return kymax;
}

int HGCalTBDDDConstants::modifyUV(int uv, int type1, int type2) const {
  // Modify u/v for transition of type1 to type2
  return (((type1 == type2) || (type1 * type2 != 0)) ? uv : ((type1 == 0) ? (2 * uv + 1) / 3 : (3 * uv) / 2));
}

int HGCalTBDDDConstants::modules(int lay, bool reco) const {
  if (getIndex(lay, reco).first < 0)
    return 0;
  else
    return max_modules_layer_[static_cast<int>(reco)][lay];
}

int HGCalTBDDDConstants::modulesInit(int lay, bool reco) const {
  int nmod(0);
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return nmod;
  for (unsigned int k = 0; k < hgpar_->waferPosX_.size(); ++k) {
    if (waferInLayerTest(k, index.first))
      ++nmod;
  }
  return nmod;
}

double HGCalTBDDDConstants::mouseBite(bool reco) const {
  return (reco ? hgpar_->mouseBite_ : HGCalTBParameters::k_ScaleToDDD * hgpar_->mouseBite_);
}

int HGCalTBDDDConstants::numberCells(bool reco) const {
  int cells(0);
  unsigned int nlayer = (reco) ? hgpar_->depth_.size() : hgpar_->layer_.size();
  for (unsigned k = 0; k < nlayer; ++k) {
    std::vector<int> ncells = numberCells(((reco) ? hgpar_->depth_[k] : hgpar_->layer_[k]), reco);
    cells = std::accumulate(ncells.begin(), ncells.end(), cells);
  }
  return cells;
}

std::vector<int> HGCalTBDDDConstants::numberCells(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  int i = index.first;
  std::vector<int> ncell;
  if ((i >= 0) && (waferHexagon6())) {
    for (unsigned int k = 0; k < hgpar_->waferTypeT_.size(); ++k) {
      if (waferInLayerTest(k, i)) {
        unsigned int cell = (hgpar_->waferTypeT_[k] - 1 == HGCalTBParameters::HGCalFine)
                                ? (hgpar_->cellFineX_.size())
                                : (hgpar_->cellCoarseX_.size());
        ncell.emplace_back(static_cast<int>(cell));
      }
    }
  }
  return ncell;
}

int HGCalTBDDDConstants::numberCellsHexagon(int wafer) const {
  if (wafer >= 0 && wafer < static_cast<int>(hgpar_->waferTypeT_.size())) {
    if (hgpar_->waferTypeT_[wafer] - 1 == 0)
      return static_cast<int>(hgpar_->cellFineX_.size());
    else
      return static_cast<int>(hgpar_->cellCoarseX_.size());
  } else {
    return 0;
  }
}

std::pair<double, double> HGCalTBDDDConstants::rangeR(double z, bool reco) const {
  double rmin(0), rmax(0);
  if (!reco) {
    rmin *= HGCalTBParameters::k_ScaleToDDD;
    rmax *= HGCalTBParameters::k_ScaleToDDD;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants:rangeR: " << z << ":0"
                                << " R " << rmin << ":" << rmax;
#endif
  return std::make_pair(rmin, rmax);
}

std::pair<double, double> HGCalTBDDDConstants::rangeRLayer(int lay, bool reco) const {
  double rmin(0), rmax(0);
  const auto& index = getIndex(lay, reco);
  if (index.first >= 0 && index.first < static_cast<int>(hgpar_->rMinLayHex_.size())) {
    rmin = hgpar_->rMinLayHex_[index.first];
    rmax = hgpar_->rMaxLayHex_[index.first];
  }
  if (!reco) {
    rmin *= HGCalTBParameters::k_ScaleToDDD;
    rmax *= HGCalTBParameters::k_ScaleToDDD;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants:rangeR: " << lay << ":" << index.first << " R " << rmin << ":"
                                << rmax;
#endif
  return std::make_pair(rmin, rmax);
}

std::pair<double, double> HGCalTBDDDConstants::rangeZ(bool reco) const {
  double zmin = (hgpar_->zLayerHex_[0] - hgpar_->waferThick_);
  double zmax = (hgpar_->zLayerHex_[hgpar_->zLayerHex_.size() - 1] + hgpar_->waferThick_);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalTBDDDConstants:rangeZ: " << zmin << ":" << zmax << ":" << hgpar_->waferThick_;
#endif
  if (!reco) {
    zmin *= HGCalTBParameters::k_ScaleToDDD;
    zmax *= HGCalTBParameters::k_ScaleToDDD;
  }
  return std::make_pair(zmin, zmax);
}

std::pair<int, int> HGCalTBDDDConstants::rowColumnWafer(int wafer) const {
  int row(0), col(0);
  if (wafer < static_cast<int>(hgpar_->waferCopy_.size())) {
    int copy = hgpar_->waferCopy_[wafer];
    col = HGCalTypes::getUnpackedU(copy);
    row = HGCalTypes::getUnpackedV(copy);
  }
  return std::make_pair(row, col);
}

std::pair<int, int> HGCalTBDDDConstants::simToReco(int cell, int lay, int mod, bool half) const {
  return std::make_pair(cell, lay);
}

int HGCalTBDDDConstants::waferFromCopy(int copy) const {
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

void HGCalTBDDDConstants::waferFromPosition(const double x, const double y, int& wafer, int& icell, int& celltyp) const {
  // Input x, y in Geant4 unit and transformed to CMSSW standard
  double xx = HGCalTBParameters::k_ScaleFromDDD * x;
  double yy = HGCalTBParameters::k_ScaleFromDDD * y;
  int size_ = static_cast<int>(hgpar_->waferCopy_.size());
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
    if (celltyp - 1 == HGCalTBParameters::HGCalFine)
      icell = cellHex(xx,
                      yy,
                      0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[0],
                      hgpar_->cellFineX_,
                      hgpar_->cellFineY_);
    else
      icell = cellHex(xx,
                      yy,
                      0.5 * HGCalTBParameters::k_ScaleFromDDD * hgpar_->cellSize_[1],
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

bool HGCalTBDDDConstants::waferInLayer(int wafer, int lay, bool reco) const {
  const auto& indx = getIndex(lay, reco);
  if (indx.first < 0)
    return false;
  return waferInLayerTest(wafer, indx.first);
}

bool HGCalTBDDDConstants::waferFullInLayer(int wafer, int lay, bool reco) const {
  const auto& indx = getIndex(lay, reco);
  if (indx.first < 0)
    return false;
  return waferInLayerTest(wafer, indx.first);
}

std::pair<double, double> HGCalTBDDDConstants::waferParameters(bool reco) const {
  if (reco)
    return std::make_pair(rmax_, hexside_);
  else
    return std::make_pair(HGCalTBParameters::k_ScaleToDDD * rmax_, HGCalTBParameters::k_ScaleToDDD * hexside_);
}

std::pair<double, double> HGCalTBDDDConstants::waferPosition(int wafer, bool reco) const {
  double xx(0), yy(0);
  if (wafer >= 0 && wafer < static_cast<int>(hgpar_->waferPosX_.size())) {
    xx = hgpar_->waferPosX_[wafer];
    yy = hgpar_->waferPosY_[wafer];
  }
  if (!reco) {
    xx *= HGCalTBParameters::k_ScaleToDDD;
    yy *= HGCalTBParameters::k_ScaleToDDD;
  }
  return std::make_pair(xx, yy);
}

int HGCalTBDDDConstants::wafers() const { return static_cast<int>(hgpar_->moduleLayR_.size()); }

int HGCalTBDDDConstants::wafers(int layer, int type) const {
  int wafer(0);
  auto itr = waferLayer_.find(layer);
  if (itr != waferLayer_.end()) {
    unsigned ity = (type > 0 && type <= 2) ? type : 0;
    wafer = (itr->second)[ity];
  }
  return wafer;
}

int HGCalTBDDDConstants::waferType(DetId const& id) const {
  return waferType(HGCalDetId(id).layer(), HGCalDetId(id).wafer());
}

int HGCalTBDDDConstants::waferType(int layer, int wafer) const {
  int type(HGCalTBParameters::HGCalCoarseThick);
  if ((wafer >= 0) && (wafer < static_cast<int>(hgpar_->waferTypeL_.size())))
    type = (hgpar_->waferTypeL_[wafer] - 1);
  return type;
}

bool HGCalTBDDDConstants::waferVirtual(int layer, int wafer) const {
  int wl = HGCalWaferIndex::waferIndex(layer, wafer, 0, true);
  return (hgpar_->waferTypes_.find(wl) != hgpar_->waferTypes_.end());
}

double HGCalTBDDDConstants::waferZ(int lay, bool reco) const {
  const auto& index = getIndex(lay, reco);
  if (index.first < 0)
    return 0;
  else
    return (reco ? hgpar_->zLayerHex_[index.first] : HGCalTBParameters::k_ScaleToDDD * hgpar_->zLayerHex_[index.first]);
}

int HGCalTBDDDConstants::cellHex(
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

std::pair<int, float> HGCalTBDDDConstants::getIndex(int lay, bool reco) const {
  int indx = layerIndex(lay, reco);
  if (indx < 0)
    return std::make_pair(-1, 0);
  float cell(0);
  if (waferHexagon6()) {
    cell = (reco ? hgpar_->moduleCellR_[0] : hgpar_->moduleCellS_[0]);
  }
  return std::make_pair(indx, cell);
}

int HGCalTBDDDConstants::layerFromIndex(int index, bool reco) const {
  int ll(-1);
  if (waferHexagon6() && reco) {
    ll = static_cast<int>(std::find(hgpar_->depthLayerF_.begin(), hgpar_->depthLayerF_.end(), index) -
                          hgpar_->depthLayerF_.begin());
    if (ll == static_cast<int>(hgpar_->depthLayerF_.size()))
      ll = -1;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "LayerFromIndex for " << index << ":" << reco << ":" << waferHexagon6() << " is"
                                << ll << ":" << (ll + hgpar_->firstLayer_);
#endif
  return ((ll < 0) ? ll : (ll + hgpar_->firstLayer_));
}

bool HGCalTBDDDConstants::isValidCell(int lay, int wafer, int cell) const {
  // Calculate the position of the cell
  // Works for options HGCalHexagon/HGCalHexagonFull
  double x = hgpar_->waferPosX_[wafer];
  double y = hgpar_->waferPosY_[wafer];
  if (hgpar_->waferTypeT_[wafer] - 1 == HGCalTBParameters::HGCalFine) {
    x += hgpar_->cellFineX_[cell];
    y += hgpar_->cellFineY_[cell];
  } else {
    x += hgpar_->cellCoarseX_[cell];
    y += hgpar_->cellCoarseY_[cell];
  }
  double rr = sqrt(x * x + y * y);
  bool result = ((rr >= hgpar_->rMinLayHex_[lay - 1]) && (rr <= hgpar_->rMaxLayHex_[lay - 1]) &&
                 (wafer < static_cast<int>(hgpar_->waferPosX_.size())));
#ifdef EDM_ML_DEBUG
  if (!result)
    edm::LogVerbatim("HGCalGeom") << "Input " << lay << ":" << wafer << ":" << cell << " Position " << x << ":" << y
                                  << ":" << rr << " Compare Limits " << hgpar_->rMinLayHex_[lay - 1] << ":"
                                  << hgpar_->rMaxLayHex_[lay - 1] << " Flag " << result;
#endif
  return result;
}

int32_t HGCalTBDDDConstants::waferIndex(int wafer, int index) const {
  int layer = layerFromIndex(index, true);
  int indx = HGCalWaferIndex::waferIndex(layer, wafer, 0);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "WaferIndex for " << wafer << ":" << index << " (" << layer << ") " << indx;
#endif
  return indx;
}

#include "FWCore/Utilities/interface/typelookup.h"

TYPELOOKUP_DATA_REG(HGCalTBDDDConstants);
