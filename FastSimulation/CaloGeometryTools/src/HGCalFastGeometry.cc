#include "FastSimulation/CaloGeometryTools/interface/HGCalFastGeometry.h"

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/HGCalCommonData/interface/HGCalWaferIndex.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

HGCalFastGeometry::HGCalFastGeometry(const HGCalDDDConstants& ddd, DetId::Detector det, double binCm)
    : ddd_(ddd), det_(det), silicon_(det != DetId::HGCalHSc) {
  const bool supported = (det == DetId::HGCalEE || det == DetId::HGCalHSi) ? ddd_.waferHexagon8()
                         : (det == DetId::HGCalHSc)                        ? ddd_.tileTrapezoid()
                                                                           : false;
  if (!supported)
    throw cms::Exception("HGCalFastGeometry") << "detector " << det
                                              << " does not match the geometry mode of the HGCalDDDConstants: only"
                                                 " Hexagon8 silicon (HGCalEE, HGCalHSi) and trapezoid scintillator"
                                                 " (HGCalHSc) are supported";

  firstLayer_ = ddd_.firstLayer();
  lastLayer_ = ddd_.lastLayer(true);
  if (lastLayer_ < firstLayer_)
    return;

  // One bucket per wafer pitch keeps the occupancy at a handful of wafers, and
  // guarantees that the containing wafer's centre is in the 3x3 neighbourhood.
  binCm = std::max(binCm, ddd_.waferSize(true) + ddd_.waferSepar(true));

  const int n = nLayers();
  layerZ_.assign(n, 0.);
  layerR_.assign(n, std::make_pair(0., 0.));
  grids_.resize(n);

  for (int lay = firstLayer_; lay <= lastLayer_; ++lay) {
    const int i = lay - firstLayer_;
    layerZ_[i] = ddd_.waferZ(lay, true);
    layerR_[i] = ddd_.rangeRLayer(lay, true);
    if (silicon_)
      buildLayer(lay, binCm);
  }
  // The total wafer count is only known after the loop; drop the spare capacity.
  wafers_.shrink_to_fit();
}

void HGCalFastGeometry::buildLayer(int layer, double binCm) {
  const HGCalParameters* p = ddd_.getParameter();
  LayerGrid& g = grids_[layer - firstLayer_];
  g.waferBegin = static_cast<uint32_t>(wafers_.size());

  // In cassette mode the wafer centres are shifted; waferPositionWithCshift()
  // applies that shift but dereferences waferInfoMap_ without checking end(),
  // so it may only be called for wafers that are actually in the map.
  const bool cassette = ddd_.cassetteMode() && ddd_.waferHexagon8File();

  for (unsigned int k = 0; k < p->waferCopy_.size(); ++k) {
    const int u = HGCalWaferIndex::waferU(p->waferCopy_[k]);
    const int v = HGCalWaferIndex::waferV(p->waferCopy_[k]);
    if (!ddd_.isValidHex8(layer, u, v, true))
      continue;

    std::pair<double, double> xy;
    if (cassette) {
      if (!ddd_.waferFileInfoExist(HGCalWaferIndex::waferIndex(layer, u, v)))
        continue;
      xy = ddd_.waferPositionWithCshift(layer, u, v, false, true, false);
    } else {
      xy = ddd_.waferPosition(layer, u, v, true, false);
    }
    wafers_.push_back(Wafer{
        static_cast<float>(xy.first), static_cast<float>(xy.second), static_cast<int16_t>(u), static_cast<int16_t>(v)});
  }
  g.waferEnd = static_cast<uint32_t>(wafers_.size());
  if (g.waferEnd == g.waferBegin)
    return;

  double xMin = std::numeric_limits<double>::max(), xMax = -xMin;
  double yMin = xMin, yMax = -xMin;
  for (uint32_t i = g.waferBegin; i < g.waferEnd; ++i) {
    xMin = std::min(xMin, static_cast<double>(wafers_[i].x));
    xMax = std::max(xMax, static_cast<double>(wafers_[i].x));
    yMin = std::min(yMin, static_cast<double>(wafers_[i].y));
    yMax = std::max(yMax, static_cast<double>(wafers_[i].y));
  }
  // One bucket of padding on each side so the 3x3 scan never leaves the grid.
  g.xMin = xMin - binCm;
  g.yMin = yMin - binCm;
  g.nx = static_cast<int>((xMax - g.xMin) / binCm) + 2;
  g.ny = static_cast<int>((yMax - g.yMin) / binCm) + 2;
  g.binCm = binCm;

  const std::size_t nb = static_cast<std::size_t>(g.nx) * g.ny;
  std::vector<uint32_t> count(nb + 1, 0);
  auto bucketOf = [&](const Wafer& w) {
    const int ix = std::min(g.nx - 1, std::max(0, static_cast<int>((w.x - g.xMin) / binCm)));
    const int iy = std::min(g.ny - 1, std::max(0, static_cast<int>((w.y - g.yMin) / binCm)));
    return static_cast<std::size_t>(iy) * g.nx + ix;
  };
  for (uint32_t i = g.waferBegin; i < g.waferEnd; ++i)
    ++count[bucketOf(wafers_[i]) + 1];
  for (std::size_t b = 0; b < nb; ++b)
    count[b + 1] += count[b];

  g.bucketOffset = std::move(count);
  g.bucketWafer.resize(g.waferEnd - g.waferBegin);
  std::vector<uint32_t> fill(nb, 0);
  for (uint32_t i = g.waferBegin; i < g.waferEnd; ++i) {
    const std::size_t b = bucketOf(wafers_[i]);
    g.bucketWafer[g.bucketOffset[b] + fill[b]++] = i;
  }
}

int HGCalFastGeometry::nearestWafer(const LayerGrid& g, double x, double y, int& u, int& v) const {
  if (g.nx == 0 || g.ny == 0)
    return -1;
  // Any point inside a wafer falls inside the padded grid. Reject everything else,
  // NaN included, before the conversion to int.
  const double fx = (x - g.xMin) / g.binCm;
  const double fy = (y - g.yMin) / g.binCm;
  if (!(fx >= 0. && fx < g.nx && fy >= 0. && fy < g.ny))
    return -1;
  const int ix0 = static_cast<int>(fx);
  const int iy0 = static_cast<int>(fy);

  // For a hexagonal lattice the Voronoi cell of a centre is the hexagon itself,
  // so the nearest centre is the containing wafer.
  double best = std::numeric_limits<double>::max();
  int bestIdx = -1;
  for (int iy = std::max(0, iy0 - 1); iy <= std::min(g.ny - 1, iy0 + 1); ++iy) {
    for (int ix = std::max(0, ix0 - 1); ix <= std::min(g.nx - 1, ix0 + 1); ++ix) {
      const std::size_t b = static_cast<std::size_t>(iy) * g.nx + ix;
      for (uint32_t j = g.bucketOffset[b]; j < g.bucketOffset[b + 1]; ++j) {
        const Wafer& w = wafers_[g.bucketWafer[j]];
        const double dx = x - w.x, dy = y - w.y;
        const double d2 = dx * dx + dy * dy;
        if (d2 < best) {
          best = d2;
          bestIdx = static_cast<int>(g.bucketWafer[j]);
        }
      }
    }
  }
  if (bestIdx < 0)
    return -1;
  u = wafers_[bestIdx].u;
  v = wafers_[bestIdx].v;
  return bestIdx;
}

DetId HGCalFastGeometry::toDetId(int layer, double x, double y, int zside) const {
  if (layer < firstLayer_ || layer > lastLayer_)
    return DetId();
  const int iz = (zside >= 0) ? 1 : -1;

  // FullSim (HGCalNumberingScheme::getUnitID) mirrors x for the -z endcap
  // before doing any geometry lookup; mirror here too so both the wafer search
  // and waferFromPosition() see the same frame.
  const double xx = iz * x;

  if (!silicon_) {
    const double z = iz * layerZ(layer);
    const std::array<int, 3> id =
        ddd_.assignCellTrap(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z), layer, true);
    if (id[2] < 0)
      return DetId();
    // Same id construction as SimG4CMS/Calo HGCalNumberingScheme: tile type,
    // SiPM size and granularity come from the DDD constants, not the cell finder.
    HGCScintillatorDetId detId(id[2], layer, iz * id[0], id[1], false, 0);
    const std::pair<int, int> typm = ddd_.tileType(layer, id[0], 0);
    if (typm.first >= 0) {
      detId.setType(typm.first);
      detId.setSiPM(typm.second);
    }
    detId.setGranularity(ddd_.tileGranularity(layer));
    return detId;
  }

  const LayerGrid& g = grids_[layer - firstLayer_];
  int waferU = 0, waferV = 0;
  if (nearestWafer(g, xx, y, waferU, waferV) < 0)
    return DetId();

  // Handing the known wafer in makes waferFromPosition() take its cheap
  // "wafer already identified" branch: no hex-containment scan, just the
  // analytic cell inversion -- the identical code path FullSim uses.
  int cellU = 0, cellV = 0, celltype = 0;
  double wt = 1.0;
  ddd_.waferFromPosition(HGCalParameters::k_ScaleToDDD * xx,
                         HGCalParameters::k_ScaleToDDD * y,
                         iz,
                         layer,
                         waferU,
                         waferV,
                         cellU,
                         cellV,
                         celltype,
                         wt,
                         false,
                         false);
  if (celltype < 0)
    return DetId();
  if (!ddd_.isValidHex8(layer, waferU, waferV, cellU, cellV, true))
    return DetId();

  return HGCSiliconDetId(det_, iz, celltype, layer, waferU, waferV, cellU, cellV);
}

double HGCalFastGeometry::layerZ(int layer) const {
  if (layer < firstLayer_ || layer > lastLayer_)
    return 0.;
  return layerZ_[layer - firstLayer_];
}

std::pair<double, double> HGCalFastGeometry::layerRadii(int layer) const {
  if (layer < firstLayer_ || layer > lastLayer_)
    return std::make_pair(0., 0.);
  return layerR_[layer - firstLayer_];
}

int HGCalFastGeometry::layerOf(double z) const {
  if (layerZ_.empty())
    return -1;
  // getLayer() clamps to the first/last layer, so the subdetector boundary is
  // applied here: half a layer spacing in front of the first and behind the last.
  const double az = std::abs(z);
  const std::size_t n = layerZ_.size();
  const double front = (n > 1) ? 0.5 * (layerZ_[1] - layerZ_[0]) : 0.;
  const double back = (n > 1) ? 0.5 * (layerZ_[n - 1] - layerZ_[n - 2]) : 0.;
  if (az < layerZ_.front() - front || az > layerZ_.back() + back)
    return -1;
  const int lay = ddd_.getLayer(az, true);
  return (lay < firstLayer_ || lay > lastLayer_) ? -1 : lay;
}

int HGCalFastGeometry::siThicknessType(const DetId& id) {
  if (id.det() == DetId::HGCalHSc)
    return -1;
  return HGCSiliconDetId(id).type();
}

double HGCalFastGeometry::simWeight(const DetId& id) const {
  if (!silicon_ || id.det() == DetId::HGCalHSc)
    return 1.0;
  const HGCalParameters* p = ddd_.getParameter();
  const int t = HGCSiliconDetId(id).type();
  if (t != HGCSiliconDetId::HGCalHD120 || p->useSimWt_ <= 0)
    return 1.0;
  if (t >= static_cast<int>(p->cellThickness_.size()) || p->waferThick_ <= 0.)
    return 1.0;
  return p->cellThickness_[t] / p->waferThick_;
}
