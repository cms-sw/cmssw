#include "Geometry/HGCalCommonData/interface/HGCalCellUV.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <iostream>
#include <array>
#include <algorithm>
#include <cassert>

HGCalCellUV::HGCalCellUV(double waferSize, double separation, int32_t nFine, int32_t nCoarse) {
  HGCalCell hgcalcell(waferSize, nFine, nCoarse);
  assert(nFine > 0 && nCoarse > 0);
  ncell_[0] = nFine;
  ncell_[1] = nCoarse;
  for (int k = 0; k < 2; ++k) {
    cellX_[k] = waferSize / (3 * ncell_[k]);
    cellY_[k] = 0.5 * sqrt3_ * cellX_[k];
    cellXTotal_[k] = (waferSize + separation) / (3 * ncell_[k]);
    cellY_[k] = 0.5 * sqrt3_ * cellXTotal_[k];
  }

  // Fill up the placement vectors
  for (int placement = 0; placement < HGCalCell::cellPlacementTotal; ++placement) {
    // Fine cells
    for (int iu = 0; iu < 2 * ncell_[0]; ++iu) {
      for (int iv = 0; iv < 2 * ncell_[0]; ++iv) {
        int u = (placement < HGCalCell::cellPlacementExtra) ? iv : iu;
        int v = (placement < HGCalCell::cellPlacementExtra) ? iu : iv;
        if (((v - u) < ncell_[0]) && (u - v) <= ncell_[0]) {
          cellPosFine_[placement][std::pair<int, int>(u, v)] = hgcalcell.cellUV2XY1(u, v, placement, 0);
        }
      }
    }
    // Coarse cells
    for (int iu = 0; iu < 2 * ncell_[1]; ++iu) {
      for (int iv = 0; iv < 2 * ncell_[1]; ++iv) {
        int u = (placement < HGCalCell::cellPlacementExtra) ? iv : iu;
        int v = (placement < HGCalCell::cellPlacementExtra) ? iu : iv;
        if (((v - u) < ncell_[1]) && (u - v) <= ncell_[1]) {
          cellPosCoarse_[placement][std::pair<int, int>(u, v)] = hgcalcell.cellUV2XY1(u, v, placement, 1);
        }
      }
    }
  }
}

std::pair<int32_t, int32_t> HGCalCellUV::cellUVFromXY1(
    double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const {
  //--- Reverse transform to placement=0, if placement index ≠ 6
  double xloc1 = (placement >= HGCalCell::cellPlacementExtra) ? xloc : -xloc;
  int rot = placement % HGCalCell::cellPlacementExtra;
  static constexpr std::array<double, 6> fcos = {{1.0, cos60_, -cos60_, -1.0, -cos60_, cos60_}};
  static constexpr std::array<double, 6> fsin = {{0.0, sin60_, sin60_, 0.0, -sin60_, -sin60_}};
  double x = xloc1 * fcos[rot] - yloc * fsin[rot];
  double y = xloc1 * fsin[rot] + yloc * fcos[rot];

  //--- Calculate coordinates in u,v,w system
  double u = x * sin60_ + y * cos60_;
  double v = -x * sin60_ + y * cos60_;
  double w = y;

  //--- Rounding in u, v, w coordinates
  int iu = std::floor(u / cellY_[type]) + 3 * (ncell_[type]) - 1;
  int iv = std::floor(v / cellY_[type]) + 3 * (ncell_[type]);
  int iw = std::floor(w / cellY_[type]) + 1;

  int isv = (iu + iw) / 3;
  int isu = (iv + iw) / 3;

  //--- Taking care of extending cells
  if ((iu + iw) < 0) {
    isu = (iv + iw + 1) / 3;
    isv = 0;
  } else if (isv - isu > ncell_[type] - 1) {
    isu = (iv + iw + 1) / 3;
    isv = (iu + iw - 1) / 3;
  } else if (isu > 2 * ncell_[type] - 1) {
    isu = 2 * ncell_[type] - 1;
    isv = (iu + iw - 1) / 3;
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellUVFromXY1: Input " << xloc << ":" << yloc << ":" << extend << " Output "
                                  << isu << ":" << isv;
  return std::make_pair(isu, isv);
}

std::pair<int32_t, int32_t> HGCalCellUV::cellUVFromXY2(
    double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const {
  //--- Using multiple inequalities to find (u, v)
  //--- Reverse transform to placement=0, if placement index ≠ 7
  double xloc1 = (placement >= HGCalCell::cellPlacementExtra) ? xloc : -1 * xloc;
  int rot = placement % HGCalCell::cellPlacementExtra;
  static constexpr std::array<double, 6> fcos = {{cos60_, 1.0, cos60_, -cos60_, -1.0, -cos60_}};
  static constexpr std::array<double, 6> fsin = {{-sin60_, 0.0, sin60_, sin60_, 0.0, -sin60_}};
  double x = xloc1 * fcos[rot] - yloc * fsin[rot];
  double y = xloc1 * fsin[rot] + yloc * fcos[rot];

  int32_t u(-100), v(-100);
  int ncell = (type != 0) ? ncell_[1] : ncell_[0];
  double r = (type != 0) ? cellY_[1] : cellY_[0];
  double l1 = (y / r) + ncell - 1.0;
  int l2 = std::floor((0.5 * y + 0.5 * x / sqrt3_) / r + ncell - 4.0 / 3.0);
  int l3 = std::floor((x / sqrt3_) / r + ncell - 4.0 / 3.0);
  double l4 = (y + sqrt3_ * x) / (2 * r) + 2 * ncell - 2;
  double l5 = (y - sqrt3_ * x) / (2 * r) - ncell;
  double u1 = (y / r) + ncell + 1.0;
  int u2 = std::ceil((0.5 * y + 0.5 * x / sqrt3_) / r + ncell + 2.0 / 3.0);
  int u3 = std::ceil((x / sqrt3_) / r + ncell);
  double u4 = l4 + 2;
  double u5 = l5 + 2;

  for (int ui = l2 + 1; ui < u2; ui++) {
    for (int vi = l3 + 1; vi < u3; vi++) {
      int c1 = 2 * ui - vi;
      int c4 = ui + vi;
      int c5 = ui - 2 * vi;
      if ((c1 < u1) && (c1 > l1) && (c4 < u4) && (c4 > l4) && (c5 < u5) && (c5 > l5)) {
        u = ui;
        v = vi;
      }
    }
  }

  //--- Taking care of extending cells
  if (v == -1) {
    if (y < (2 * u - v - ncell) * r) {
      v += 1;
    } else {
      u += 1;
      v += 1;
    }
  }
  if (v - u == ncell) {
    if ((y + sqrt3_ * x) < ((u + v - 2 * ncell + 1) * 2 * r)) {
      v += -1;
    } else {
      u += 1;
    }
  }
  if (u == 2 * ncell) {
    if ((y - sqrt3_ * x) < ((u - 2 * v + ncell - 1) * 2 * r)) {
      u += -1;
    } else {
      u += -1;
      v += -1;
    }
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellUVFromXY2: Input " << xloc << ":" << yloc << ":" << extend << " Output " << u
                                  << ":" << v;
  return std::make_pair(u, v);
}

std::pair<int32_t, int32_t> HGCalCellUV::cellUVFromXY3(
    double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const {
  //--- Using Cube coordinates to find the (u, v)
  //--- Reverse transform to placement=0, if placement index ≠ 6
  double xloc1 = (placement >= HGCalCell::cellPlacementExtra) ? xloc : -xloc;
  int rot = placement % HGCalCell::cellPlacementExtra;
  static constexpr std::array<double, 6> fcos = {{1.0, cos60_, -cos60_, -1.0, -cos60_, cos60_}};
  static constexpr std::array<double, 6> fsin = {{0.0, sin60_, sin60_, 0.0, -sin60_, -sin60_}};
  double xprime = xloc1 * fcos[rot] - yloc * fsin[rot];
  double yprime = xloc1 * fsin[rot] + yloc * fcos[rot];
  double x = xprime + cellX_[type];
  double y = yprime;

  x = x / cellX_[type];
  y = y / cellY_[type];

  double cu = 2 * x / 3;
  double cv = -x / 3 + y / 2;
  double cw = -x / 3 - y / 2;

  int iu = std::round(cu);
  int iv = std::round(cv);
  int iw = std::round(cw);

  if (iu + iv + iw != 0) {
    double arr[] = {std::abs(cu - iu), std::abs(cv - iv), std::abs(cw - iw)};
    int i = std::distance(arr, std::max_element(arr, arr + 3));

    if (i == 1)
      iv = (std::round(cv) == std::floor(cv)) ? std::ceil(cv) : std::floor(cv);
    else if (i == 2)
      iw = (std::round(cw) == std::floor(cw)) ? std::ceil(cw) : std::floor(cw);
  }

  //--- Taking care of extending cells
  int u(ncell_[type] + iv), v(ncell_[type] - 1 - iw);
  double xcell = (1.5 * (v - u) + 0.5) * cellX_[type];
  double ycell = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
  if (v == -1) {
    if ((yprime - sqrt3_ * xprime) < (ycell - sqrt3_ * xcell)) {
      v += 1;
    } else {
      u += 1;
      v += 1;
    }
  }
  if (v - u == ncell_[type]) {
    if (yprime < ycell) {
      v += -1;
    } else {
      u += 1;
    }
  }
  if (u == 2 * ncell_[type]) {
    if ((yprime + sqrt3_ * xprime) > (ycell + sqrt3_ * xcell)) {
      u += -1;
    } else {
      u += -1;
      v += -1;
    }
  }

  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellUVFromXY3: Input " << xloc << ":" << yloc << ":" << extend << " Output " << u
                                  << ":" << v;
  return std::make_pair(u, v);
}

std::pair<int, int> HGCalCellUV::cellUVFromXY4(
    double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) {
  if (type != 0)
    return cellUVFromXY4(
        xloc, yloc, ncell_[1], cellX_[1], cellY_[1], cellXTotal_[1], cellY_[1], cellPosCoarse_[placement], extend, debug);
  else
    return cellUVFromXY4(
        xloc, yloc, ncell_[0], cellX_[0], cellY_[0], cellXTotal_[0], cellY_[0], cellPosFine_[placement], extend, debug);
}

std::pair<int, int> HGCalCellUV::cellUVFromXY4(double xloc,
                                               double yloc,
                                               int n,
                                               double cellX,
                                               double cellY,
                                               double cellXTotal,
                                               double cellYTotal,
                                               std::map<std::pair<int, int>, std::pair<double, double> >& cellPos,
                                               bool extend,
                                               bool debug) {
  std::pair<int, int> uv = std::make_pair(-1, -1);
  std::map<std::pair<int, int>, std::pair<double, double> >::const_iterator itr;
  for (itr = cellPos.begin(); itr != cellPos.end(); ++itr) {
    double delX = std::abs(xloc - (itr->second).first);
    double delY = std::abs(yloc - (itr->second).second);
    if ((delX < cellX) && (delY < cellY)) {
      if ((delX < (0.5 * cellX)) || (delY < (2.0 * cellY - sqrt3_ * delX))) {
        uv = itr->first;
        break;
      }
    }
  }
  if ((uv.first < 0) && extend) {
    for (itr = cellPos.begin(); itr != cellPos.end(); ++itr) {
      double delX = std::abs(xloc - (itr->second).first);
      double delY = std::abs(yloc - (itr->second).second);
      if ((delX < cellXTotal) && (delY < cellYTotal)) {
        if ((delX < (0.5 * cellXTotal)) || (delY < (2.0 * cellYTotal - sqrt3_ * delX))) {
          uv = itr->first;
          break;
        }
      }
    }
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellUVFromXY4: Input " << xloc << ":" << yloc << ":" << extend << " Output "
                                  << uv.first << ":" << uv.second;
  return uv;
}

std::pair<int32_t, int32_t> HGCalCellUV::cellUVFromXY1(
    double xloc, double yloc, int32_t placement, int32_t type, int32_t partial, bool extend, bool debug) const {
  std::pair<int, int> uv = HGCalCellUV::cellUVFromXY1(xloc, yloc, placement, type, extend, debug);
  int u = uv.first;
  int v = uv.second;
  if (partial == HGCalTypes::WaferLDTop) {
    if (u * HGCalTypes::edgeWaferLDTop[0] + v * HGCalTypes::edgeWaferLDTop[1] == HGCalTypes::edgeWaferLDTop[2] + 1) {
      double xloc1 = (placement >= HGCalCell::cellPlacementExtra) ? xloc : -xloc;
      int rot = placement % HGCalCell::cellPlacementExtra;
      static constexpr std::array<double, 6> fcos = {{1.0, cos60_, -cos60_, -1.0, -cos60_, cos60_}};
      static constexpr std::array<double, 6> fsin = {{0.0, sin60_, sin60_, 0.0, -sin60_, -sin60_}};
      double xprime = -1 * (xloc1 * fcos[rot] - yloc * fsin[rot]);
      double yprime = xloc1 * fsin[rot] + yloc * fcos[rot];
      double xcell = -1 * (1.5 * (v - u) + 0.5) * cellX_[type];
      double ycell = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      if ((yprime - sqrt3_ * xprime) > (ycell - sqrt3_ * xcell)) {
        u += -1;
      } else {
        u += -1;
        v += -1;
      }
    }
  } else if (partial == HGCalTypes::WaferHDBottom) {
    if (u * HGCalTypes::edgeWaferHDBottom[0] + v * HGCalTypes::edgeWaferHDBottom[1] ==
        HGCalTypes::edgeWaferHDBottom[2] + 1) {
      double xloc1 = (placement >= HGCalCell::cellPlacementExtra) ? xloc : -xloc;
      int rot = placement % HGCalCell::cellPlacementExtra;
      static constexpr std::array<double, 6> fcos = {{1.0, cos60_, -cos60_, -1.0, -cos60_, cos60_}};
      static constexpr std::array<double, 6> fsin = {{0.0, sin60_, sin60_, 0.0, -sin60_, -sin60_}};
      double xprime = -1 * (xloc1 * fcos[rot] - yloc * fsin[rot]);
      double yprime = xloc1 * fsin[rot] + yloc * fcos[rot];
      double xcell = -1 * (1.5 * (v - u) + 0.5) * cellX_[type];
      double ycell = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      if ((yprime - sqrt3_ * xprime) > (ycell - sqrt3_ * xcell)) {
        u += 1;
        v += 1;
      } else {
        u += 1;
      }
    }
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "cellUVFromXY5: Input " << xloc << ":" << yloc << ":" << extend << " Output " << u
                                  << ":" << v;
  return std::make_pair(u, v);
}
