#include "Geometry/HGCalCommonData/interface/HGCalWafer.h"
#include <vector>

HGCalWafer::HGCalWafer(double waferSize, int32_t nFine, int32_t nCoarse) {
  ncell_[0] = nFine;
  ncell_[1] = nCoarse;
  for (int k = 0; k < 2; ++k) {
    cellX_[k] = waferSize / (3 * ncell_[k]);
    cellY_[k] = sqrt3By2_ * cellX_[k];
  }
}

std::pair<double, double> HGCalWafer::HGCalWaferUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  switch (placementIndex) {
    case (HGCalWafer::waferPlacementIndex6):
      x = (1.5 * (v - u) + 0.5) * cellX_[type];
      y = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex7):
      x = (1.5 * (v - ncell_[type]) + 1.0) * cellX_[type];
      y = (2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex8):
      x = (1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = -(2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex9):
      x = -(1.5 * (v - u) + 0.5) * cellX_[type];
      y = -(v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex10):
      x = -(1.5 * (v - ncell_[type]) + 1) * cellX_[type];
      y = -(2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex11):
      x = -(1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = (2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex0):
      x = (1.5 * (u - v) - 0.5) * cellX_[type];
      y = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex1):
      x = -(1.5 * (v - ncell_[type]) + 1.0) * cellX_[type];
      y = (2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex2):
      x = -(1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = -(2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex3):
      x = -(1.5 * (u - v) - 0.5) * cellX_[type];
      y = -(v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalWafer::waferPlacementIndex4):
      x = (1.5 * (v - ncell_[type]) + 1) * cellX_[type];
      y = -(2 * u - v - ncell_[type]) * cellY_[type];
      break;
    default:
      x = (1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = (2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
  }
  return std::make_pair(x, y);
}

std::pair<double, double> HGCalWafer::HGCalWaferUV2XY2(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  if (placementIndex < HGCalWafer::waferPlacementExtra) {
    double x0 = (1.5 * (u - v) - 0.5) * cellX_[type];
    double y0 = (u + v - 2 * ncell_[type] + 1) * cellY_[type];
    const std::vector<double> fcos = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
    const std::vector<double> fsin = {0.0, sqrt3By2_, sqrt3By2_, 0.0, -sqrt3By2_, -sqrt3By2_};
    x = x0 * fcos[placementIndex] - y0 * fsin[placementIndex];
    y = x0 * fsin[placementIndex] + y0 * fcos[placementIndex];
  } else {
    double x0 = (1.5 * (v - ncell_[type]) + 1.0) * cellX_[type];
    double y0 = (2 * u - v - ncell_[type]) * cellY_[type];
    const std::vector<double> fcos = {0.5, 1.0, 0.5, -0.5, -1.0, -0.5};
    const std::vector<double> fsin = {sqrt3By2_, 0.0, -sqrt3By2_, -sqrt3By2_, 0.0, sqrt3By2_};
    x = x0 * fcos[placementIndex - HGCalWafer::waferPlacementExtra] -
        y0 * fsin[placementIndex - HGCalWafer::waferPlacementExtra];
    y = x0 * fsin[placementIndex - HGCalWafer::waferPlacementExtra] +
        y0 * fcos[placementIndex - HGCalWafer::waferPlacementExtra];
  }
  return std::make_pair(x, y);
}

std::pair<int, int> HGCalWafer::HGCalWaferUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  int cell(0), cellx(0), cellt(HGCalWafer::cornerCell);
  if (placementIndex >= HGCalWafer::waferPlacementExtra) {
    const std::vector<int> itype0 = {0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 3, 4, 5};
    const std::vector<int> itype1 = {0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2};
    const std::vector<int> itype2 = {0, 10, 11, 6, 7, 8, 9, 5, 3, 4, 5, 3, 4};
    const std::vector<int> itype3 = {0, 4, 5, 0, 1, 2, 3, 2, 0, 1, 2, 0, 1};
    const std::vector<int> itype4 = {0, 8, 9, 10, 11, 6, 7, 4, 5, 3, 4, 5, 3};
    const std::vector<int> itype5 = {0, 2, 3, 4, 5, 0, 1, 1, 2, 0, 1, 2, 0};
    if (u == 0 && v == 0) {
      cellx = 0;
    } else if (u == 0 && (v - u) == (ncell_[type] - 1)) {
      cellx = 1;
    } else if ((v - u) == (ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 2;
    } else if (u == (2 * ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 3;
    } else if (u == (2 * ncell_[type] - 1) && (u - v) == ncell_[type]) {
      cellx = 4;
    } else if ((u - v) == ncell_[type] && v == 0) {
      cellx = 5;
    } else if (u == 0) {
      cellx = 6;
      cellt = HGCalWafer::truncatedCell;
    } else if ((v - u) == (ncell_[type] - 1)) {
      cellx = 9;
      cellt = HGCalWafer::extendedCell;
    } else if (v == (2 * ncell_[type] - 1)) {
      cellx = 7;
      cellt = HGCalWafer::truncatedCell;
    } else if (u == (2 * ncell_[type] - 1)) {
      cellx = 10;
      cellt = HGCalWafer::extendedCell;
    } else if ((u - v) == ncell_[type]) {
      cellx = 8;
      cellt = HGCalWafer::truncatedCell;
    } else if (v == 0) {
      cellx = 11;
      cellt = HGCalWafer::extendedCell;
    }
    switch (placementIndex) {
      case (HGCalWafer::waferPlacementIndex6):
        cell = itype0[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex7):
        cell = itype1[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex8):
        cell = itype2[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex9):
        cell = itype3[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex10):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  } else {
    const std::vector<int> itype0 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 0, 1, 2};
    const std::vector<int> itype1 = {0, 7, 8, 9, 10, 11, 6, 4, 5, 3, 3, 4, 5};
    const std::vector<int> itype2 = {0, 3, 4, 5, 0, 1, 2, 2, 0, 1, 1, 2, 0};
    const std::vector<int> itype3 = {0, 9, 10, 11, 6, 7, 8, 5, 3, 4, 4, 5, 3};
    const std::vector<int> itype4 = {0, 5, 0, 1, 2, 3, 4, 0, 1, 2, 2, 0, 1};
    const std::vector<int> itype5 = {0, 11, 6, 7, 8, 9, 10, 3, 4, 5, 3, 4, 5};
    if (u == 0 && v == 0) {
      cellx = 0;
    } else if (v == 0 && (u - v) == (ncell_[type])) {
      cellx = 1;
    } else if ((u - v) == (ncell_[type]) && u == (2 * ncell_[type] - 1)) {
      cellx = 2;
    } else if (u == (2 * ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 3;
    } else if (v == (2 * ncell_[type] - 1) && (v - u) == (ncell_[type] - 1)) {
      cellx = 4;
    } else if ((v - u) == (ncell_[type] - 1) && u == 0) {
      cellx = 5;
    } else if (v == 0) {
      cellx = 9;
      cellt = HGCalWafer::extendedCell;
    } else if ((u - v) == ncell_[type]) {
      cellx = 6;
      cellt = HGCalWafer::truncatedCell;
    } else if (u == (2 * ncell_[type] - 1)) {
      cellx = 10;
      cellt = HGCalWafer::extendedCell;
    } else if (v == (2 * ncell_[type] - 1)) {
      cellx = 7;
      cellt = HGCalWafer::truncatedCell;
    } else if ((v - u) == (ncell_[type] - 1)) {
      cellx = 11;
      cellt = HGCalWafer::extendedCell;
    } else if (u == 0) {
      cellx = 8;
      cellt = HGCalWafer::truncatedCell;
    }
    switch (placementIndex) {
      case (HGCalWafer::waferPlacementIndex0):
        cell = itype0[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex1):
        cell = itype1[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex2):
        cell = itype2[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex3):
        cell = itype3[cellx];
        break;
      case (HGCalWafer::waferPlacementIndex4):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  }
  return std::make_pair(cell, cellt);
}

int HGCalWafer::HGCalWaferPlacementIndex(int32_t iz, int32_t fwdBack, int32_t orient) {
  int32_t indx = ((iz * fwdBack) > 0) ? orient : (orient + HGCalWafer::waferPlacementExtra);
  return indx;
}
