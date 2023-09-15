#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"
#include <vector>

//#define EDM_ML_DEBUG

HGCalCell::HGCalCell(double waferSize, int32_t nFine, int32_t nCoarse) {
  ncell_[0] = nFine;
  ncell_[1] = nCoarse;
  for (int k = 0; k < 2; ++k) {
    cellX_[k] = waferSize / (3 * ncell_[k]);
    cellY_[k] = sqrt3By2_ * cellX_[k];
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "HGCalCell initialized with waferSize " << waferSize << " number of cells " << nFine
                                << ":" << nCoarse;
#endif
}

std::pair<double, double> HGCalCell::cellUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  switch (placementIndex) {
    case (HGCalCell::cellPlacementIndex6):
      x = (1.5 * (v - u) + 0.5) * cellX_[type];
      y = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex7):
      x = (1.5 * (v - ncell_[type]) + 1.0) * cellX_[type];
      y = (2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex8):
      x = (1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = -(2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex9):
      x = -(1.5 * (v - u) + 0.5) * cellX_[type];
      y = -(v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex10):
      x = -(1.5 * (v - ncell_[type]) + 1) * cellX_[type];
      y = -(2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex11):
      x = -(1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = (2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex0):
      x = (1.5 * (u - v) - 0.5) * cellX_[type];
      y = (v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex1):
      x = -(1.5 * (v - ncell_[type]) + 1.0) * cellX_[type];
      y = (2 * u - v - ncell_[type]) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex2):
      x = -(1.5 * (u - ncell_[type]) + 0.5) * cellX_[type];
      y = -(2 * v - u - ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex3):
      x = -(1.5 * (u - v) - 0.5) * cellX_[type];
      y = -(v + u - 2 * ncell_[type] + 1) * cellY_[type];
      break;
    case (HGCalCell::cellPlacementIndex4):
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

std::pair<double, double> HGCalCell::cellUV2XY2(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  if (placementIndex < HGCalCell::cellPlacementExtra) {
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
    x = x0 * fcos[placementIndex - HGCalCell::cellPlacementExtra] -
        y0 * fsin[placementIndex - HGCalCell::cellPlacementExtra];
    y = x0 * fsin[placementIndex - HGCalCell::cellPlacementExtra] +
        y0 * fcos[placementIndex - HGCalCell::cellPlacementExtra];
  }
  return std::make_pair(x, y);
}

std::pair<int, int> HGCalCell::cellUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  int cell(0), cellx(0), cellt(HGCalCell::fullCell);
  if (placementIndex >= HGCalCell::cellPlacementExtra) {
    const std::vector<int> itype0 = {0, 7, 8, 9, 10, 11, 6, 3, 4, 5, 4, 5, 3};
    const std::vector<int> itype1 = {0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2};
    const std::vector<int> itype2 = {0, 11, 6, 7, 8, 9, 10, 5, 3, 4, 3, 4, 5};
    const std::vector<int> itype3 = {0, 4, 5, 0, 1, 2, 3, 2, 0, 1, 2, 0, 1};
    const std::vector<int> itype4 = {0, 9, 10, 11, 6, 7, 8, 4, 5, 3, 5, 3, 4};
    const std::vector<int> itype5 = {0, 2, 3, 4, 5, 0, 1, 1, 2, 0, 1, 2, 0};
    if (u == 0 && v == 0) {
      cellx = 1;
      cellt = HGCalCell::cornerCell;
    } else if (u == 0 && (v - u) == (ncell_[type] - 1)) {
      cellx = 2;
      cellt = HGCalCell::cornerCell;
    } else if ((v - u) == (ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 3;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 4;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell_[type] - 1) && (u - v) == ncell_[type]) {
      cellx = 5;
      cellt = HGCalCell::cornerCell;
    } else if ((u - v) == ncell_[type] && v == 0) {
      cellx = 6;
      cellt = HGCalCell::cornerCell;
    } else if (u == 0) {
      cellx = 7;
      cellt = HGCalCell::truncatedCell;
    } else if ((v - u) == (ncell_[type] - 1)) {
      cellx = 10;
      cellt = HGCalCell::extendedCell;
    } else if (v == (2 * ncell_[type] - 1)) {
      cellx = 8;
      cellt = HGCalCell::truncatedCell;
    } else if (u == (2 * ncell_[type] - 1)) {
      cellx = 11;
      cellt = HGCalCell::extendedCell;
    } else if ((u - v) == ncell_[type]) {
      cellx = 9;
      cellt = HGCalCell::truncatedCell;
    } else if (v == 0) {
      cellx = 12;
      cellt = HGCalCell::extendedCell;
    }
    switch (placementIndex) {
      case (HGCalCell::cellPlacementIndex6):
        cell = itype0[cellx];
        break;
      case (HGCalCell::cellPlacementIndex7):
        cell = itype1[cellx];
        break;
      case (HGCalCell::cellPlacementIndex8):
        cell = itype2[cellx];
        break;
      case (HGCalCell::cellPlacementIndex9):
        cell = itype3[cellx];
        break;
      case (HGCalCell::cellPlacementIndex10):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  } else {
    const std::vector<int> itype0 = {0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 0, 1, 2};
    const std::vector<int> itype1 = {0, 8, 9, 10, 11, 6, 7, 4, 5, 3, 4, 5, 3};
    const std::vector<int> itype2 = {0, 3, 4, 5, 0, 1, 2, 2, 0, 1, 1, 2, 0};
    const std::vector<int> itype3 = {0, 10, 11, 6, 7, 8, 9, 5, 3, 4, 5, 3, 4};
    const std::vector<int> itype4 = {0, 5, 0, 1, 2, 3, 4, 0, 1, 2, 2, 0, 1};
    const std::vector<int> itype5 = {0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 3, 4, 5};
    if (u == 0 && v == 0) {
      cellx = 1;
      cellt = HGCalCell::cornerCell;
    } else if (v == 0 && (u - v) == (ncell_[type])) {
      cellx = 2;
      cellt = HGCalCell::cornerCell;
    } else if ((u - v) == (ncell_[type]) && u == (2 * ncell_[type] - 1)) {
      cellx = 3;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell_[type] - 1) && v == (2 * ncell_[type] - 1)) {
      cellx = 4;
      cellt = HGCalCell::cornerCell;
    } else if (v == (2 * ncell_[type] - 1) && (v - u) == (ncell_[type] - 1)) {
      cellx = 5;
      cellt = HGCalCell::cornerCell;
    } else if ((v - u) == (ncell_[type] - 1) && u == 0) {
      cellx = 6;
      cellt = HGCalCell::cornerCell;
    } else if (v == 0) {
      cellx = 10;
      cellt = HGCalCell::extendedCell;
    } else if ((u - v) == ncell_[type]) {
      cellx = 7;
      cellt = HGCalCell::truncatedCell;
    } else if (u == (2 * ncell_[type] - 1)) {
      cellx = 11;
      cellt = HGCalCell::extendedCell;
    } else if (v == (2 * ncell_[type] - 1)) {
      cellx = 8;
      cellt = HGCalCell::truncatedCell;
    } else if ((v - u) == (ncell_[type] - 1)) {
      cellx = 12;
      cellt = HGCalCell::extendedCell;
    } else if (u == 0) {
      cellx = 9;
      cellt = HGCalCell::truncatedCell;
    }
    switch (placementIndex) {
      case (HGCalCell::cellPlacementIndex0):
        cell = itype0[cellx];
        break;
      case (HGCalCell::cellPlacementIndex1):
        cell = itype1[cellx];
        break;
      case (HGCalCell::cellPlacementIndex2):
        cell = itype2[cellx];
        break;
      case (HGCalCell::cellPlacementIndex3):
        cell = itype3[cellx];
        break;
      case (HGCalCell::cellPlacementIndex4):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  }
  return std::make_pair(cell, cellt);
}

int HGCalCell::cellPlacementIndex(int32_t iz, int32_t frontBack, int32_t orient) {
  int32_t indx = ((iz * frontBack) > 0) ? orient : (orient + HGCalCell::cellPlacementExtra);
  return indx;
}

std::pair<int32_t, int32_t> HGCalCell::cellOrient(int32_t placementIndex) {
  int32_t orient = (placementIndex >= HGCalCell::cellPlacementExtra) ? (placementIndex - HGCalCell::cellPlacementExtra)
                                                                     : placementIndex;
  int32_t frontBackZside = (placementIndex >= HGCalCell::cellPlacementExtra) ? 1 : -1;
  return std::make_pair(orient, frontBackZside);
}

std::pair<int32_t, int32_t> HGCalCell::cellType(int32_t u, int32_t v, int32_t ncell, int32_t placementIndex) {
  int cell(0), cellx(0), cellt(HGCalCell::fullCell);
  if (placementIndex >= HGCalCell::cellPlacementExtra) {
    const std::vector<int> itype0 = {HGCalCell::centralCell,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge};
    const std::vector<int> itype1 = {HGCalCell::centralCell,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge};
    const std::vector<int> itype2 = {HGCalCell::centralCell,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge};
    const std::vector<int> itype3 = {HGCalCell::centralCell,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge};
    const std::vector<int> itype4 = {HGCalCell::centralCell,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge};
    const std::vector<int> itype5 = {HGCalCell::centralCell,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge};
    if (u == 0 && v == 0) {
      cellx = 1;
      cellt = HGCalCell::cornerCell;
    } else if (u == 0 && (v - u) == (ncell - 1)) {
      cellx = 2;
      cellt = HGCalCell::cornerCell;
    } else if ((v - u) == (ncell - 1) && v == (2 * ncell - 1)) {
      cellx = 3;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell - 1) && v == (2 * ncell - 1)) {
      cellx = 4;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell - 1) && (u - v) == ncell) {
      cellx = 5;
      cellt = HGCalCell::cornerCell;
    } else if ((u - v) == ncell && v == 0) {
      cellx = 6;
      cellt = HGCalCell::cornerCell;
    } else if (u == 0) {
      cellx = 7;
      cellt = HGCalCell::truncatedCell;
    } else if ((v - u) == (ncell - 1)) {
      cellx = 8;
      cellt = HGCalCell::extendedCell;
    } else if (v == (2 * ncell - 1)) {
      cellx = 9;
      cellt = HGCalCell::truncatedCell;
    } else if (u == (2 * ncell - 1)) {
      cellx = 10;
      cellt = HGCalCell::extendedCell;
    } else if ((u - v) == ncell) {
      cellx = 11;
      cellt = HGCalCell::truncatedCell;
    } else if (v == 0) {
      cellx = 12;
      cellt = HGCalCell::extendedCell;
    }
    switch (placementIndex) {
      case (HGCalCell::cellPlacementIndex6):
        cell = itype0[cellx];
        break;
      case (HGCalCell::cellPlacementIndex7):
        cell = itype1[cellx];
        break;
      case (HGCalCell::cellPlacementIndex8):
        cell = itype2[cellx];
        break;
      case (HGCalCell::cellPlacementIndex9):
        cell = itype3[cellx];
        break;
      case (HGCalCell::cellPlacementIndex10):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  } else {
    const std::vector<int> itype0 = {HGCalCell::centralCell,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge};
    const std::vector<int> itype1 = {HGCalCell::centralCell,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge};
    const std::vector<int> itype2 = {HGCalCell::centralCell,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge};
    const std::vector<int> itype3 = {HGCalCell::centralCell,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge};
    const std::vector<int> itype4 = {HGCalCell::centralCell,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::leftEdge,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge};
    const std::vector<int> itype5 = {HGCalCell::centralCell,
                                     HGCalCell::bottomLeftCorner,
                                     HGCalCell::bottomCorner,
                                     HGCalCell::bottomRightCorner,
                                     HGCalCell::topRightCorner,
                                     HGCalCell::topCorner,
                                     HGCalCell::topLeftCorner,
                                     HGCalCell::bottomLeftEdge,
                                     HGCalCell::bottomRightEdge,
                                     HGCalCell::rightEdge,
                                     HGCalCell::topRightEdge,
                                     HGCalCell::topLeftEdge,
                                     HGCalCell::leftEdge};
    if (u == 0 && v == 0) {
      cellx = 1;
      cellt = HGCalCell::cornerCell;
    } else if (v == 0 && (u - v) == (ncell)) {
      cellx = 2;
      cellt = HGCalCell::cornerCell;
    } else if ((u - v) == (ncell) && u == (2 * ncell - 1)) {
      cellx = 3;
      cellt = HGCalCell::cornerCell;
    } else if (u == (2 * ncell - 1) && v == (2 * ncell - 1)) {
      cellx = 4;
      cellt = HGCalCell::cornerCell;
    } else if (v == (2 * ncell - 1) && (v - u) == (ncell - 1)) {
      cellx = 5;
      cellt = HGCalCell::cornerCell;
    } else if ((v - u) == (ncell - 1) && u == 0) {
      cellx = 6;
      cellt = HGCalCell::cornerCell;
    } else if (v == 0) {
      cellx = 7;
      cellt = HGCalCell::extendedCell;
    } else if ((u - v) == ncell) {
      cellx = 8;
      cellt = HGCalCell::truncatedCell;
    } else if (u == (2 * ncell - 1)) {
      cellx = 9;
      cellt = HGCalCell::extendedCell;
    } else if (v == (2 * ncell - 1)) {
      cellx = 10;
      cellt = HGCalCell::truncatedCell;
    } else if ((v - u) == (ncell - 1)) {
      cellx = 11;
      cellt = HGCalCell::extendedCell;
    } else if (u == 0) {
      cellx = 12;
      cellt = HGCalCell::truncatedCell;
    }
    switch (placementIndex) {
      case (HGCalCell::cellPlacementIndex0):
        cell = itype0[cellx];
        break;
      case (HGCalCell::cellPlacementIndex1):
        cell = itype1[cellx];
        break;
      case (HGCalCell::cellPlacementIndex2):
        cell = itype2[cellx];
        break;
      case (HGCalCell::cellPlacementIndex3):
        cell = itype3[cellx];
        break;
      case (HGCalCell::cellPlacementIndex4):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  }
  return std::make_pair(cell, cellt);
}
