#include "Geometry/HGCalCommonData/interface/HGCalWafer.h"
#include <vector>

HGCalWafer::HGCalWafer(double waferSize, int32_t nFine, int32_t nCoarse) {
  N_[0] = nFine;
  N_[1] = nCoarse;
  for (int k = 0; k < 2; ++k) {
    R_[k] = waferSize / (3 * N_[k]);
    r_[k] = sqrt3By2_ * R_[k];
  }
}

std::pair<double, double> HGCalWafer::HGCalWaferUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  switch (placementIndex) {
    case (HGCalWafer::WaferPlacementIndex6):
      x = (1.5 * (v - u) + 0.5) * R_[type];
      y = (v + u - 2 * N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex7):
      x = (1.5 * (v - N_[type]) + 1.0) * R_[type];
      y = (2 * u - v - N_[type]) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex8):
      x = (1.5 * (u - N_[type]) + 0.5) * R_[type];
      y = -(2 * v - u - N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex9):
      x = -(1.5 * (v - u) + 0.5) * R_[type];
      y = -(v + u - 2 * N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex10):
      x = -(1.5 * (v - N_[type]) + 1) * R_[type];
      y = -(2 * u - v - N_[type]) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex11):
      x = -(1.5 * (u - N_[type]) + 0.5) * R_[type];
      y = (2 * v - u - N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex0):
      x = (1.5 * (u - v) + 0.5) * R_[type];
      y = (v + u - 2 * N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex1):
      x = (1.5 * (v - N_[type]) + 0.5) * R_[type];
      y = -(2 * v - u - N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex2):
      x = -(1.5 * (u - N_[type]) + 1) * R_[type];
      y = -(2 * v - u - N_[type]) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex3):
      x = -(1.5 * (u - v) + 0.5) * R_[type];
      y = -(v + u - 2 * N_[type] + 1) * r_[type];
      break;
    case (HGCalWafer::WaferPlacementIndex4):
      x = (1.5 * (u - N_[type]) + 0.5) * R_[type];
      y = -(2 * u - v - N_[type] + 1) * r_[type];
      break;
    default:
      x = (1.5 * (u - N_[type]) + 1) * R_[type];
      y = (2 * v - u - N_[type]) * r_[type];
      break;
  }
  return std::make_pair(x, y);
}

std::pair<double, double> HGCalWafer::HGCalWaferUV2XY2(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  double x(0), y(0);
  if (placementIndex < HGCalWafer::WaferPlacementExtra) {
    double x0 = (1.5 * (u - v) - 0.5) * R_[type];
    double y0 = (u + v - 2 * N_[type] + 1) * r_[type];
    const std::vector<double> fac1 = {1.0, 0.5, -0.5, -1.0, -0.5, 0.5};
    const std::vector<double> fac2 = {0.0, sqrt3By2_, sqrt3By2_, 0.0, -sqrt3By2_, -sqrt3By2_};
    x = x0 * fac1[placementIndex] - y0 * fac2[placementIndex];
    y = x0 * fac2[placementIndex] + y0 * fac1[placementIndex];
  } else {
    double x0 = (1.5 * (v - N_[type]) + 1.0) * R_[type];
    double y0 = (2 * u - v - N_[type]) * r_[type];
    const std::vector<double> fac1 = {0.5, 1.0, 0.5, -0.5, -1.0, -0.5};
    const std::vector<double> fac2 = {sqrt3By2_, 0.0, -sqrt3By2_, -sqrt3By2_, 0.0, sqrt3By2_};
    x = x0 * fac1[placementIndex - HGCalWafer::WaferPlacementExtra] - y0 * fac2[placementIndex - HGCalWafer::WaferPlacementExtra];
    y = x0 * fac2[placementIndex - HGCalWafer::WaferPlacementExtra] + y0 * fac1[placementIndex - HGCalWafer::WaferPlacementExtra];
  }
  return std::make_pair(x, y);
}

int HGCalWafer::HGCalWaferUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type) {
  if (type != 0)
    type = 1;
  int cell(0), cellx(0);
  if (placementIndex < HGCalWafer::WaferPlacementExtra) {
    const std::vector<int> itype0 = {0, 6, 7, 8, 9, 10, 11, 3, 4, 5, 3, 4, 5};
    const std::vector<int> itype1 = {0, 0, 1, 2, 3, 4, 5, 0, 1, 2, 0, 1, 2};
    const std::vector<int> itype2 = {0, 10, 11, 6, 7, 8, 9, 5, 3, 4, 5, 3, 4};
    const std::vector<int> itype3 = {0, 4, 5, 0, 1, 2, 3, 2, 0, 1, 2, 0, 1};
    const std::vector<int> itype4 = {0, 8, 9, 10, 11, 6, 7, 4, 5, 3, 4, 5, 3};
    const std::vector<int> itype5 = {0, 2, 3, 4, 5, 0, 1, 1, 2, 0, 1, 2, 0};
    if (u == 0 && v == 0)
      cellx = 1;
    else if (u == 0 && (v - u) == (2 * N_[type] - 1))
      cellx = 2;
    else if ((v - u) == (N_[type] - 1) && v == (2 * N_[type] - 1))
      cellx = 2;
    else if (u == (2 * N_[type] - 1) && v == (2 * N_[type] - 1))
      cellx = 3;
    else if (u == (2 * N_[type] - 1) && (u - v) == N_[type])
      cellx = 4;
    else if ((u - v) == N_[type] && v == 0)
      cellx = 5;
    else if (u == 0)
      cellx = 6;
    else if ((v - u) == (N_[type] - 1))
      cellx = 7;
    else if (v == (2 * N_[type] - 1))
      cellx = 8;
    else if (u == (2 * N_[type] - 1))
      cellx = 9;
    else if ((u - v) == N_[type])
      cellx = 10;
    else if (v == 0)
      cellx = 11;
    switch (placementIndex) {
      case (HGCalWafer::WaferPlacementIndex6):
        cell = itype0[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex7):
        cell = itype1[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex8):
        cell = itype2[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex9):
        cell = itype3[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex10):
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
    if (u == 0 && v == 0)
      cellx = 1;
    else if (v == 0 && (u - v) == (N_[type] - 1))
      cellx = 2;
    else if ((u - v) == (N_[type] - 1) && u == (2 * N_[type] - 1))
      cellx = 2;
    else if (u == (2 * N_[type] - 1) && v == (2 * N_[type] - 1))
      cellx = 3;
    else if (v == (2 * N_[type] - 1) && (v - u) == N_[type])
      cellx = 4;
    else if ((v - u) == N_[type] && u == 0)
      cellx = 5;
    else if (v == 0)
      cellx = 6;
    else if ((u - v) == (N_[type] - 1))
      cellx = 7;
    else if (u == (2 * N_[type] - 1))
      cellx = 8;
    else if (v == (2 * N_[type] - 1))
      cellx = 9;
    else if ((v - u) == N_[type])
      cellx = 10;
    else if (u == 0)
      cellx = 11;
    switch (placementIndex) {
      case (HGCalWafer::WaferPlacementIndex0):
        cell = itype0[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex1):
        cell = itype1[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex2):
        cell = itype2[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex3):
        cell = itype3[cellx];
        break;
      case (HGCalWafer::WaferPlacementIndex4):
        cell = itype4[cellx];
        break;
      default:
        cell = itype5[cellx];
        break;
    }
  }
  return cell;
}

int HGCalWafer::HGCalWaferPlacementIndex(int32_t iz, int32_t fwdBack, int32_t orient) {
  int32_t indx = ((iz * fwdBack) > 0) ? orient : (orient + HGCalWafer::WaferPlacementExtra);
  return indx;
}
