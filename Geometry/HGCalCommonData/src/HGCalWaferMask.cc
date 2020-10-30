#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
//#define EDM_ML_DEBUG

bool HGCalWaferMask::maskCell(int u, int v, int n, int ncor, int fcor, int corners) {
  /*
Masks each cell (or not) according to its wafer and cell position (detId) and to the user needs (corners).
Each wafer has k_CornerSize corners which are defined in anti-clockwise order starting from the corner at the top, which is always #0. 'ncor' denotes the number of corners inside the physical region. 'fcor' is the defined to be the first corner that appears inside the detector's physical volume in anti-clockwise order. 
The argument 'corners' controls the types of wafers the user wants: for instance, corners=3 masks all wafers that have at least 3 corners inside the physical region. 
 */
  bool mask(false);
  if (ncor < corners) {
    mask = true;
  } else {
    if (ncor == HGCalGeomTools::k_fourCorners) {
      switch (fcor) {
        case (0): {
          mask = (v >= n);
          break;
        }
        case (1): {
          mask = (u >= n);
          break;
        }
        case (2): {
          mask = (u > v);
          break;
        }
        case (3): {
          mask = (v < n);
          break;
        }
        case (4): {
          mask = (u < n);
          break;
        }
        default: {
          mask = (u <= v);
          break;
        }
      }
    } else {
      switch (fcor) {
        case (0): {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((u > 2 * v) && (v < n));
          } else {
            mask = ((u >= n) && (v >= n) && ((u + v) > (3 * n - 2)));
          }
          break;
        }
        case (1): {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((u + v) < n);
          } else {
            mask = ((u >= n) && (u > v) && ((2 * u - v) > 2 * n));
          }
          break;
        }
        case (2): {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((u < n) && (v > u) && (v > (2 * u - 1)));
          } else {
            mask = ((u > 2 * v) && (v < n));
          }
          break;
        }
        case (3): {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((v >= u) && ((2 * v - u) > (2 * n - 2)));
          } else {
            mask = ((u + v) < n);
          }
          break;
        }
        case (4): {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((u >= n) && (v >= n) && ((u + v) > (3 * n - 2)));
          } else {
            mask = ((u < n) && (v > u) && (v > (2 * u - 1)));
          }
          break;
        }
        default: {
          if (ncor == HGCalGeomTools::k_threeCorners) {
            mask = !((u >= n) && (u > v) && ((2 * u - v) > 2 * n));
          } else {
            mask = ((v >= u) && ((2 * v - u) > (2 * n - 2)));
          }
          break;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Corners: " << ncor << ":" << fcor << " N " << n << " u " << u << " v " << v
                                << " Mask " << mask;
#endif
  return mask;
}

bool HGCalWaferMask::goodCell(int u, int v, int n, int type, int rotn) {
  bool good(false);
  int n2 = n / 2;
  int n4 = n / 4;
  switch (type) {
    case (HGCalTypes::WaferFull): {  //WaferFull
      good = true;
      break;
    }
    case (HGCalTypes::WaferFive): {  //WaferFive
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          int u2 = u / 2;
          good = ((v - u2) < n);
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((v + u) < (3 * n - 1));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int v2 = (v + 1) / 2;
          good = ((u - v2) < n);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          int u2 = (u + 1) / 2;
          good = (u2 <= v);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((v + u) >= n);
          break;
        }
        default: {
          int v2 = v / 2;
          good = (u > v2);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferChopTwo): {  //WaferChopTwo
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = (v < (3 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = (u < (3 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((u - v) <= n2);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= n2);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= n2);
          break;
        }
        default: {
          good = ((v - u) < n2);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferChopTwoM): {  //WaferChopTwoM
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = (v < (5 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = (u < (5 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((u - v) <= n4);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= (3 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= (3 * n4));
          break;
        }
        default: {
          good = ((v - u) < n4);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferHalf): {  //WaferHalf
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = (v < n);
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = (u < n);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = (v >= u);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= n);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= n);
          break;
        }
        default: {
          good = (u > v);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferSemi): {  //WaferSemi
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((u + v) < (2 * n));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((2 * u - v) < n);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((2 * v - u) >= n);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((u + v) >= (2 * n));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((2 * u - v) > n);
          break;
        }
        default: {
          good = ((2 * v - u) < n);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferThree): {  //WaferThree
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((v + u) < n);
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          int v2 = v / 2;
          good = (u <= v2);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int u2 = (u / 2);
          good = ((v - u2) >= n);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((v + u) >= (3 * n - 1));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          int v2 = ((v + 1) / 2);
          good = ((u - v2) >= n);
          break;
        }
        default: {
          int u2 = ((u + 1) / 2);
          good = (v < u2);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferSemi2): {  //WaferSemi2
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((u + v) < (3 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((2 * u - v) < n2);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) >= (3 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((u + v) > (5 * n2 - 1));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((2 * u - v) > (3 * n2));
          break;
        }
        default: {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) < n4);
          break;
        }
      }
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "u|v " << u << ":" << v << " N " << n << " type " << type << " rot " << rotn
                                << " good " << good;
#endif
  return good;
}

int HGCalWaferMask::getRotation(int zside, int type, int rotn) {
  if (rotn >= HGCalTypes::WaferCornerMax)
    rotn = HGCalTypes::WaferCorner0;
  int newrotn(rotn);
  if ((zside < 0) && (type != HGCalTypes::WaferFull)) {
    if (type == HGCalTypes::WaferFive) {  //WaferFive
      static const int rot1[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner4,
                                                           HGCalTypes::WaferCorner3,
                                                           HGCalTypes::WaferCorner2,
                                                           HGCalTypes::WaferCorner1,
                                                           HGCalTypes::WaferCorner0,
                                                           HGCalTypes::WaferCorner5};
      newrotn = rot1[rotn];
    } else if ((type == HGCalTypes::WaferThree) || (type == HGCalTypes::WaferSemi) ||
               (type == HGCalTypes::WaferSemi2)) {  //WaferThree/WaferSemi/WaferSemi2
      static const int rot2[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner2,
                                                           HGCalTypes::WaferCorner1,
                                                           HGCalTypes::WaferCorner0,
                                                           HGCalTypes::WaferCorner5,
                                                           HGCalTypes::WaferCorner4,
                                                           HGCalTypes::WaferCorner3};
      newrotn = rot2[rotn];
    } else {  //WaferHalf/WaferChopTwo/WaferChopTwoM
      static const int rot3[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner3,
                                                           HGCalTypes::WaferCorner2,
                                                           HGCalTypes::WaferCorner1,
                                                           HGCalTypes::WaferCorner0,
                                                           HGCalTypes::WaferCorner5,
                                                           HGCalTypes::WaferCorner4};
      newrotn = rot3[rotn];
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "zside " << zside << " type " << type << " rotn " << rotn << ":" << newrotn;
#endif
  return newrotn;
}

std::pair<int, int> HGCalWaferMask::getTypeMode(const double& xpos,
                                                const double& ypos,
                                                const double& delX,
                                                const double& delY,
                                                const double& rin,
                                                const double& rout,
                                                const int& wType,
                                                const int& mode,
                                                bool debug) {
  int ncor(0), iok(0);
  int type(HGCalTypes::WaferFull), rotn(HGCalTypes::WaferCorner0);

  static const int corners = 6;
  static const int base = 10;
  double rin2 = rin * rin;
  double rout2 = rout * rout;
  double dx0[corners] = {
      0.0, HGCalTypes::c10 * delX, HGCalTypes::c10 * delX, 0.0, -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX};
  double dy0[corners] = {-HGCalTypes::c10 * delY,
                         -HGCalTypes::c50 * delY,
                         HGCalTypes::c50 * delY,
                         HGCalTypes::c10 * delY,
                         HGCalTypes::c50 * delY,
                         -HGCalTypes::c50 * delY};
  double xc[corners], yc[corners];
  for (int k = 0; k < corners; ++k) {
    xc[k] = xpos + dx0[k];
    yc[k] = ypos + dy0[k];
    double rpos2 = (xc[k] * xc[k] + yc[k] * yc[k]);
    if (rpos2 <= rout2 && rpos2 >= rin2) {
      ++ncor;
      iok = iok * base + 1;
    } else {
      iok *= base;
    }
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "I/p " << xpos << ":" << ypos << ":" << delX << ":" << delY << ":" << rin << ":"
                                  << rout << ":" << wType << ":" << mode << " Corners " << ncor << " iok " << iok;

  static const int ipat5[corners] = {101111, 110111, 111011, 111101, 111110, 11111};
  static const int ipat4[corners] = {100111, 110011, 111001, 111100, 11110, 1111};
  static const int ipat3[corners] = {100011, 110001, 111000, 11100, 1110, 111};
  double dx1[corners] = {HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c50 * delX,
                         -HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX};
  double dy1[corners] = {
      -HGCalTypes::c75 * delY, 0.0, HGCalTypes::c75 * delY, HGCalTypes::c75 * delY, 0.0, -HGCalTypes::c75 * delY};
  double dx2[corners] = {HGCalTypes::c50 * delX,
                         -HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX,
                         HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX};
  double dy2[corners] = {
      HGCalTypes::c75 * delY, HGCalTypes::c75 * delY, 0.0, -HGCalTypes::c75 * delY, -HGCalTypes::c75 * delY, 0.0};
  double dx3[corners] = {
      HGCalTypes::c22 * delX, delX, HGCalTypes::c77 * delX, -HGCalTypes::c22 * delX, -delX, -HGCalTypes::c77 * delX};
  double dy3[corners] = {-HGCalTypes::c88 * delY,
                         -HGCalTypes::c27 * delY,
                         HGCalTypes::c61 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c27 * delY,
                         -HGCalTypes::c61 * delY};
  double dx4[corners] = {HGCalTypes::c22 * delX,
                         -HGCalTypes::c77 * delX,
                         -delX,
                         -HGCalTypes::c22 * delX,
                         HGCalTypes::c77 * delX,
                         HGCalTypes::c10 * delX};
  double dy4[corners] = {HGCalTypes::c88 * delY,
                         HGCalTypes::c61 * delY,
                         -HGCalTypes::c27 * delY,
                         -HGCalTypes::c88 * delY,
                         -HGCalTypes::c61 * delY,
                         HGCalTypes::c27 * delY};
  double dx5[corners] = {-HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX,
                         HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c50 * delX};
  double dy5[corners] = {
      HGCalTypes::c75 * delY, 0.0, -HGCalTypes::c75 * delY, -HGCalTypes::c75 * delY, 0.0, HGCalTypes::c75 * delY};
  double dx6[corners] = {-HGCalTypes::c77 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c22 * delX,
                         HGCalTypes::c77 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c22 * delX};
  double dy6[corners] = {HGCalTypes::c61 * delY,
                         -HGCalTypes::c27 * delY,
                         -HGCalTypes::c88 * delY,
                         -HGCalTypes::c61 * delY,
                         HGCalTypes::c27 * delY,
                         HGCalTypes::c88 * delY};

  if (ncor == HGCalGeomTools::k_allCorners) {
  } else if (ncor == HGCalGeomTools::k_fiveCorners) {
    rotn = static_cast<int>(std::find(ipat5, ipat5 + 6, iok) - ipat5);
    type = HGCalTypes::WaferFive;
  } else if (ncor == HGCalGeomTools::k_fourCorners) {
    rotn = static_cast<int>(std::find(ipat4, ipat4 + 6, iok) - ipat4);
    type = HGCalTypes::WaferHalf;
    double rpos12 = ((xpos + dx1[rotn]) * (xpos + dx1[rotn]) + (ypos + dy1[rotn]) * (ypos + dy1[rotn]));
    double rpos22(0);
    if (rpos12 <= rout2 && rpos12 >= rin2) {
      rpos22 = ((xpos + dx2[rotn]) * (xpos + dx2[rotn]) + (ypos + dy2[rotn]) * (ypos + dy2[rotn]));
      if (rpos22 <= rout2 && rpos22 >= rin2)
        type = HGCalTypes::WaferChopTwo;
    }
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "Test for Chop2 " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                    << type;
    if ((type == HGCalTypes::WaferHalf) && (wType == 0)) {
      rpos12 = ((xpos + dx3[rotn]) * (xpos + dx3[rotn]) + (ypos + dy3[rotn]) * (ypos + dy3[rotn]));
      if (rpos12 <= rout2 && rpos12 >= rin2) {
        rpos22 = ((xpos + dx4[rotn]) * (xpos + dx4[rotn]) + (ypos + dy4[rotn]) * (ypos + dy4[rotn]));
        if (rpos22 <= rout2 && rpos22 >= rin2)
          type = HGCalTypes::WaferChopTwoM;
      }
      if (debug)
        edm::LogVerbatim("HGCalGeom") << "Test for Chop2M " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                      << type;
    }
  } else if (ncor == HGCalGeomTools::k_threeCorners) {
    rotn = static_cast<int>(std::find(ipat3, ipat3 + 6, iok) - ipat3);
    type = HGCalTypes::WaferThree;
    double rpos12 = ((xpos + dx1[rotn]) * (xpos + dx1[rotn]) + (ypos + dy1[rotn]) * (ypos + dy1[rotn]));
    double rpos22(0);
    if (rpos12 <= rout2 && rpos12 >= rin2) {
      rpos22 = ((xpos + dx5[rotn]) * (xpos + dx5[rotn]) + (ypos + dy5[rotn]) * (ypos + dy5[rotn]));
      if (rpos22 <= rout2 && rpos22 >= rin2)
        type = HGCalTypes::WaferSemi;
    }
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "Test for Semi " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                    << type;
    if ((type == HGCalTypes::WaferThree) && (wType == 0)) {
      rpos12 = ((xpos + dx3[rotn]) * (xpos + dx3[rotn]) + (ypos + dy3[rotn]) * (ypos + dy3[rotn]));
      if (rpos12 <= rout2 && rpos12 >= rin2) {
        rpos22 = ((xpos + dx6[rotn]) * (xpos + dx6[rotn]) + (ypos + dy6[rotn]) * (ypos + dy6[rotn]));
        if (rpos22 <= rout2 && rpos22 >= rin2)
          type = HGCalTypes::WaferSemi2;
      }
      if (debug)
        edm::LogVerbatim("HGCalGeom") << "Test for SemiM " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                      << type;
    }
  } else {
    type = HGCalTypes::WaferOut;
  }

  if (debug)
    edm::LogVerbatim("HGCalGeom") << "I/p " << xpos << ":" << ypos << ":" << delX << ":" << delY << ":" << rin << ":"
                                  << rout << ":" << wType << ":" << mode << " o/p " << iok << ":" << ncor << ":" << type
                                  << ":" << rotn;
  return ((mode == 0) ? std::make_pair(ncor, rotn) : std::make_pair(type, (rotn + HGCalWaferMask::k_OffsetRotation)));
}

bool HGCalWaferMask::goodTypeMode(
    double xpos, double ypos, double delX, double delY, double rin, double rout, int part, int rotn, bool debug) {
  if (part < 0 || part > HGCalTypes::WaferSizeMax)
    return false;
  if (rotn < 0 || rotn > HGCalTypes::WaferCornerMax)
    return false;
  double rin2 = rin * rin;
  double rout2 = rout * rout;
  static const int corners = HGCalTypes::WaferCornerMax;
  static const int corner2 = 2 * HGCalTypes::WaferCornerMax;
  static const int base = 10;
  static const int base2 = 100;
  double dx0[corners] = {
      0.0, HGCalTypes::c10 * delX, HGCalTypes::c10 * delX, 0.0, -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX};
  double dy0[corners] = {-HGCalTypes::c10 * delY,
                         -HGCalTypes::c50 * delY,
                         HGCalTypes::c50 * delY,
                         HGCalTypes::c10 * delY,
                         HGCalTypes::c50 * delY,
                         -HGCalTypes::c50 * delY};
  double dx1[corners] = {HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c50 * delX,
                         -HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX};
  double dy1[corners] = {
      -HGCalTypes::c75 * delY, 0.0, HGCalTypes::c75 * delY, HGCalTypes::c75 * delY, 0.0, -HGCalTypes::c75 * delY};
  double dx2[corner2] = {HGCalTypes::c22 * delX,
                         HGCalTypes::c77 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c77 * delX,
                         HGCalTypes::c22 * delX,
                         -HGCalTypes::c22 * delX,
                         -HGCalTypes::c77 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c77 * delX,
                         -HGCalTypes::c22 * delX};
  double dy2[corner2] = {-HGCalTypes::c88 * delY,
                         -HGCalTypes::c61 * delY,
                         -HGCalTypes::c27 * delY,
                         HGCalTypes::c27 * delY,
                         HGCalTypes::c61 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c61 * delY,
                         HGCalTypes::c27 * delY,
                         -HGCalTypes::c27 * delY,
                         -HGCalTypes::c61 * delY,
                         -HGCalTypes::c88 * delY};
  bool ok(true);
  int ncf(-1);
  switch (part) {
    case (HGCalTypes::WaferThree): {
      static const int nc0[corners] = {450, 150, 201, 312, 423, 534};
      int nc = nc0[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      break;
    }
    case (HGCalTypes::WaferSemi2): {
      static const int nc10[corners] = {450, 150, 201, 312, 423, 534};
      static const int nc11[corners] = {700, 902, 1104, 106, 308, 510};
      int nc = nc10[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc11[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base2;
        double xc1 = xpos + dx2[k];
        double yc1 = ypos + dy2[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base2;
      }
      break;
    }
    case (HGCalTypes::WaferSemi): {
      static const int nc20[corners] = {450, 150, 201, 312, 423, 534};
      static const int nc21[corners] = {30, 14, 25, 30, 41, 52};
      int nc = nc20[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc21[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx1[k];
        double yc1 = ypos + dy1[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base;
      }
      break;
    }
    case (HGCalTypes::WaferHalf): {
      static const int nc3[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      int nc = nc3[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      break;
    }
    case (HGCalTypes::WaferChopTwoM): {
      static const int nc40[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      static const int nc41[corners] = {500, 702, 904, 1106, 108, 310};
      int nc = nc40[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc41[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base2;
        double xc1 = xpos + dx2[k];
        double yc1 = ypos + dy2[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base2;
      }
      break;
    }
    case (HGCalTypes::WaferChopTwo): {
      static const int nc50[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      static const int nc51[corners] = {20, 13, 24, 35, 40, 51};
      int nc = nc50[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc51[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx1[k];
        double yc1 = ypos + dy1[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base;
      }
      break;
    }
    case (HGCalTypes::WaferFive): {
      static const int nc6[corners] = {23450, 13450, 24501, 35012, 40123, 51234};
      int nc = nc6[rotn];
      for (int k1 = 0; k1 < 5; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
      }
      break;
    }
    default: {
      for (int k = 0; k < corners; ++k) {
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        double rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
      }
      break;
    }
  }
  if (debug)
    edm::LogVerbatim("HGCalGeom") << "I/p "
                                  << ":" << xpos << ":" << ypos << ":" << delX << ":" << delY << ":" << rin << ":"
                                  << rout << ":" << part << ":" << rotn << " Results " << ok << ":" << ncf;
  return ok;
}
