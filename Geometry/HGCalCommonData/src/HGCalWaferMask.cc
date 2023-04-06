#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalTypes.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>
#include <sstream>

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
  // for V15 and V16
  bool good(false);
  int n2 = n / 2;
  int n4 = n / 4;
  int n3 = (n + 1) / 3;
  switch (type) {
    case (HGCalTypes::WaferFull): {  //WaferFull
      good = true;
      break;
    }
    case (HGCalTypes::WaferFive): {  //WaferFive
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          int u2 = u / 2;
          good = ((v - u2) <= n);
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((v + u) < (3 * n));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int v2 = (v + 1) / 2;
          good = ((u - v2) <= n);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          int u2 = (u - 1) / 2;
          good = (u2 <= v);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((v + u) >= n - 1);
          break;
        }
        default: {
          int v2 = v / 2;
          good = (u >= v2);
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
          good = (u >= n2 - 1);
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
          good = (u <= (5 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((u - v) <= n4);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= (3 * n4 - 1));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= (3 * n4));
          break;
        }
        default: {
          good = ((v - u) <= n4);
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
          good = (u <= n);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = (v >= u);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= n - 1);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= n);
          break;
        }
        default: {
          good = (u >= v);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferSemi): {  //WaferSemi
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((u + v) <= (2 * n));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((2 * u - v) <= (n + 1));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((2 * v - u) >= (n - 2));
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((u + v) >= (2 * n - 2));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((2 * u - v) >= (n - 1));
          break;
        }
        default: {
          good = ((2 * v - u) <= n);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferThree): {  //WaferThree
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((v + u) <= n);
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((2 * u - v) <= 1);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int u2 = ((u > 0) ? (u / 2) : 0);
          int uv = v - u2;
          good = (uv >= (n - 1));
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((v + u) >= (3 * n - 2));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          int uv = 2 * u - v;
          good = (uv >= (2 * n - 1));
          break;
        }
        default: {
          int uv = u - 2 * v;
          good = (uv >= 0);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferSemi2): {  //WaferSemi2
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((u + v) <= (4 * n3 + 1));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((2 * u - v) <= n2);
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) >= (3 * n4 - 1));
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((u + v) >= (5 * n2 - 1));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((2 * u - v) >= (3 * n2));
          break;
        }
        default: {
          int u2 = (u + 1) / 2;
          good = ((v - u2) < n4);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferFive2): {  //WaferFive2
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = ((2 * v - u) <= (3 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = ((u + v) < (5 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((2 * u - v) >= (3 * n2));
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = ((2 * v - u) >= n3);
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = ((u + v) > (4 * n3));
          break;
        }
        default: {
          good = ((2 * u - v) >= n2);
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferHalf2): {  //WaferHalf2
      switch (rotn) {
        case (HGCalTypes::WaferCorner0): {
          good = (v <= (3 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner1): {
          good = (u <= (3 * n4));
          break;
        }
        case (HGCalTypes::WaferCorner2): {
          good = ((v - u) >= n4 - 1);
          break;
        }
        case (HGCalTypes::WaferCorner3): {
          good = (v >= (5 * n4 - 1));
          break;
        }
        case (HGCalTypes::WaferCorner4): {
          good = (u >= (5 * n4 - 1));
          break;
        }
        default: {
          good = ((u - v) >= n4);
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

bool HGCalWaferMask::goodCell(int u, int v, int waferType) {
  // for V17
  bool good(false);
  switch (waferType) {
    case (HGCalTypes::WaferFull): {  //WaferFull
      good = true;
      break;
    }
    case (HGCalTypes::WaferLDTop): {
      good = (u * HGCalTypes::edgeWaferLDTop[0] + v * HGCalTypes::edgeWaferLDTop[1] <= HGCalTypes::edgeWaferLDTop[2]);
      break;
    }
    case (HGCalTypes::WaferLDBottom): {
      good = (u * HGCalTypes::edgeWaferLDBottom[0] + v * HGCalTypes::edgeWaferLDBottom[1] <=
              HGCalTypes::edgeWaferLDBottom[2]);
      break;
    }
    case (HGCalTypes::WaferLDLeft): {
      good =
          (u * HGCalTypes::edgeWaferLDLeft[0] + v * HGCalTypes::edgeWaferLDLeft[1] <= HGCalTypes::edgeWaferLDLeft[2]);
      break;
    }
    case (HGCalTypes::WaferLDRight): {
      good = (u * HGCalTypes::edgeWaferLDRight[0] + v * HGCalTypes::edgeWaferLDRight[1] <=
              HGCalTypes::edgeWaferLDRight[2]);
      break;
    }
    case (HGCalTypes::WaferLDFive): {
      good =
          (u * HGCalTypes::edgeWaferLDFive[0] + v * HGCalTypes::edgeWaferLDFive[1] <= HGCalTypes::edgeWaferLDFive[2]);
      break;
    }
    case (HGCalTypes::WaferLDThree): {
      good = (u * HGCalTypes::edgeWaferLDThree[0] + v * HGCalTypes::edgeWaferLDThree[1] <=
              HGCalTypes::edgeWaferLDThree[2]);
      break;
    }
    case (HGCalTypes::WaferHDTop): {
      good = (u * HGCalTypes::edgeWaferHDTop[0] + v * HGCalTypes::edgeWaferHDTop[1] <= HGCalTypes::edgeWaferHDTop[2]);
      break;
    }
    case (HGCalTypes::WaferHDBottom): {
      good = (u * HGCalTypes::edgeWaferHDBottom[0] + v * HGCalTypes::edgeWaferHDBottom[1] <=
              HGCalTypes::edgeWaferHDBottom[2]);
      break;
    }
    case (HGCalTypes::WaferHDLeft): {
      good =
          (u * HGCalTypes::edgeWaferHDLeft[0] + v * HGCalTypes::edgeWaferHDLeft[1] <= HGCalTypes::edgeWaferHDLeft[2]);
      break;
    }
    case (HGCalTypes::WaferHDRight): {
      good = (u * HGCalTypes::edgeWaferHDRight[0] + v * HGCalTypes::edgeWaferHDRight[1] <=
              HGCalTypes::edgeWaferHDRight[2]);
      break;
    }
    case (HGCalTypes::WaferHDFive): {
      good =
          (u * HGCalTypes::edgeWaferHDFive[0] + v * HGCalTypes::edgeWaferHDFive[1] <= HGCalTypes::edgeWaferHDFive[2]);
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "u|v " << u << ":" << v << " WaferType " << waferType << " good " << good;
#endif
  return good;
}

int HGCalWaferMask::getRotation(int zside, int type, int rotn) {
  // Needs extension for V17
  if (rotn >= HGCalTypes::WaferCornerMax)
    rotn = HGCalTypes::WaferCorner0;
  int newrotn(rotn);
  if ((zside < 0) && (type != HGCalTypes::WaferFull)) {
    if ((type == HGCalTypes::WaferFive) || (type == HGCalTypes::WaferFive2)) {  //WaferFive/WaferFive2
      static constexpr int rot1[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner4,
                                                               HGCalTypes::WaferCorner3,
                                                               HGCalTypes::WaferCorner2,
                                                               HGCalTypes::WaferCorner1,
                                                               HGCalTypes::WaferCorner0,
                                                               HGCalTypes::WaferCorner5};
      newrotn = rot1[rotn];
    } else if ((type == HGCalTypes::WaferThree) || (type == HGCalTypes::WaferSemi) ||
               (type == HGCalTypes::WaferSemi2)) {  //WaferThree/WaferSemi/WaferSemi2
      static constexpr int rot2[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner2,
                                                               HGCalTypes::WaferCorner1,
                                                               HGCalTypes::WaferCorner0,
                                                               HGCalTypes::WaferCorner5,
                                                               HGCalTypes::WaferCorner4,
                                                               HGCalTypes::WaferCorner3};
      newrotn = rot2[rotn];
    } else {  //WaferHalf/WaferChopTwo/WaferChopTwoM/WaferHalf2
      static constexpr int rot3[HGCalTypes::WaferCornerMax] = {HGCalTypes::WaferCorner3,
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
  // No need to extend this for V17 -- use flat file information only
  int ncor(0), iok(0);
  int type(HGCalTypes::WaferFull), rotn(HGCalTypes::WaferCorner0);

  static constexpr int corners = 6;
  static constexpr int base = 10;
  double rin2 = rin * rin;
  double rout2 = rout * rout;
  double dx0[corners] = {HGCalTypes::c00 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c00 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c10 * delX};
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

  static constexpr int ipat5[corners] = {101111, 110111, 111011, 111101, 111110, 11111};
  static constexpr int ipat4[corners] = {100111, 110011, 111001, 111100, 11110, 1111};
  static constexpr int ipat3[corners] = {100011, 110001, 111000, 11100, 1110, 111};
  static constexpr int ipat2[corners] = {11, 100001, 110000, 11000, 1100, 110};
  double dx1[corners] = {HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c50 * delX,
                         -HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX};
  double dy1[corners] = {-HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         HGCalTypes::c75 * delY,
                         HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         -HGCalTypes::c75 * delY};
  double dx2[corners] = {HGCalTypes::c50 * delX,
                         -HGCalTypes::c50 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c50 * delX,
                         HGCalTypes::c50 * delX,
                         HGCalTypes::c10 * delX};
  double dy2[corners] = {HGCalTypes::c75 * delY,
                         HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         -HGCalTypes::c75 * delY,
                         -HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY};
  double dx3[corners] = {HGCalTypes::c22 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c77 * delX,
                         -HGCalTypes::c22 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c77 * delX};
  double dy3[corners] = {-HGCalTypes::c88 * delY,
                         -HGCalTypes::c27 * delY,
                         HGCalTypes::c61 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c27 * delY,
                         -HGCalTypes::c61 * delY};
  double dx4[corners] = {HGCalTypes::c22 * delX,
                         -HGCalTypes::c77 * delX,
                         -HGCalTypes::c10 * delX,
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
  double dy5[corners] = {HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         -HGCalTypes::c75 * delY,
                         -HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         HGCalTypes::c75 * delY};
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
  double dx7[corners] = {-HGCalTypes::c22 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c77 * delX,
                         HGCalTypes::c22 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c77 * delX};
  double dy7[corners] = {HGCalTypes::c88 * delY,
                         HGCalTypes::c27 * delY,
                         -HGCalTypes::c61 * delY,
                         -HGCalTypes::c88 * delY,
                         -HGCalTypes::c27 * delY,
                         HGCalTypes::c61 * delY};
  double dx8[corners] = {HGCalTypes::c77 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c22 * delX,
                         -HGCalTypes::c77 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c22 * delX};
  double dy8[corners] = {-HGCalTypes::c61 * delY,
                         HGCalTypes::c27 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c61 * delY,
                         -HGCalTypes::c27 * delY,
                         -HGCalTypes::c88 * delY};
  double dx9[corners] = {-HGCalTypes::c22 * delX,
                         HGCalTypes::c77 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c22 * delX,
                         -HGCalTypes::c77 * delX,
                         -HGCalTypes::c10 * delX};
  double dy9[corners] = {-HGCalTypes::c88 * delY,
                         -HGCalTypes::c61 * delY,
                         HGCalTypes::c27 * delY,
                         HGCalTypes::c88 * delY,
                         HGCalTypes::c61 * delY,
                         -HGCalTypes::c27 * delY};

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
    double rpos12 = ((xpos + dx7[rotn]) * (xpos + dx7[rotn]) + (ypos + dy7[rotn]) * (ypos + dy7[rotn]));
    double rpos22(0);
    if (rpos12 <= rout2 && rpos12 >= rin2) {
      rpos22 = ((xpos + dx8[rotn]) * (xpos + dx8[rotn]) + (ypos + dy8[rotn]) * (ypos + dy8[rotn]));
      if (rpos22 <= rout2 && rpos22 >= rin2)
        type = HGCalTypes::WaferFive2;
    }
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "Test for Five2 " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                    << type;
    if ((type == HGCalTypes::WaferThree) && (wType == 0)) {
      rpos12 = ((xpos + dx1[rotn]) * (xpos + dx1[rotn]) + (ypos + dy1[rotn]) * (ypos + dy1[rotn]));
      if (rpos12 <= rout2 && rpos12 >= rin2) {
        rpos22 = ((xpos + dx5[rotn]) * (xpos + dx5[rotn]) + (ypos + dy5[rotn]) * (ypos + dy5[rotn]));
        if (rpos22 <= rout2 && rpos22 >= rin2)
          type = HGCalTypes::WaferSemi;
      }
      if (debug)
        edm::LogVerbatim("HGCalGeom") << "Test for Semi " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                      << type;
    }
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
  } else if (ncor == HGCalGeomTools::k_twoCorners) {
    rotn = static_cast<int>(std::find(ipat2, ipat2 + 6, iok) - ipat2);
    type = HGCalTypes::WaferOut;
    double rpos12 = ((xpos + dx7[rotn]) * (xpos + dx7[rotn]) + (ypos + dy7[rotn]) * (ypos + dy7[rotn]));
    double rpos22(0);
    if (rpos12 <= rout2 && rpos12 >= rin2) {
      rpos22 = ((xpos + dx9[rotn]) * (xpos + dx9[rotn]) + (ypos + dy9[rotn]) * (ypos + dy9[rotn]));
      if (rpos22 <= rout2 && rpos22 >= rin2)
        type = HGCalTypes::WaferHalf2;
      else
        rotn = HGCalTypes::WaferCorner0;
    }
    if (debug)
      edm::LogVerbatim("HGCalGeom") << "Test for Half2 " << std::sqrt(rpos12) << ":" << std::sqrt(rpos22) << " Type "
                                    << type;
  } else {
    type = HGCalTypes::WaferOut;
  }

  if (debug)
    edm::LogVerbatim("HGCalGeom") << "I/p " << xpos << ":" << ypos << ":" << delX << ":" << delY << ":" << rin << ":"
                                  << rout << ":" << wType << ":" << mode << " o/p " << iok << ":" << ncor << ":" << type
                                  << ":" << rotn;
  return ((mode == 0) ? std::make_pair(ncor, rotn) : std::make_pair(type, (rotn + HGCalTypes::k_OffsetRotation)));
}

bool HGCalWaferMask::goodTypeMode(
    double xpos, double ypos, double delX, double delY, double rin, double rout, int part, int rotn, bool debug) {
  // Needs extension for V17
  if (part < 0 || part > HGCalTypes::WaferSizeMax)
    return false;
  if (rotn < 0 || rotn > HGCalTypes::WaferCornerMax)
    return false;
  double rin2 = rin * rin;
  double rout2 = rout * rout;
  double rpos2(0);
  static constexpr int corners = HGCalTypes::WaferCornerMax;
  static constexpr int corner2 = 2 * HGCalTypes::WaferCornerMax;
  static constexpr int base = 10;
  static constexpr int base2 = 100;
  double dx0[corners] = {HGCalTypes::c00 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c10 * delX,
                         HGCalTypes::c00 * delX,
                         -HGCalTypes::c10 * delX,
                         -HGCalTypes::c10 * delX};
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
  double dy1[corners] = {-HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         HGCalTypes::c75 * delY,
                         HGCalTypes::c75 * delY,
                         HGCalTypes::c00 * delY,
                         -HGCalTypes::c75 * delY};
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
      static constexpr int nc0[corners] = {450, 150, 201, 312, 423, 534};
      int nc = nc0[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc10[corners] = {450, 150, 201, 312, 423, 534};
      static constexpr int nc11[corners] = {700, 902, 1104, 106, 308, 510};
      int nc = nc10[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc20[corners] = {450, 150, 201, 312, 423, 534};
      static constexpr int nc21[corners] = {30, 14, 25, 30, 41, 52};
      int nc = nc20[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc3[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      int nc = nc3[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc40[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      static constexpr int nc41[corners] = {500, 702, 904, 1106, 108, 310};
      int nc = nc40[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc50[corners] = {3450, 1450, 2501, 3012, 4123, 5234};
      static constexpr int nc51[corners] = {20, 13, 24, 35, 40, 51};
      int nc = nc50[rotn];
      for (int k1 = 0; k1 < 4; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
        rpos2 = (xc1 * xc1 + yc1 * yc1);
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
      static constexpr int nc6[corners] = {23450, 13450, 24501, 35012, 40123, 51234};
      int nc = nc6[rotn];
      for (int k1 = 0; k1 < 5; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
      }
      break;
    }
    case (HGCalTypes::WaferFive2): {
      static constexpr int nc60[corners] = {450, 150, 201, 312, 423, 534};
      static constexpr int nc61[corners] = {601, 803, 1005, 7, 209, 411};
      int nc = nc60[rotn];
      for (int k1 = 0; k1 < 3; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc61[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base2;
        double xc1 = xpos + dx2[k];
        double yc1 = ypos + dy2[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base2;
      }
      break;
    }
    case (HGCalTypes::WaferHalf2): {
      static constexpr int nc70[corners] = {45, 50, 1, 12, 23, 34};
      static constexpr int nc71[corners] = {611, 801, 1003, 5, 207, 409};
      int nc = nc70[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base;
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
        nc /= base;
      }
      nc = nc71[rotn];
      for (int k1 = 0; k1 < 2; ++k1) {
        int k = nc % base2;
        double xc1 = xpos + dx2[k];
        double yc1 = ypos + dy2[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k + base2;
          break;
        }
        nc /= base2;
      }
      break;
    }
    default: {
      for (int k = 0; k < corners; ++k) {
        double xc1 = xpos + dx0[k];
        double yc1 = ypos + dy0[k];
        rpos2 = (xc1 * xc1 + yc1 * yc1);
        if ((rpos2 > rout2) || (rpos2 < rin2)) {
          ok = false;
          ncf = k;
          break;
        }
      }
      break;
    }
  }
  if (debug || (!ok))
    edm::LogVerbatim("HGCalGeom") << "I/p "
                                  << ":" << xpos << ":" << ypos << ":" << delX << ":" << delY << ":" << rin << ":"
                                  << rout << ":" << part << ":" << rotn << " Results " << ok << ":" << ncf << " R "
                                  << rin2 << ":" << rout2 << ":" << rpos2;
  return ok;
}

std::vector<std::pair<double, double> > HGCalWaferMask::waferXY(
    int part, int ori, int zside, double waferSize, double offset, double xpos, double ypos) {
  // Good for V15 and V16 versions
  std::vector<std::pair<double, double> > xy;
  int orient = getRotation(-zside, part, ori);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Part " << part << " zSide " << zside << " Orient " << ori << ":" << orient;
#endif
  /*
    The exact contour of partial wafers are obtained by joining points on
    the circumference of a full wafer.
    Numbering the points along the edges of a hexagonal wafer, starting from
    the bottom corner:

                                   3
                               15     18
                             9           8
                          19               14
                        4                     2 
                       16                    23
                       10                     7
                       20                    13
                        5                     1
                          17               22
                            11           6
                               21     12
                                   0

	Depending on the wafer type and orientation index, the corners
	are chosen in the variable *np*
  */
  double delX = 0.5 * waferSize;
  double delY = delX / sin_60_;
  double dx[48] = {HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX, HGCalTypes::c50 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c50 * delX,  -HGCalTypes::c50 * delX, -HGCalTypes::c10 * delX, -HGCalTypes::c50 * delX,
                   HGCalTypes::c22 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c77 * delX,  -HGCalTypes::c22 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c77 * delX, HGCalTypes::c22 * delX,  -HGCalTypes::c77 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c22 * delX, HGCalTypes::c77 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c50 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c50 * delX,  -HGCalTypes::c50 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c50 * delX, HGCalTypes::c50 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c50 * delX,  -HGCalTypes::c50 * delX, -HGCalTypes::c10 * delX, -HGCalTypes::c50 * delX,
                   HGCalTypes::c22 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c77 * delX,  -HGCalTypes::c22 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c77 * delX, HGCalTypes::c22 * delX,  -HGCalTypes::c77 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c22 * delX, HGCalTypes::c77 * delX,  HGCalTypes::c10 * delX};
  double dy[48] = {-HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,
                   HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY, -HGCalTypes::c75 * delY, HGCalTypes::c00 * delY,
                   HGCalTypes::c75 * delY,  HGCalTypes::c75 * delY,  HGCalTypes::c00 * delY,  -HGCalTypes::c75 * delY,
                   -HGCalTypes::c88 * delY, -HGCalTypes::c27 * delY, HGCalTypes::c61 * delY,  HGCalTypes::c88 * delY,
                   HGCalTypes::c27 * delY,  -HGCalTypes::c61 * delY, HGCalTypes::c88 * delY,  HGCalTypes::c61 * delY,
                   -HGCalTypes::c27 * delY, -HGCalTypes::c88 * delY, -HGCalTypes::c61 * delY, HGCalTypes::c27 * delY,
                   -HGCalTypes::c75 * delY, HGCalTypes::c00 * delY,  -HGCalTypes::c75 * delY, HGCalTypes::c00 * delY,
                   HGCalTypes::c75 * delY,  HGCalTypes::c75 * delY,  HGCalTypes::c00 * delY,  -HGCalTypes::c75 * delY,
                   HGCalTypes::c75 * delY,  HGCalTypes::c75 * delY,  HGCalTypes::c00 * delY,  -HGCalTypes::c75 * delY,
                   -HGCalTypes::c88 * delY, -HGCalTypes::c27 * delY, HGCalTypes::c61 * delY,  HGCalTypes::c88 * delY,
                   HGCalTypes::c27 * delY,  -HGCalTypes::c61 * delY, HGCalTypes::c88 * delY,  HGCalTypes::c61 * delY,
                   -HGCalTypes::c27 * delY, -HGCalTypes::c88 * delY, -HGCalTypes::c61 * delY, HGCalTypes::c27 * delY};

  double offsetx[48] = {0.0,
                        -offset,
                        -offset,
                        0.0,
                        offset,
                        offset,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        -offset,
                        0.0,
                        -offset,
                        -offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        -offset,
                        -offset,
                        0.0,
                        -offset,
                        -offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        -offset,
                        -offset};
  double offsety[48] = {offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset * sin_60_,
                        0.0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        offset * sin_60_,
                        0.0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        offset * sin_60_,
                        0.0,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_};

  if (part == HGCalTypes::WaferFull) {
    int np[7] = {0, 1, 2, 3, 4, 5, 0};
    for (int k = 0; k < 7; ++k)
      xy.push_back(std::make_pair((xpos + dx[np[k]] + offsetx[np[k]]), (ypos + dy[np[k]] + offsety[np[k]])));
  } else if (part == HGCalTypes::WaferFive) {
    int np[6][6] = {{0, 2, 3, 4, 5, 0},
                    {1, 3, 4, 5, 0, 1},
                    {2, 4, 5, 0, 1, 2},
                    {3, 5, 0, 1, 2, 3},
                    {4, 0, 1, 2, 3, 4},
                    {5, 1, 2, 3, 4, 5}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHalf) {
    int np[6][5] = {
        {0, 3, 4, 5, 0}, {1, 4, 5, 0, 1}, {2, 5, 0, 1, 2}, {3, 0, 1, 2, 3}, {4, 1, 2, 3, 4}, {5, 2, 3, 4, 5}};
    for (int k = 0; k < 5; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferThree) {
    int np[6][4] = {{0, 4, 5, 0}, {1, 5, 0, 1}, {2, 0, 1, 2}, {3, 1, 2, 3}, {4, 2, 3, 4}, {5, 3, 4, 5}};
    for (int k = 0; k < 4; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferChopTwo) {
    int np[6][7] = {{24, 32, 3, 4, 5, 0, 24},
                    {25, 33, 4, 5, 0, 1, 25},
                    {26, 34, 5, 0, 1, 2, 26},
                    {27, 35, 0, 1, 2, 3, 27},
                    {28, 30, 1, 2, 3, 4, 28},
                    {29, 31, 2, 3, 4, 5, 29}};
    for (int k = 0; k < 7; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferSemi) {
    int np[6][6] = {{6, 9, 4, 5, 0, 6},
                    {7, 10, 5, 0, 1, 7},
                    {8, 11, 0, 1, 2, 8},
                    {9, 6, 1, 2, 3, 9},
                    {10, 7, 2, 3, 4, 10},
                    {11, 8, 3, 4, 5, 11}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferChopTwoM) {
    int np[6][7] = {{36, 42, 3, 4, 5, 0, 36},
                    {37, 43, 4, 5, 0, 1, 37},
                    {38, 44, 5, 0, 1, 2, 38},
                    {39, 45, 0, 1, 2, 3, 39},
                    {40, 46, 1, 2, 3, 4, 40},
                    {41, 47, 2, 3, 4, 5, 41}};
    for (int k = 0; k < 7; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferSemi2) {
    int np[6][6] = {{12, 19, 4, 5, 0, 12},
                    {13, 20, 5, 0, 1, 13},
                    {14, 21, 0, 1, 2, 14},
                    {15, 22, 1, 2, 3, 15},
                    {16, 23, 2, 3, 4, 16},
                    {17, 18, 3, 4, 5, 17}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferFive2) {
    int np[6][6] = {{22, 15, 4, 5, 0, 22},
                    {23, 16, 5, 0, 1, 23},
                    {18, 17, 0, 1, 2, 18},
                    {19, 12, 1, 2, 3, 19},
                    {20, 13, 2, 3, 4, 20},
                    {21, 14, 3, 4, 5, 21}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHalf2) {
    int np[6][5] = {{45, 39, 4, 5, 45},
                    {46, 40, 5, 0, 46},
                    {47, 41, 0, 1, 47},
                    {42, 36, 1, 2, 42},
                    {43, 37, 2, 3, 43},
                    {44, 38, 3, 4, 44}};
    for (int k = 0; k < 5; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[orient][k]] + offsetx[np[orient][k]]),
                                  (ypos + dy[np[orient][k]] + offsety[np[orient][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[orient][k] << ":" << dx[np[orient][k]] + offsetx[np[orient][k]]
                                    << ":" << dy[np[orient][k]] + offsety[np[orient][k]];
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "I/p: " << part << ":" << ori << ":" << zside << ":" << delX << ":" << delY << ":"
                                << xpos << ":" << ypos << " O/p having " << xy.size() << " points:";
  std::ostringstream st1;
  for (unsigned int i = 0; i < xy.size(); ++i)
    st1 << " [" << i << "] " << xy[i].first << ":" << xy[i].second;
  edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
  return xy;
}

std::vector<std::pair<double, double> > HGCalWaferMask::waferXY(
    int part, int place, double waferSize, double offset, double xpos, double ypos) {
  std::vector<std::pair<double, double> > xy;
  // Good for V17 version and uses partial wafer type & placement index
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Part " << part << " Placement Index " << place;
#endif
  /*
    The exact contour of partial wafers are obtained by joining points on
    the circumference of a full wafer.
    Numbering the points along the edges of a hexagonal wafer, starting from
    the bottom corner:
                                   3
                               15     18
                             9           8
                          19               14
                        4                     2 
                       16                    23
                       10                     7
                       20                    13
                        5                     1
                          17               22
                            11           6
                               21     12
                                   0
	Depending on the wafer type and placement index, the corners
	are chosen in the variable *np*
        The points 24-35 are the same as points 12-23 with different offset
  */
  double delX = 0.5 * waferSize;
  double delY = delX / sin_60_;
  double dx[60] = {HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX, HGCalTypes::c50 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c50 * delX,  -HGCalTypes::c50 * delX, -HGCalTypes::c10 * delX, -HGCalTypes::c50 * delX,
                   HGCalTypes::c22 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c77 * delX,  -HGCalTypes::c22 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c77 * delX, HGCalTypes::c22 * delX,  -HGCalTypes::c77 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c22 * delX, HGCalTypes::c77 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c22 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c77 * delX,  -HGCalTypes::c22 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c77 * delX, HGCalTypes::c22 * delX,  -HGCalTypes::c77 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c22 * delX, HGCalTypes::c77 * delX,  HGCalTypes::c10 * delX,
                   HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX, HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  
                   HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,  -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX,
                   HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,
                   -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX, HGCalTypes::c00 * delX,  HGCalTypes::c10 * delX,  
                   HGCalTypes::c10 * delX,  HGCalTypes::c00 * delX,  -HGCalTypes::c10 * delX, -HGCalTypes::c10 * delX,};
  double dy[60] = {-HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,
                   HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY, -HGCalTypes::c75 * delY, HGCalTypes::c00 * delY,
                   HGCalTypes::c75 * delY,  HGCalTypes::c75 * delY,  HGCalTypes::c00 * delY,  -HGCalTypes::c75 * delY,
                   -HGCalTypes::c88 * delY, -HGCalTypes::c27 * delY, HGCalTypes::c61 * delY,  HGCalTypes::c88 * delY,
                   HGCalTypes::c27 * delY,  -HGCalTypes::c61 * delY, HGCalTypes::c88 * delY,  HGCalTypes::c61 * delY,
                   -HGCalTypes::c27 * delY, -HGCalTypes::c88 * delY, -HGCalTypes::c61 * delY, HGCalTypes::c27 * delY,
                   -HGCalTypes::c88 * delY, -HGCalTypes::c27 * delY, HGCalTypes::c61 * delY,  HGCalTypes::c88 * delY,
                   HGCalTypes::c27 * delY,  -HGCalTypes::c61 * delY, HGCalTypes::c88 * delY,  HGCalTypes::c61 * delY,
                   -HGCalTypes::c27 * delY, -HGCalTypes::c88 * delY, -HGCalTypes::c61 * delY, HGCalTypes::c27 * delY,
                   -HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,
                   HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY, -HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, 
                   HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,  HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY,
                   -HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,
                   HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY, -HGCalTypes::c10 * delY, -HGCalTypes::c50 * delY, 
                   HGCalTypes::c50 * delY,  HGCalTypes::c10 * delY,  HGCalTypes::c50 * delY,  -HGCalTypes::c50 * delY,};

  double offsetx[60] = {0.0,
                        -offset,
                        -offset,
                        0.0,
                        offset,
                        offset,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        -offset * cos_60_,
                        -offset,
                        0.0,
                        -offset,
                        -offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        offset,
                        offset,
                        0.0,
                        -offset,
                        -offset,
                        -offset,
                        -offset / cos_60_,
                        -offset,
                        offset,
                        offset / cos_60_,
                        offset,
                        offset,
                        -offset,
                        -offset / cos_60_,
                        -offset,
                        offset,
                        offset / cos_60_,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset,
                        offset * cos_60_,
                        offset * cos_60_,
                        -offset * cos_60_,
                        -offset,
                        -offset * cos_60_,
                        offset * cos_60_,
                        offset};
  double offsety[60] = {offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset * sin_60_,
                        0.0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        offset * sin_60_,
                        0.0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0.0,
                        offset * sin_60_,
                        offset * sin_60_,
                        0.0,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        -offset / sin_60_,
                        -offset / tan_60_,
                        offset / tan_60_,
                        offset / sin_60_,
                        offset / tan_60_,
                        -offset / tan_60_,
                        offset * tan_60_,
                        0,
                        -offset * tan_60_,
                        -offset * tan_60_,
                        0,
                        offset * tan_60_,
                        offset * tan_60_,
                        offset * tan_60_,
                        0,
                        -offset * tan_60_,
                        -offset * tan_60_,
                        0,
                        offset * sin_60_,
                        0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0,
                        offset * sin_60_,
                        offset * sin_60_,
                        offset * sin_60_,
                        0,
                        -offset * sin_60_,
                        -offset * sin_60_,
                        0};

  if (part == HGCalTypes::WaferFull) {
    int np[7] = {0, 1, 2, 3, 4, 5, 0};
    for (int k = 0; k < 7; ++k)
      xy.push_back(std::make_pair((xpos + dx[np[k]] + offsetx[np[k]]), (ypos + dy[np[k]] + offsety[np[k]])));
  } else if (part == HGCalTypes::WaferLDTop) {
    int np[12][5] = {{0, 1, 4, 5, 0},
                     {1, 2, 5, 0, 1},
                     {2, 3, 0, 1, 2},
                     {3, 4, 1, 2, 3},
                     {4, 5, 2, 3, 4},
                     {5, 0, 3, 4, 5},
                     {0, 1, 2, 5, 0},
                     {5, 0, 1, 4, 5},
                     {4, 5, 0, 3, 4},
                     {3, 4, 5, 2, 3},
                     {2, 3, 4, 1, 2},
                     {1, 2, 3, 0, 1}};
    for (int k = 0; k < 5; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferLDBottom) {
    int np[12][5] = {{1, 2, 3, 4, 1},
                     {2, 3, 4, 5, 2},
                     {3, 4, 5, 0, 3},
                     {4, 5, 0, 1, 4},
                     {5, 0, 1, 2, 5},
                     {0, 1, 2, 3, 0},
                     {5, 2, 3, 4, 5},
                     {4, 1, 2, 3, 4},
                     {3, 0, 1, 2, 3},
                     {2, 5, 0, 1, 2},
                     {1, 4, 5, 0, 1},
                     {0, 3, 4, 5, 0}};
    for (int k = 0; k < 5; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferLDLeft) {
    int np[12][6] = {{0, 1, 2, 8, 11, 0},
                     {1, 2, 3, 9, 6, 1},
                     {2, 3, 4, 10, 7, 2},
                     {3, 4, 5, 11, 8, 3},
                     {4, 5, 0, 6, 9, 4},
                     {5, 0, 1, 7, 10, 5},
                     {0, 6, 9, 4, 5, 0},
                     {5, 11, 8, 3, 4, 5},
                     {4, 10, 7, 2, 3, 4},
                     {3, 9, 6, 1, 2, 3},
                     {2, 8, 11, 0, 1, 2},
                     {1, 7, 10, 5, 0, 1}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferLDRight) {
    int np[12][6] = {{5, 11, 8, 3, 4, 5},
                     {0, 6, 9, 4, 5, 0},
                     {1, 7, 10, 5, 0, 1},
                     {2, 8, 11, 0, 1, 2},
                     {3, 9, 6, 1, 2, 3},
                     {4, 10, 7, 2, 3, 4},
                     {1, 2, 3, 9, 6, 1},
                     {0, 1, 2, 8, 11, 0},
                     {5, 0, 1, 7, 10, 5},
                     {4, 5, 0, 6, 9, 4},
                     {3, 4, 5, 11, 8, 3},
                     {2, 3, 4, 10, 7, 2}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferLDFive) {
    int np[12][6] = {{0, 1, 2, 57, 53, 0},
                     {1, 2, 3, 58, 48, 1},
                     {2, 3, 4, 59, 49, 2},
                     {3, 4, 5, 54, 50, 3},
                     {4, 5, 0, 55, 51, 4},
                     {5, 0, 1, 56, 52, 5},
                     {0, 1, 3, 58, 53, 0},
                     {5, 0, 2, 57, 52, 5},
                     {4, 5, 1, 56, 51, 4},
                     {3, 4, 0, 55, 50, 3},
                     {2, 3, 5, 54, 49, 2},
                     {1, 2, 4, 59, 48, 1}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferLDThree) {
    int np[12][4] = {{41, 45, 4, 41},
                     {36, 46, 5, 36},
                     {37, 47, 0, 37},
                     {38, 42, 1, 38},
                     {39, 43, 2, 39},
                     {40, 44, 3, 40},
                     {43, 2, 39, 43},
                     {42, 1, 38, 42},
                     {47, 0, 37, 47},
                     {46, 5, 36, 46},
                     {45, 4, 41, 45},
                     {44, 3, 40, 44}};
    for (int k = 0; k < 4; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHDTop) {
    int np[12][5] = {{0, 34, 28, 5, 0},
                     {1, 35, 29, 0, 1},
                     {2, 30, 24, 1, 2},
                     {3, 31, 25, 2, 3},
                     {4, 32, 26, 3, 4},
                     {5, 33, 27, 4, 5},
                     {0, 1, 35, 29, 0},
                     {5, 0, 34, 28, 5},
                     {4, 5, 33, 27, 4},
                     {3, 4, 32, 26, 3},
                     {2, 3, 31, 25, 2},
                     {1, 2, 30, 24, 1}};
    for (int k = 0; k < 5; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHDBottom) {
    int np[12][7] = {{1, 2, 3, 4, 28, 34, 1},
                     {2, 3, 4, 5, 29, 35, 2},
                     {3, 4, 5, 0, 24, 30, 3},
                     {4, 5, 0, 1, 25, 31, 4},
                     {5, 0, 1, 2, 26, 32, 5},
                     {0, 1, 2, 3, 27, 33, 0},
                     {5, 29, 35, 2, 3, 4, 5},
                     {4, 28, 34, 1, 2, 3, 4},
                     {3, 27, 33, 0, 1, 2, 3},
                     {2, 26, 32, 5, 0, 1, 2},
                     {1, 25, 31, 4, 5, 0, 1},
                     {0, 24, 30, 3, 4, 5, 0}};
    for (int k = 0; k < 7; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHDLeft) {
    int np[12][6] = {{0, 1, 2, 14, 21, 0},
                     {1, 2, 3, 15, 22, 1},
                     {2, 3, 4, 16, 23, 2},
                     {3, 4, 5, 17, 18, 3},
                     {4, 5, 0, 12, 19, 4},
                     {5, 0, 1, 13, 20, 5},
                     {0, 12, 19, 4, 5, 0},
                     {5, 17, 18, 3, 4, 5},
                     {4, 16, 23, 2, 3, 4},
                     {3, 15, 22, 1, 2, 3},
                     {2, 14, 21, 0, 1, 2},
                     {1, 13, 20, 5, 0, 1}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHDRight) {
    int np[12][6] = {{5, 17, 18, 3, 4, 5},
                     {0, 12, 19, 4, 5, 0},
                     {1, 13, 20, 5, 0, 1},
                     {2, 14, 21, 0, 1, 2},
                     {3, 15, 22, 1, 2, 3},
                     {4, 16, 23, 2, 3, 4},
                     {1, 2, 3, 15, 22, 1},
                     {0, 1, 2, 14, 21, 0},
                     {5, 0, 1, 13, 20, 5},
                     {4, 5, 0, 12, 19, 4},
                     {3, 4, 5, 17, 18, 3},
                     {2, 3, 4, 16, 23, 2}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  } else if (part == HGCalTypes::WaferHDFive) {
    int np[12][6] = {{0, 1, 2, 18, 17, 0},
                     {1, 2, 3, 19, 12, 1},
                     {2, 3, 4, 20, 13, 2},
                     {3, 4, 5, 21, 14, 3},
                     {4, 5, 0, 22, 15, 4},
                     {5, 0, 1, 23, 16, 5},
                     {0, 22, 15, 4, 5, 0},
                     {5, 21, 14, 3, 4, 5},
                     {4, 20, 13, 2, 3, 4},
                     {3, 19, 12, 1, 2, 3},
                     {2, 18, 17, 0, 1, 2},
                     {1, 23, 16, 5, 0, 1}};
    for (int k = 0; k < 6; ++k) {
      xy.push_back(std::make_pair((xpos + dx[np[place][k]] + offsetx[np[place][k]]),
                                  (ypos + dy[np[place][k]] + offsety[np[place][k]])));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("HGCalGeom") << k << ":" << np[place][k] << ":" << dx[np[place][k]] + offsetx[np[place][k]]
                                    << ":" << dy[np[place][k]] + offsety[np[place][k]];
#endif
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "I/p: " << part << ":" << place << ":" << delX << ":" << delY << ":" << xpos << ":"
                                << ypos << " O/p having " << xy.size() << " points:";
  std::ostringstream st1;
  for (unsigned int i = 0; i < xy.size(); ++i)
    st1 << " [" << i << "] " << xy[i].first << ":" << xy[i].second;
  edm::LogVerbatim("HGCalGeom") << st1.str();
#endif
  return xy;
}
