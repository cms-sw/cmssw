#include "Geometry/HGCalCommonData/interface/HGCalWaferMask.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeomTools.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <algorithm>

//#define EDM_ML_DEBUG

bool HGCalWaferMask::maskCell(int u, int v, int N, int ncor, int fcor, int corners) {
  /*
Masks each cell (or not) according to its wafer and cell position (detId) and to the user needs (corners).
Each wafer has k_CornerSize corners which are defined in anti-clockwise order starting from the corner at the top, which is always #0. 'ncor' denotes the number of corners inside the physical region. 'fcor' is the defined to be the first corner that appears inside the detector's physical volume in anti-clockwise order. 
The argument 'corners' controls the types of wafers the user wants: for instance, corners=3 masks all wafers that have at least 3 corners inside the physical region. 
 */
  bool mask(false);
  if (ncor < corners) {
    mask = true;
  } else {
    if (ncor == 4) {
      switch (fcor) {
        case (0): {
          mask = (v >= N);
          break;
        }
        case (1): {
          mask = (u >= N);
          break;
        }
        case (2): {
          mask = (u > v);
          break;
        }
        case (3): {
          mask = (v < N);
          break;
        }
        case (4): {
          mask = (u < N);
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
          if (ncor == 3) {
            mask = !((u > 2 * v) && (v < N));
          } else {
            mask = ((u >= N) && (v >= N) && ((u + v) > (3 * N - 2)));
          }
          break;
        }
        case (1): {
          if (ncor == 3) {
            mask = !((u + v) < N);
          } else {
            mask = ((u >= N) && (u > v) && ((2 * u - v) > 2 * N));
          }
          break;
        }
        case (2): {
          if (ncor == 3) {
            mask = !((u < N) && (v > u) && (v > (2 * u - 1)));
          } else {
            mask = ((u > 2 * v) && (v < N));
          }
          break;
        }
        case (3): {
          if (ncor == 3) {
            mask = !((v >= u) && ((2 * v - u) > (2 * N - 2)));
          } else {
            mask = ((u + v) < N);
          }
          break;
        }
        case (4): {
          if (ncor == 3) {
            mask = !((u >= N) && (v >= N) && ((u + v) > (3 * N - 2)));
          } else {
            mask = ((u < N) && (v > u) && (v > (2 * u - 1)));
          }
          break;
        }
        default: {
          if (ncor == 3) {
            mask = !((u >= N) && (u > v) && ((2 * u - v) > 2 * N));
          } else {
            mask = ((v >= u) && ((2 * v - u) > (2 * N - 2)));
          }
          break;
        }
      }
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Corners: " << ncor << ":" << fcor << " N " << N << " u " << u << " v " << v
                                << " Mask " << mask;
#endif
  return mask;
}

bool HGCalWaferMask::goodCell(int u, int v, int N, int type, int rotn) {
  bool good(false);
  int n2 = N / 2;
  switch (type) {
    case (HGCalGeomTools::WaferFull): {  //WaferFull
      good = true;
      break;
    }
    case (HGCalGeomTools::WaferFive): {  //WaferFive
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          int u2 = (u + 1) / 2;
          good = ((v - u2) < N);
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = ((v + u) < (3 * N));
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          good = ((2 * u - v) <= (2 * N));
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          int u2 = u / 2;
          good = (v >= u2);
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = ((v + u) <= (N - 1));
          break;
        }
        default: {
          good = (v <= (2 * u));
          break;
        }
      }
      break;
    }
    case (HGCalGeomTools::WaferChopTwo): {  //WaferChopTwo
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = (v <= (3 * n2));
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = (u <= (3 * n2));
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          good = ((u - v) <= (n2 + 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = (v >= (n2 - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = (u >= (n2 - 1));
          break;
        }
        default: {
          good = ((v - u) <= n2);
          break;
        }
      }
      break;
    }
    case (HGCalGeomTools::WaferChopTwoM): {  //WaferChopTwoM
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = (v < (3 * n2));
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = (u < (3 * n2));
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          good = ((u - v) <= n2);
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = (v >= n2);
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
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
    case (HGCalGeomTools::WaferHalf): {  //WaferHalf
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = (v < N);
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = (u <= N);
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          good = (v >= u);
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = (v >= (N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = (u >= N);
          break;
        }
        default: {
          good = (u >= v);
          break;
        }
      }
      break;
    }
    case (HGCalGeomTools::WaferSemi): {  //WaferSemi
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = ((u + v) <= (2 * N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = ((2 * u - v) >= N);
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) >= (n2 - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = ((u + v) >= (2 * N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = ((2 * u - v) <= N);
          break;
        }
        default: {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) <= (n2 - 1));
          break;
        }
      }
      break;
    }
    case (HGCalGeomTools::WaferThree): {  //WaferThree
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = ((v + u) < N);
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = (v >= (2 * u));
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          int u2 = (u / 2);
          good = ((v - u2) >= N);
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = ((v + u) >= (3 * N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = ((2 * u - v) >= (2 * N));
          break;
        }
        default: {
          int u2 = (u / 2);
          good = (v <= u2);
          break;
        }
      }
      break;
    }
    case (HGCalGeomTools::WaferSemi2): {  //WaferSemi2
      switch (rotn) {
        case (HGCalGeomTools::WaferCorner0): {
          good = ((u + v) < (2 * N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner1): {
          good = ((2 * u - v) > N);
          break;
        }
        case (HGCalGeomTools::WaferCorner2): {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) > (n2 - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner3): {
          good = ((u + v) > (2 * N - 1));
          break;
        }
        case (HGCalGeomTools::WaferCorner4): {
          good = ((2 * u - v) < N);
          break;
        }
        default: {
          int u2 = ((u + 1) / 2);
          good = ((v - u2) < (n2 - 1));
          break;
        }
      }
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "u|v " << u << ":" << v << " N " << N << " type " << type << " rot " << rotn
                                << " good " << good;
#endif
  return good;
}

std::pair<int, int> HGCalWaferMask::getTypeMode(const double& xpos,
                                                const double& ypos,
                                                const double& r1,
                                                const double& R1,
                                                const double& rin,
                                                const double& rout,
                                                const int& NW,
                                                const int& mode) {
  int ncor(0), fcor(0), iok(0);
  int type(HGCalGeomTools::WaferFull), rotn(HGCalGeomTools::WaferCorner0);
  static const double sqrt3 = std::sqrt(3.0);
  double dxw = r1 / (NW * sqrt3);
  double dyw = 0.5 * r1 / NW;

  static const int corners = 6;
  double xc[corners], yc[corners];
  xc[0] = xpos;
  yc[0] = ypos + R1;
  xc[1] = xpos - r1;
  yc[1] = ypos + 0.5 * R1;
  xc[2] = xpos - r1;
  yc[2] = ypos - 0.5 * R1;
  xc[3] = xpos;
  yc[3] = ypos - R1;
  xc[4] = xpos + r1;
  yc[4] = ypos - 0.5 * R1;
  xc[5] = xpos + r1;
  yc[5] = ypos + 0.5 * R1;
  for (int k = 0; k < corners; ++k) {
    double rpos = sqrt(xc[k] * xc[k] + yc[k] * yc[k]);
    if (rpos <= rout && rpos >= rin) {
      ++ncor;
      iok = iok * 10 + 1;
    } else {
      iok *= 10;
    }
  }
  static const int ipat5[corners] = {111110, 11111, 101111, 110111, 111011, 111101};
  static const int ipat4[corners] = {111100, 11110, 1111, 100111, 110011, 111001};
  static const int ipat3[corners] = {111000, 11100, 1110, 111, 100011, 110001};
  double dx1[corners] = {(0.5 * r1 + dxw), -0.5 * r1, -r1, -0.5 * r1, (0.5 * r1 - dxw), r1};
  double dy1[corners] = {(0.75 * R1 - dyw), 0.75 * R1, dyw, -0.75 * R1, -(0.75 * R1 + dyw), 0.0};
  double dx2[corners] = {(0.5 * r1 + dxw), r1, (0.5 * r1 - dxw), -0.5 * r1, -r1, -0.5 * r1};
  double dy2[corners] = {-(0.75 * R1 - dyw), 0.0, (0.75 * R1 + dyw), 0.75 * R1, -dyw, -0.75 * R1};
  double dx3[corners] = {(0.5 * r1 - dxw), -0.5 * r1, -r1, -0.5 * r1, (0.5 * r1 + dxw), r1};
  double dy3[corners] = {(0.75 * R1 + dyw), 0.75 * R1, -dyw, -0.75 * R1, -(0.75 * R1 - dyw), 0.0};
  double dx4[corners] = {(0.5 * r1 - dxw), r1, (0.5 * r1 + dxw), 0.5 * r1, -r1, -0.5 * r1};
  double dy4[corners] = {-(0.75 * R1 + dyw), 0.0, (0.75 * R1 - dyw), 0.75 * R1, dyw, -0.75 * R1};
  double dx5[corners] = {0.5 * r1, -0.5 * r1, -r1, -0.5 * r1, 0.5 * r1, r1};
  double dy5[corners] = {0.75 * R1, 0.75 * R1, 0.0, -0.75 * R1, -0.75 * R1, 0.0};
  double dx6[corners] = {-0.5 * r1, 0.5 * r1, r1, 0.5 * r1, -0.5 * r1, -r1};
  double dy6[corners] = {-0.75 * R1, -0.75 * R1, 0.0, 0.75 * R1, 0.75 * R1, 0.0};

  if (ncor == 6) {
  } else if (ncor == 5) {
    fcor = static_cast<int>(std::find(ipat5, ipat5 + 6, iok) - ipat5);
    type = HGCalGeomTools::WaferFive;
    rotn = fcor + 1;
    if (rotn > 5)
      rotn = 0;
  } else if (ncor == 4) {
    fcor = static_cast<int>(std::find(ipat4, ipat4 + 6, iok) - ipat4);
    type = HGCalGeomTools::WaferHalf;
    rotn = fcor;
    double rpos = sqrt((xpos + dx1[fcor]) * (xpos + dx1[fcor]) + (ypos + dy1[fcor]) * (ypos + dy1[fcor]));
    if (rpos <= rout && rpos >= rin) {
      rpos = sqrt((xpos + dx2[fcor]) * (xpos + dx2[fcor]) + (ypos + dy2[fcor]) * (ypos + dy2[fcor]));
      if (rpos <= rout && rpos >= rin)
        type = HGCalGeomTools::WaferChopTwo;
    }
    if (type == HGCalGeomTools::WaferHalf) {
      rpos = sqrt((xpos + dx3[fcor]) * (xpos + dx3[fcor]) + (ypos + dy3[fcor]) * (ypos + dy3[fcor]));
      if (rpos <= rout && rpos >= rin) {
        rpos = sqrt((xpos + dx4[fcor]) * (xpos + dx4[fcor]) + (ypos + dy4[fcor]) * (ypos + dy4[fcor]));
        if (rpos <= rout && rpos >= rin)
          type = HGCalGeomTools::WaferChopTwoM;
      }
    }
  } else if (ncor == 3) {
    fcor = static_cast<int>(std::find(ipat3, ipat3 + 6, iok) - ipat3);
    type = HGCalGeomTools::WaferThree;
    rotn = fcor - 1;
    if (rotn < 0)
      rotn = HGCalGeomTools::WaferCorner5;
    double rpos = sqrt((xpos + dx5[fcor]) * (xpos + dx5[fcor]) + (ypos + dy5[fcor]) * (ypos + dy5[fcor]));
    if (rpos <= rout && rpos >= rin) {
      rpos = sqrt((xpos + dx6[fcor]) * (xpos + dx6[fcor]) + (ypos + dy6[fcor]) * (ypos + dy6[fcor]));
      if (rpos <= rout && rpos >= rin)
        type = HGCalGeomTools::WaferSemi;
    }
  } else {
    type = HGCalGeomTools::WaferOut;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("HGCalGeom") << "Corners " << ncor << ":" << fcor << " Type " << type << ":" << rotn;
#endif
  return ((mode == 0) ? std::make_pair(ncor, fcor) : std::make_pair(type, 10 + rotn));
}
