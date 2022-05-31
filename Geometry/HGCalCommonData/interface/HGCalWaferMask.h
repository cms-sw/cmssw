#ifndef HGCalCommonData_HGCalWaferMask_h
#define HGCalCommonData_HGCalWaferMask_h

/** \class HGCalWaferMask
 *
 * this class determines the masking of wafers to mimic partial wafers
 *
 *  $Date: 2019/01/15 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 *
 */

#include <cmath>
#include <vector>
#include <array>

class HGCalWaferMask {
public:
  HGCalWaferMask() = default;

  // Decides if the cell is present or not from # oc corners (for V14)
  static bool maskCell(int u, int v, int N, int ncor, int fcor, int corners);
  // Decides if the cell is present or not from # oc corners (for V15, V16)
  static bool goodCell(int u, int v, int N, int type, int rotn);
  // Decides if the cell is present or not (for v17)
  static bool goodCell(int u, int v, int waferType);
  // Converts rotation index (as otained from flat file) depending on
  // zside and type (for V15, V16)
  static int getRotation(int zside, int type, int rotn);
  // Get partial wafer type and orientation (for V15, V16)
  static std::pair<int, int> getTypeMode(const double& xpos,
                                         const double& ypos,
                                         const double& delX,
                                         const double& delY,
                                         const double& rin,
                                         const double& rout,
                                         const int& waferType,
                                         const int& mode,
                                         bool debug = false);
  // Checks partial wafer type and orientation (for V15, V16)
  static bool goodTypeMode(double xpos,
                           double ypos,
                           double delX,
                           double delY,
                           double rin,
                           double rout,
                           int part,
                           int rotn,
                           bool debug = false);
  // Gets the corners of the partial wafers from its type, orientation, zside
  // (Good for V15, V16 geometries)
  static std::vector<std::pair<double, double> > waferXY(
      int part, int orient, int zside, double delX, double delY, double xpos, double ypos);
  // Gets the corners of the partial wafers from its type, placement index
  // (Good for V17 geometry)
  static std::vector<std::pair<double, double> > waferXY(
      int part, int placement, double delX, double delY, double xpos, double ypos);

  // Condition for cells in partial wafer: arr[0]*u + arr[1]*v <= arr[2]
  static constexpr std::array<int, 3> edgeWaferLDTop = {{1, 0, 7}};
  static constexpr std::array<int, 3> edgeWaferLDBottom = {{-1, 0, -8}};
  static constexpr std::array<int, 3> edgeWaferLDLeft = {{-1, 2, 7}};
  static constexpr std::array<int, 3> edgeWaferLDRight = {{1, -2, -7}};
  static constexpr std::array<int, 3> edgeWaferLDFive = {{-1, 2, 15}};
  static constexpr std::array<int, 3> edgeWaferLDThree = {{1, -2, -15}};
  static constexpr std::array<int, 3> edgeWaferHDTop = {{1, 0, 9}};
  static constexpr std::array<int, 3> edgeWaferHDBottom = {{-1, 0, -10}};
  static constexpr std::array<int, 3> edgeWaferHDLeft = {{-1, 2, 4}};
  static constexpr std::array<int, 3> edgeWaferHDRight = {{1, -2, -18}};
  static constexpr std::array<int, 3> edgeWaferHDFive = {{-1, 2, 18}};

  static constexpr int k_OffsetRotation = 10;
};

#endif
