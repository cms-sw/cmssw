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
                                         const bool& v17,
                                         const bool& debug = false);
  // Checks partial wafer type and orientation (for V15, V16)
  static bool goodTypeMode(const double& xpos,
                           const double& ypos,
                           const double& delX,
                           const double& delY,
                           const double& rin,
                           const double& rout,
                           const int& part,
                           const int& rotn,
                           const bool& v17,
                           const bool& debug = false);
  // Gets the corners of the partial wafers from its type, orientation, zside
  // (Good for V15, V16 geometries)
  static std::vector<std::pair<double, double> > waferXY(const int& part,
                                                         const int& orient,
                                                         const int& zside,
                                                         const double& waferSize,
                                                         const double& offset,
                                                         const double& xpos,
                                                         const double& ypos,
                                                         const bool& v17);
  // Gets the corners of the partial wafers from its type, placement index
  // (Good for V17 geometry)
  static std::vector<std::pair<double, double> > waferXY(const int& part,
                                                         const int& placement,
                                                         const double& wafersize,
                                                         const double& offset,
                                                         const double& xpos,
                                                         const double& ypos,
                                                         const bool& v17);

  static std::array<double, 4> maskCut(
      const int& part, const int& place, const double& waferSize, const double& offset, const bool& v17OrLess);

private:
  static constexpr double sqrt3_ = 1.732050807568877;  // std::sqrt(3.0) in double precision
  static constexpr double sin_60_ = 0.5 * sqrt3_;
  static constexpr double cos_60_ = 0.5;
  static constexpr double tan_60_ = sqrt3_;
  static constexpr std::array<double, 12> tan_1 = {
      {-sqrt3_, sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0}};
  static constexpr std::array<double, 12> cos_1 = {{0.5, -0.5, -1.0, -0.5, 0.5, 1.0, -0.5, 0.5, 1.0, 0.5, -0.5, -1.0}};
  static constexpr std::array<double, 12> cot_1 = {
      {sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0}};
};

#endif
