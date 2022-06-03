#ifndef Geometry_HGCalCommonData_HGCalCellUV_h
#define Geometry_HGCalCommonData_HGCalCellUV_h

#include <cstdint>
#include <iterator>
#include <map>
#include "Geometry/HGCalCommonData/interface/HGCalCell.h"

class HGCalCellUV {
public:
  HGCalCellUV(double waferSize, double separation, int32_t nFine, int32_t nCoarse);

  std::pair<int32_t, int32_t> cellUVFromXY1(
      double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const;

  std::pair<int32_t, int32_t> cellUVFromXY2(
      double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const;

  std::pair<int32_t, int32_t> cellUVFromXY3(
      double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug) const;

  std::pair<int32_t, int32_t> cellUVFromXY4(
      double xloc, double yloc, int32_t placement, int32_t type, bool extend, bool debug);

  std::pair<int32_t, int32_t> cellUVFromXY1(
      double xloc, double yloc, int32_t placement, int32_t type, int32_t partial, bool extend, bool debug) const;

private:
  std::pair<int32_t, int32_t> cellUVFromXY4(double xloc,
                                            double yloc,
                                            int ncell,
                                            double cellX,
                                            double cellY,
                                            double cellXTotal,
                                            double cellYTotal,
                                            std::map<std::pair<int, int>, std::pair<double, double> >& cellPos,
                                            bool extend,
                                            bool debug);

  static constexpr double sqrt3_ = 1.732050807568877;  // std::sqrt(3.0) in double precision
  static constexpr double sin60_ = 0.5 * sqrt3_;
  static constexpr double cos60_ = 0.5;

  int32_t ncell_[2];
  double cellX_[2], cellY_[2], cellXTotal_[2], cellYTotal_[2], waferSize;

  std::map<std::pair<int32_t, int32_t>, std::pair<double, double> > cellPosFine_[HGCalCell::cellPlacementTotal],
      cellPosCoarse_[HGCalCell::cellPlacementTotal];
};

#endif
