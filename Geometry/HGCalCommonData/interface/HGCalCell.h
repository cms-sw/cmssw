#ifndef Geometry_HGCalCommonData_HGCalCell_h
#define Geometry_HGCalCommonData_HGCalCell_h

#include <cmath>
#include <cstdint>

class HGCalCell {
public:
  HGCalCell(double waferSize, int32_t nFine, int32_t nCoarse);

  static constexpr int32_t waferOrient0 = 0;
  static constexpr int32_t waferOrient1 = 1;
  static constexpr int32_t waferOrient2 = 2;
  static constexpr int32_t waferOrient3 = 3;
  static constexpr int32_t waferOrient4 = 4;
  static constexpr int32_t waferOrient5 = 5;

  static constexpr int32_t cellPlacementIndex0 = 0;
  static constexpr int32_t cellPlacementIndex1 = 1;
  static constexpr int32_t cellPlacementIndex2 = 2;
  static constexpr int32_t cellPlacementIndex3 = 3;
  static constexpr int32_t cellPlacementIndex4 = 4;
  static constexpr int32_t cellPlacementIndex5 = 5;
  static constexpr int32_t cellPlacementIndex6 = 6;
  static constexpr int32_t cellPlacementIndex7 = 7;
  static constexpr int32_t cellPlacementIndex8 = 8;
  static constexpr int32_t cellPlacementIndex9 = 9;
  static constexpr int32_t cellPlacementIndex10 = 10;
  static constexpr int32_t cellPlacementIndex11 = 11;

  static constexpr int32_t cellPlacementExtra = 6;
  static constexpr int32_t cellPlacementOld = 7;
  static constexpr int32_t cellPlacementTotal = 12;

  static constexpr int32_t fullCell = 0;
  static constexpr int32_t cornerCell = 1;
  static constexpr int32_t truncatedCell = 2;
  static constexpr int32_t extendedCell = 3;

  std::pair<double, double> HGCalCellUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  std::pair<double, double> HGCalCellUV2XY2(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  std::pair<int32_t, int32_t> HGCalCellUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  static int32_t HGCalCellPlacementIndex(int32_t iz, int32_t fwdBack, int32_t orient);

private:
  const double sqrt3By2_ = (0.5 * std::sqrt(3.0));
  int32_t ncell_[2];
  double cellX_[2], cellY_[2];
};

#endif
