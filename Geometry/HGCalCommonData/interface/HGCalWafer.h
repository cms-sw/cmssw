#ifndef Geometry_HGCalCommonData_HGCalWafer_h
#define Geometry_HGCalCommonData_HGCalWafer_h

#include <cmath>
#include <cstdint>

class HGCalWafer {
public:
  HGCalWafer(double waferSize, int32_t nFine, int32_t nCoarse);

  static constexpr int32_t waferOrient0 = 0;
  static constexpr int32_t waferOrient1 = 1;
  static constexpr int32_t waferOrient2 = 2;
  static constexpr int32_t waferOrient3 = 3;
  static constexpr int32_t waferOrient4 = 4;
  static constexpr int32_t waferOrient5 = 5;

  static constexpr int32_t waferPlacementIndex0 = 0;
  static constexpr int32_t waferPlacementIndex1 = 1;
  static constexpr int32_t waferPlacementIndex2 = 2;
  static constexpr int32_t waferPlacementIndex3 = 3;
  static constexpr int32_t waferPlacementIndex4 = 4;
  static constexpr int32_t waferPlacementIndex5 = 5;
  static constexpr int32_t waferPlacementIndex6 = 6;
  static constexpr int32_t waferPlacementIndex7 = 7;
  static constexpr int32_t waferPlacementIndex8 = 8;
  static constexpr int32_t waferPlacementIndex9 = 9;
  static constexpr int32_t waferPlacementIndex10 = 10;
  static constexpr int32_t waferPlacementIndex11 = 11;

  static constexpr int32_t waferPlacementExtra = 6;
  static constexpr int32_t waferPlacementOld = 7;
  static constexpr int32_t waferPlacementTotal = 12;

  static constexpr int32_t cornerCell = 0;
  static constexpr int32_t truncatedCell = 1;
  static constexpr int32_t extendedCell = 2;

  std::pair<double, double> HGCalWaferUV2XY1(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  std::pair<double, double> HGCalWaferUV2XY2(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  std::pair<int32_t, int32_t> HGCalWaferUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  static int32_t HGCalWaferPlacementIndex(int32_t iz, int32_t fwdBack, int32_t orient);

private:
  const double sqrt3By2_ = (0.5 * std::sqrt(3.0));
  int32_t ncell_[2];
  double cellX_[2], cellY_[2];
};

#endif
