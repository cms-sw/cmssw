#ifndef Geometry_HGCalCommonData_HGCalWafer_h
#define Geometry_HGCalCommonData_HGCalWafer_h

#include <cmath>
#include <cstdint>

class HGCalWafer {
public:
  HGCalWafer(double waferSize, int32_t nFine, int32_t nCoarse);

  static constexpr int32_t WaferOrient0 = 0;
  static constexpr int32_t WaferOrient1 = 1;
  static constexpr int32_t WaferOrient2 = 2;
  static constexpr int32_t WaferOrient3 = 3;
  static constexpr int32_t WaferOrient4 = 4;
  static constexpr int32_t WaferOrient5 = 5;

  static constexpr int32_t WaferPlacementIndex0 = 0;
  static constexpr int32_t WaferPlacementIndex1 = 1;
  static constexpr int32_t WaferPlacementIndex2 = 2;
  static constexpr int32_t WaferPlacementIndex3 = 3;
  static constexpr int32_t WaferPlacementIndex4 = 4;
  static constexpr int32_t WaferPlacementIndex5 = 5;
  static constexpr int32_t WaferPlacementIndex6 = 6;
  static constexpr int32_t WaferPlacementIndex7 = 7;
  static constexpr int32_t WaferPlacementIndex8 = 8;
  static constexpr int32_t WaferPlacementIndex9 = 9;
  static constexpr int32_t WaferPlacementIndex10 = 10;
  static constexpr int32_t WaferPlacementIndex11 = 11;

  static constexpr int32_t WaferPlacementExtra = 6;
  static constexpr int32_t WaferPlacementOld = 7;

  std::pair<double,double> HGCalWaferUV2XY(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  int32_t HGCalWaferUV2Cell(int32_t u, int32_t v, int32_t placementIndex, int32_t type);
  static int32_t HGCalWaferPlacementIndex(int32_t iz, int32_t fwdBack,  int32_t orient);

private:
  const double factor_;
  int32_t N_[2];
  double R_[2], r_[2];
};

#endif
