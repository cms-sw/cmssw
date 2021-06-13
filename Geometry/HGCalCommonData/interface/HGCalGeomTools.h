#ifndef Geometry_HGCalCommonData_HGCalGeomTools_h
#define Geometry_HGCalCommonData_HGCalGeomTools_h

#include <cmath>
#include <cstdint>
#include <vector>

class HGCalGeomTools {
public:
  HGCalGeomTools();
  ~HGCalGeomTools() {}

  static constexpr int k_allCorners = 6;
  static constexpr int k_fiveCorners = 5;
  static constexpr int k_fourCorners = 4;
  static constexpr int k_threeCorners = 3;
  static constexpr int k_twoCorners = 2;

  static void radius(double zf,
                     double zb,
                     std::vector<double> const& zFront1,
                     std::vector<double> const& rFront1,
                     std::vector<double> const& slope1,
                     std::vector<double> const& zFront2,
                     std::vector<double> const& rFront2,
                     std::vector<double> const& slope2,
                     int flag,
                     std::vector<double>& zz,
                     std::vector<double>& rin,
                     std::vector<double>& rout);
  static double radius(double z,
                       std::vector<double> const& zFront,
                       std::vector<double> const& rFront,
                       std::vector<double> const& slope);
  static double radius(
      double z, int layer0, int layerf, std::vector<double> const& zFront, std::vector<double> const& rFront);
  std::pair<double, double> shiftXY(int waferPosition, double waferSize) const;
  static double slope(double z, std::vector<double> const& zFront, std::vector<double> const& slope);
  static std::pair<double, double> zradius(double z1,
                                           double z2,
                                           std::vector<double> const& zFront,
                                           std::vector<double> const& rFront);
  static std::pair<int32_t, int32_t> waferCorner(
      double xpos, double ypos, double r, double R, double rMin, double rMax, bool oldBug = false);

private:
  static constexpr double tol_ = 0.0001;
  double factor_;
};

#endif
