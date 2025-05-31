#ifndef Geometry_HGCalCommonData_HGCGuardRing_h
#define Geometry_HGCalCommonData_HGCGuardRing_h

#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "G4ThreeVector.hh"

#include <vector>

class HGCGuardRing {
public:
  HGCGuardRing(const HGCalDDDConstants& hgc);
  bool exclude(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV);
  bool excludePartial(G4ThreeVector& point, int zside, int frontBack, int layer, int waferU, int waferV);
  static bool insidePolygon(double x, double y, const std::vector<std::pair<double, double> >& xyv);

private:
  static constexpr double sqrt3_ = 1.732050807568877;  // std::sqrt(3.0) in double precision
  const HGCalDDDConstants& hgcons_;
  const HGCalGeometryMode::GeometryMode modeUV_;
  const bool v17OrLess_;
  const double waferSize_, sensorSizeOffset_, guardRingOffset_;
  static constexpr std::array<double, 12> tan_1 = {
      {-sqrt3_, sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0}};
  static constexpr std::array<double, 12> cos_1 = {{0.5, -0.5, -1.0, -0.5, 0.5, 1.0, -0.5, 0.5, 1.0, 0.5, -0.5, -1.0}};
  static constexpr std::array<double, 12> cot_1 = {
      {sqrt3_, -sqrt3_, 0.0, sqrt3_, -sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0, -sqrt3_, sqrt3_, 0.0}};
  double offset_, offsetPartial_, xmax_, ymax_, c22_, c27_;
};

#endif  // HGCGuardRing_h
