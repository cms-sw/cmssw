#ifndef HGCalCommonData_HGCalWaferType_h
#define HGCalCommonData_HGCalWaferType_h

/** \class HGCalWaferType
 *
 * this class determines the wafer type depending on its position
 * (taken from Philippe Bloch's parametrisation)
 * rad100, rad200 parameters assume r,z to be in cm
 * xpos, ypos, zpos, zmin, waferSize are all in mm
 *
 *  $Date: 2018/03/22 00:06:50 $
 * \author Sunanda Banerjee, Fermilab <sunanda.banerjee@cern.ch>
 *
 */

#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include <cmath>
#include <vector>

class HGCalWaferType {
public:
  HGCalWaferType(const std::vector<double>& rad100,
                 const std::vector<double>& rad200,
                 double waferSize,
                 double zMin,
                 int choice,
                 unsigned int cutValue,
                 double cutFracArea);
  ~HGCalWaferType() = default;

  static int getCassette(int index, const HGCalParameters::waferInfo_map& wafers);
  static int getOrient(int index, const HGCalParameters::waferInfo_map& wafers);
  static int getPartial(int index, const HGCalParameters::waferInfo_map& wafers);
  static int getType(int index, const HGCalParameters::waferInfo_map& wafers);
  static int getType(int index, const std::vector<int>& indices, const std::vector<int>& types);
  int getType(double xpos, double ypos, double zpos);
  std::pair<double, double> rLimits(double zpos);

private:
  double areaPolygon(std::vector<double> const&, std::vector<double> const&);
  std::pair<double, double> intersection(
      int, int, std::vector<double> const&, std::vector<double> const&, double xp, double yp, double rr);

  const double sqrt3_ = 1.0 / std::sqrt(3.0);
  const std::vector<double> rad100_;
  const std::vector<double> rad200_;
  const double waferSize_;
  const double zMin_;
  const int choice_;
  const unsigned int cutValue_;
  const double cutFracArea_;
  double r_, R_;
};

#endif
