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

#include <cmath>
#include <vector>
#include <iostream>

class HGCalWaferType {

public:

  HGCalWaferType(const std::vector<double>& rad100, 
		 const std::vector<double>& rad200,
		 double waferSize, double zMin, int cutValue);
  ~HGCalWaferType();
  int getType(double xpos, double ypos, double zpos);
  std::pair<double,double> rLimits(double zpos);

private:

  const double              sqrt3_ = 1.0/std::sqrt(3.0);
  const std::vector<double> rad100_;
  const std::vector<double> rad200_;
  const double              waferSize_;
  const double              zMin_;
  const int                 cutValue_;
  double                    r_, R_;
};

#endif
