#ifndef Geometry_HGCalCommonData_HGCalGeomTools_h
#define Geometry_HGCalCommonData_HGCalGeomTools_h

#include <vector>

class HGCalGeomTools {

public:
  
  HGCalGeomTools() {}
  ~HGCalGeomTools() {}
  static double radius(double z, std::vector<double> const& zFront,
		       std::vector<double> const& rFront,
		       std::vector<double> const& slope);

};

#endif
