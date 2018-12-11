#ifndef Geometry_HGCalCommonData_HGCalGeomTools_h
#define Geometry_HGCalCommonData_HGCalGeomTools_h

#include <cstdint>
#include <vector>

class HGCalGeomTools {

public:
  
  HGCalGeomTools() {}
  ~HGCalGeomTools() {}
  static double radius(double z, std::vector<double> const& zFront,
		       std::vector<double> const& rFront,
		       std::vector<double> const& slope);
  static double radius(double z, int layer0, int layerf,
		       std::vector<double> const& zFront,
		       std::vector<double> const& rFront);
  static std::pair<int32_t,int32_t> waferCorner(double xpos, double ypos,
						double r, double R, 
						double rMin, double rMax,
						bool oldBug=false);

private:
  static constexpr double tol = 0.0001;
};

#endif
