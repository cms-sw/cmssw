#include <iostream>


namespace geometryDetails {
  std::ostream & print3D(std::ostream& s, double x, double y, double z) {
    return s << " (" << x << ',' << y << ',' << z << ") ";
  } 
  std::ostream & print2D(std::ostream& s, double x, double y) {
    return s << " (" << x << ',' << y << ") ";
  } 

}
