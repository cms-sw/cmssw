#include "MagneticField/Interpolation/src/Grid1D.h"
#include "MagneticField/Interpolation/src/Grid3D.h"

namespace {

  Grid3D const  * factory() {

    Grid1D ga(0.,10.,5);
    Grid1D gb(-10.,10.,11);
    Grid1D gc(-10.,10.,11);

    std::vector< Grid3D::ValueType>  data;
    data.reserve(ga.nodes()*gb.nodes()*gc.nodes());
    for (int i=0; i<ga.nodes(); ++i) 
      for (int j=0; j<gb.nodes(); ++j) 
	for (int k=0; k<gc.nodes(); ++k) {
	  data.push_back(Grid3D::ValueType(ga.node(i),gb.node(j),gc.node(k)));	  
	}
    
    return new Grid3D(ga,gb,gc,data);

  }

}




#include "MagneticField/Interpolation/src/LinearGridInterpolator3D.h"
#include <iostream>
int main() {

  Grid3D const  * grid = factory();

  LinearGridInterpolator3D inter(*grid);
  
  std::cout << inter.interpolate(7.5,7.2,-3.4) << std::endl;
  std::cout << inter.interpolate(-0.5,10.2,-3.4) << std::endl;


  delete grid;
  return 0;
}
