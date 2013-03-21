#include "Grid1D.h"
#include "Grid3D.h"
#include <cassert>

namespace {
  bool testGrid1D( Grid1D const & grid)  {
    
    bool ok=true;
    
    Grid1D::Scalar f;
    int i = grid.index(7.2,f); 
    ok &= grid.inRange(i);
    
    ok &=  (8==i);
    ok &= (0.6==f);
    
    return ok;
  }
}


#include<iostream>
#include<cstdlib>
#include<cstdio>

namespace {
  void print(Grid1D grid,  Grid1D::Scalar a) {
    Grid1D::Scalar f;
    int i = grid.index(a,f); 
    ::printf("%i %f %a\n",i,f,f);
    grid.normalize(i,f);
    ::printf("%i %f %a\n",i,f,f);
  }
}

int grid1d_t() {

  bool ok=true;
  Grid1D grid(-10.,10.,11);

  print(grid, 7.2);
  print(grid, 10.);
  print(grid, -10.2);
  print(grid, 10.2);

  ok &= testGrid1D(grid);

  assert(ok? 0 : 1);
  return ok ? 0 : 1;

}



namespace {

  Grid3D const  * factory() {

    Grid1D ga(0.,10.,5);
    Grid1D gb(-10.,10.,11);
    Grid1D gc(-10.,10.,11);

    std::vector< Grid3D::BVector>  data;
    data.reserve(ga.nodes()*gb.nodes()*gc.nodes());
    for (int i=0; i<ga.nodes(); ++i) 
      for (int j=0; j<gb.nodes(); ++j) 
	for (int k=0; k<gc.nodes(); ++k) {
	  data.push_back(Grid3D::BVector(10*ga.node(i),10*gb.node(j),10*gc.node(k)));	  
	}
    
    return new Grid3D(ga,gb,gc,data);

  }

}




#include "LinearGridInterpolator3D.h"
#include <iostream>

int grid3d_t() {
  
  Grid3D const  * grid = factory();
  
  LinearGridInterpolator3D inter(*grid);
  
  std::cout << inter.interpolate(7.5,7.2,-3.4) << std::endl;
  std::cout << inter.interpolate(-0.5,10.2,-3.4) << std::endl;
  
  
  delete grid;
  return 0;
}
