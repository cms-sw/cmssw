#include "MagneticField/Interpolation/src/Grid1D.h"
#include <cassert>


bool testGrid1D( Grid1D const & grid)  {

  bool ok=true;

  Grid1D::Scalar f;
  int i = grid.index(7.2,f); 
  ok &= grid.inRange(i);

  ok &=  (8==i);
  ok &= (0.6==f);

  return ok;
}



#include<iostream>
#include<cstdlib>

void print(Grid1D grid,  Grid1D::Scalar a) {
  Grid1D::Scalar f;
  int i = grid.index(a,f); 
  ::printf("%i %f %a\n",i,f,f);
  grid.normalize(i,f);
  ::printf("%i %f %a\n",i,f,f);
}

int main() {

  bool ok=true;
  Grid1D grid(-10.,10.,11);

  print(grid, 7.2);
  print(grid, 10.);
  print(grid, -10.2);
  print(grid, 10.2);

  ok &= testGrid1D(grid);

  assert(ok);
  return ok ? 0 : 1;

}
