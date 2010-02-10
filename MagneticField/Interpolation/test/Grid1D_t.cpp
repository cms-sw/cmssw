#include "MagneticField/Interpolation/src/Grid1D.h"
#include <cassert>


bool testGrid1D( Grid1D const & grid)  {

  bool ok=true;

  Grid1D::Scalar f;
  int i = grid.index(7.2,f); 
  ok &= grid.inRange(i);

  ok &=  (7==i);
  ok &= (1.2==f);

  return ok;
}



#include<iostream>
#include<cstdlib>

int main() {

  bool ok=true;
  Grid1D grid(-10.,10.,10);

  Grid1D::Scalar f;
  int i = grid.index(7.2,f); 
  ::printf("%i #f %a\n",i,f,f);

  ok &= testGrid1D(grid);

  assert(ok);
  return ok ? 0 : 1;

}
