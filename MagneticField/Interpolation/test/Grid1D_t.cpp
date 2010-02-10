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




int main() {

  bool ok=true;
  Grid1D grid(-10.,10.,10);

  ok &= testGrid1D(grid);

  assert(ok);
  return ok ? 0 : 1;

}
