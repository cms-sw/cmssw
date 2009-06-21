#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h" 


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

int main() {

  RectangularPlaneBounds bound(1.,1.,1);

  Local3DPoint in(0.,0.,0.);

  Local3DPoint outY(0.,3.,0.);

  LocalError err(0.1,0.0,0.1);

  bool ok;

  {
    edm::HRTimeType s= edm::hrRealTime();
    ok = bound.inside(in,err);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    ok = bound.inside(outY,err);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
  }

  
  return 0;

}
