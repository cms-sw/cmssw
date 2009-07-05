#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h" 


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

void st(){}
void en(){}

int main() {

  RectangularPlaneBounds bound(1.,1.,1);

  Local3DPoint a(10.,10.,10.);

  Local3DPoint in(0.,0.,0.);

  Local3DPoint outY(0.,3.,0.);

  Local3DPoint outZ(0.,3.,3.);

  LocalError err(0.1,0.0,0.1);

  bool ok;

  // usual first to load whatever
  {
    edm::HRTimeType s= edm::hrRealTime();
    ok = bound.inside(a);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "not inside?" << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(outZ);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "not inside?" << std::endl;
  }

 {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(in);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!ok) std::cout << "not inside?" << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(in,err);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!ok) std::cout << "not inside?" << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(outY,err);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "inside?" << std::endl;
  }

  
  return 0;

}
