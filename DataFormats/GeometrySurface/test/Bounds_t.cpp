#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/LocalError.h" 


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

void st(){}
void en(){}

int main() {

  RectangularPlaneBounds bound(1.,1.,1.);

  Local3DPoint a(10.,10.,10.);

  Local3DPoint in(0.,0.,0.);
  Local3DPoint in2(0.5,0.,0.);

  Local3DPoint outY(0.,3.,0.);
  Local3DPoint out2(0.,1.5,0.);

  Local3DPoint outZ(0.,3.,3.);

  LocalError err(0.1,0.0,0.1);
  LocalError errB(1.0,0.0,1.0);

  bool ok;

  // usual first to load whatever
  {
    edm::HRTimeType s= edm::hrRealTime();
    ok = bound.inside(a);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "inside?" << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(outZ);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "inside?" << std::endl;
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
    auto io = bound.inout(in,err);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!io.first) std::cout << "in not inside?" << std::endl;
    if (io.second) std::cout << "in outside?" << std::endl;
  }

 {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    auto io = bound.inout(in2,errB);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!io.first) std::cout << "in2 not inside?" << std::endl;
    if (!io.second) std::cout << "in2 not outside?" << std::endl;
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

 {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    auto io = bound.inout(outY,err);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (io.first) std::cout << "outY inside?" << std::endl;
    if (!io.second) std::cout << "outY not outside?" << std::endl;
  }

 {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    auto io = bound.inout(out2,errB);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!io.first) std::cout << "out2 not inside?" << std::endl;
    if (!io.second) std::cout << "out2 not outside?" << std::endl;
  }


  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(Local2DPoint(in.x(),in.y()),1.f);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!ok) std::cout << "not inside?" << std::endl;
  }

  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(Local2DPoint(outY.x(),outY.y()),1.f);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (ok) std::cout << "inside?" << std::endl;
  }
  {
    edm::HRTimeType s= edm::hrRealTime();
    st();
    ok = bound.inside(Local2DPoint(outY.x(),outY.y()),10.f);
    en();
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    if (!ok) std::cout << "not inside?" << std::endl;
  }

  
  return 0;

}
