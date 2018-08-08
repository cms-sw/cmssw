#include <iostream>
#include "DataFormats/Math/interface/SSEVec.h"

int main() {
{
  mathSSE::Vec4<float> yAxis(-0.0144846,0.932024,-0.362108);
  mathSSE::Vec4<float> zAxis(-0.204951,0.351689,0.913406);
    
  auto xAxis = ::cross(yAxis,zAxis);

  const mathSSE::Vec4<float> correctXAxis(0.978666,0.0874447,0.185925);
   std::cout <<" x axis "<<xAxis<<std::endl;
   if ( abs(xAxis.o.theX-correctXAxis.o.theX )> 0.000001 or
        abs(xAxis.o.theY-correctXAxis.o.theY) > 0.000001 or
        abs(xAxis.o.theZ-correctXAxis.o.theZ) > 0.000001) {
     std::cout <<"BAD since not same as "<<correctXAxis<<std::endl;
     return 1;
   }
}
{
  mathSSE::Vec4<double> yAxis(-0.0144846,0.932024,-0.362108);
  mathSSE::Vec4<double> zAxis(-0.204951,0.351689,0.913406);

  auto xAxis = ::cross(yAxis,zAxis);

  const mathSSE::Vec4<float> correctXAxis(0.978666,0.0874447,0.185925);
   std::cout <<" x axis "<<xAxis<<std::endl;
   if ( abs(xAxis.o.theX-correctXAxis.o.theX )> 0.000001 or
        abs(xAxis.o.theY-correctXAxis.o.theY) > 0.000001 or
        abs(xAxis.o.theZ-correctXAxis.o.theZ) > 0.000001) {
     std::cout <<"BAD since not same as "<<correctXAxis<<std::endl;
     return 1;
   }
}


   return 0;
}

