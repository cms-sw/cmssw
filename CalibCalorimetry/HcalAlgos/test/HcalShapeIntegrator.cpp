#include "CalibCalorimetry/HcalAlgos/interface/HcalShapeIntegrator.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include <iostream>
#include <cassert>
#include <cmath>

int main()
{
  HcalPulseShapes shapes;
  const HcalPulseShapes::Shape & shape(shapes.hbShape());
  HcalShapeIntegrator i(&shape);
  float maxdiff = 0.;
  float maxtime = 0.;

  for(float t = -100; t < 200; t += 0.25) 
  {
    float v1 = shape.integrate(t, t+100.);
    float v2 = i(t, t+100.);
    // only print interesting quantities
    if(v1 > 0. && v1 < 1.)
    {
      float diff = fabs(v1-v2);
      if(diff > maxdiff) {
        maxdiff = diff;
        maxtime = t;
      }
    }
  }
  std::cout << "Biggest discrepancy is " << maxdiff << " at time " << maxtime << std::endl;
}
