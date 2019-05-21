#ifndef HcalAlgos_HcalShapeIntegrator_h
#define HcalAlgos_HcalShapeIntegrator_h

/**  This class takes an existing Shape, and
     integrates it, summing up all the values,
     each nanosecond
*/

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
#include <vector>

class HcalShapeIntegrator {
public:
  HcalShapeIntegrator(const HcalPulseShapes::Shape* aShape);
  float operator()(double startTime, double stopTime) const;

private:
  float at(double time) const;

  int nbin_;
  std::vector<float> v_;
};

#endif
