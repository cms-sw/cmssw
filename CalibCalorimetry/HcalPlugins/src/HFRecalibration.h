#ifndef CalibCalorimetry_HFRecalibration_h
#define CalibCalorimetry_HFRecalibration_h
//
// Simple class with parameterized function provided by James Wetzel  
// to compansate for darkening of HF QP fibers   
// in Upgrade conditions
// correction = f (integrated lumi, depth, ieta)
//
#include <cmath>
#include <iostream>

class HFRecalibration {

public:
  HFRecalibration();
  ~HFRecalibration();
  double getCorr(int ieta, int idepth, double lumi);
private:

};

#endif // HFRecalibration_h
