#ifndef HcalAlgos_HcalPulseContainmentAlgo_h
#define HcalAlgos_HcalPulseContainmentAlgo_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalShapeIntegrator.h"

class HcalPulseContainmentAlgo {
public:
  HcalPulseContainmentAlgo(const HcalPulseShape * shape,
                      int num_samples,
                      double fixedphase_ns);
  HcalPulseContainmentAlgo(int num_samples,
                      double fixedphase_ns);
  std::pair<double,double> calcpair(double);
private:
  void init(int num_samples);
  double fixedphasens_;
  double integrationwindowns_;
  double time0shiftns_;
  HcalShapeIntegrator integrator_;
};

#endif

