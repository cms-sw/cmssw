#ifndef HcalAlgos_HcalPulseContainmentAlgo_h
#define HcalAlgos_HcalPulseContainmentAlgo_h

#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShape.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalShapeIntegrator.h"

class HcalTimeSlew;

class HcalPulseContainmentAlgo {
public:
  HcalPulseContainmentAlgo(const HcalPulseShape* shape,
                           int num_samples,
                           double fixedphase_ns,
                           bool phaseAsInSim,
                           const HcalTimeSlew* hcalTimeSlew_delay);
  HcalPulseContainmentAlgo(int num_samples,
                           double fixedphase_ns,
                           bool phaseAsInSim,
                           const HcalTimeSlew* hcalTimeSlew_delay);
  std::pair<double, double> calcpair(double);

private:
  void init(int num_samples);
  double fixedphasens_;
  double integrationwindowns_;
  double time0shiftns_;
  bool phaseAsInSim_;
  HcalShapeIntegrator integrator_;
  const HcalTimeSlew* hcalTimeSlew_delay_;
};

#endif
