#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>
#include <iostream>

void HcalTimeSlew::addM2ParameterSet(float tzero, float slope, float tmax) {
  parametersM2_.emplace_back(tzero, slope, tmax);
}

void HcalTimeSlew::addM3ParameterSet(double cap,
                                     double tspar0,
                                     double tspar1,
                                     double tspar2,
                                     double tspar0_siPM,
                                     double tspar1_siPM,
                                     double tspar2_siPM) {
  parametersM3_.emplace_back(cap, tspar0, tspar1, tspar2, tspar0_siPM, tspar1_siPM, tspar2_siPM);
}

// Used by M2/Simulation
float HcalTimeSlew::delay(float fC, BiasSetting bias) const {
  float rawDelay = parametersM2_[bias].tzero + parametersM2_[bias].slope * std::log(fC);
  return (rawDelay < 0) ? (0) : ((rawDelay > parametersM2_[bias].tmax) ? (parametersM2_[bias].tmax) : (rawDelay));
}

// Used by M3
double HcalTimeSlew::delay(double fC, ParaSource source, BiasSetting bias, bool isHPD) const {
  double rawDelay = 0.0;
  if (source == TestStand) {
    return HcalTimeSlew::delay(fC, bias);
  } else if (isHPD) {
    rawDelay = std::fmin(
        parametersM3_[source].cap,
        parametersM3_[source].tspar0 + parametersM3_[source].tspar1 * std::log(fC + parametersM3_[source].tspar2));
  } else {
    rawDelay = parametersM3_[source].cap + parametersM3_[source].tspar0_siPM;
  }
  return rawDelay;
}
