#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>
#include <iostream>

void HcalTimeSlew::addM2ParameterSet(double tzero, double slope, double tmax){
  parametersM2_.emplace_back(tzero,slope,tmax);
}

void HcalTimeSlew::addM3ParameterSet(double cap, double tspar0, double tspar1, double tspar2, double tspar0_siPM, double tspar1_siPM, double tspar2_siPM){
  parametersM3_.emplace_back(cap, tspar0, tspar1, tspar2, tspar0_siPM, tspar1_siPM, tspar2_siPM);
}

// used by M2
double HcalTimeSlew::delay(double fC, BiasSetting bias) const {
  std::cout<<"M2 delay: bias = "<<bias<<std::endl;
  double rawDelay = parametersM2_[bias].tzero + parametersM2_[bias].slope*log(fC);
  return (rawDelay < 0)?(0):((rawDelay > parametersM2_[bias].tmax)?(parametersM2_[bias].tmax):(rawDelay));
}

// used by M3
double HcalTimeSlew::delay(double fC, ParaSource source, BiasSetting bias, double par0, double par1, double par2, bool isHPD) const {
  std::cout<<"M3 delay: bias = "<<bias<<std::endl;
  if (source==TestStand) {
    return HcalTimeSlew::delay(fC, bias);
  }
  else if (source==InputPars) {
    if(isHPD) return std::fmin(parametersM3_[0].cap, par0 + par1*log(fC+par2));
    return parametersM3_[0].cap+parametersM3_[0].tspar0_siPM;
  }
  else if (source==Data || source==MC){
    if(isHPD) return std::fmin(parametersM3_[source-1].cap,parametersM3_[source-1].tspar0+parametersM3_[source-1].tspar1*log(fC+parametersM3_[source-1].tspar2));
    return parametersM3_[source-1].cap+parametersM3_[source-1].tspar0_siPM;
  }
  return 0;
}
