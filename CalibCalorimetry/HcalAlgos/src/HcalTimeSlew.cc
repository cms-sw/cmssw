#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>
#include <iostream>

void HcalTimeSlew::addM2ParameterSet(double tzero, double slope, double tmax){
  parametersM2_.emplace_back(tzero,slope,tmax);
}

void HcalTimeSlew::addM3ParameterSet(double cap, double tspar0, double tspar1, double tspar2, double tspar0_siPM, double tspar1_siPM, double tspar2_siPM){
  parametersM3_.emplace_back(cap, tspar0, tspar1, tspar2, tspar0_siPM, tspar1_siPM, tspar2_siPM);
}

// Used by M2/Simulation
double HcalTimeSlew::delay(double fC, BiasSetting bias) const {  
  if(!parametersM2_.empty()){ 
    double rawDelay = parametersM2_[bias].tzero + parametersM2_[bias].slope*log(fC);
    return (rawDelay < 0)?(0):((rawDelay > parametersM2_[bias].tmax)?(parametersM2_[bias].tmax):(rawDelay));
  }
  else{//Default parameters
    double rawDelay = 13.307784 + -1.556668*log(fC);
    return (rawDelay < 0)?(0):((rawDelay > 10.00)?(10.00):(rawDelay));
  }
}

// Used by M3
double HcalTimeSlew::delay(double fC, ParaSource source, bool isHPD) const {
  double rawDelay = 0.0;
  if(!parametersM3_.empty()){
    if(isHPD){
      rawDelay = std::fmin(parametersM3_[source].cap,parametersM3_[source].tspar0+parametersM3_[source].tspar1*log(fC+parametersM3_[source].tspar2));
    }
    else{
      rawDelay = parametersM3_[source].cap+parametersM3_[source].tspar0_siPM;  
    }
  }
  else{//Default parameters
    if(isHPD){
      rawDelay = std::fmin(6.0,12.2999-2.19142*log(fC));
    }
    else{
      rawDelay = 6.0;
    }
  }
  return (rawDelay < 0)?(0):(rawDelay);
}
