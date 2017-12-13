#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>
#include <iostream>

void HcalTimeSlew::addM2ParameterSet(double tzero, double slope, double tmax){
  //std::cout<<"M2 ParameterSet: tzero = "<<tzero<<"  slope = "<<slope<<"  tmax = "<<tmax<<std::endl;
  parametersM2_.emplace_back(tzero,slope,tmax);
}

void HcalTimeSlew::addM3ParameterSet(double cap, double tspar0, double tspar1, double tspar2, double tspar0_siPM, double tspar1_siPM, double tspar2_siPM){
  //std::cout<<"M3 ParameterSet: cap = "<<cap<<"  tspar0 = "<<tspar0<<" tspar1 = "<<tspar1<<"  tspar2 = "<<tspar2<<" tspar0_siPM = "<<tspar0_siPM<<"  tspar1_siPM = "<<tspar1_siPM<<" tspar2_siPM = "<<tspar2_siPM<<std::endl;
  parametersM3_.emplace_back(cap, tspar0, tspar1, tspar2, tspar0_siPM, tspar1_siPM, tspar2_siPM);
}

// used by M2
double HcalTimeSlew::delay(double fC, BiasSetting bias) const {
  //std::cout<<"M2 delay: charge = "<<fC<<"  bias = "<<bias<<std::endl;
  double rawDelay = parametersM2_[bias].tzero + parametersM2_[bias].slope*log(fC);
  //std::cout<<"tzero = "<<parametersM2_[bias].tzero<<"  slope  = "<<parametersM2_[bias].slope<<"  tmax = "<<parametersM2_[bias].tmax<<std::endl;
  return (rawDelay < 0)?(0):((rawDelay > parametersM2_[bias].tmax)?(parametersM2_[bias].tmax):(rawDelay));
}

// used by M3
double HcalTimeSlew::delay(double fC, ParaSource source, bool isHPD) const {
  //std::cout<<"M3 delay: charge = "<<fC<<" source = "<<source<<" isHPD = "<<isHPD<<std::endl;
  //std::cout<<"cap = "<<parametersM3_[1].cap<<" tspar0 = "<<parametersM3_[1].tspar0<<" tspar1 = "<<parametersM3_[1].tspar1<<"  tspar2 = "<<parametersM3_[1].tspar2<<" tspar0_siPM = "<<parametersM3_[1].tspar0_siPM<<"  tspar1_siPM = "<<parametersM3_[1].tspar1_siPM<<" tspar2_siPM = "<<parametersM3_[1].tspar2_siPM<<std::endl;
  if(fC < 0) return 0;
  else{
    if(isHPD) return std::fmin(parametersM3_[source].cap,parametersM3_[source].tspar0+parametersM3_[source].tspar1*log(fC+parametersM3_[source].tspar2));
    return parametersM3_[source].cap+parametersM3_[source].tspar0_siPM;
  }
}
