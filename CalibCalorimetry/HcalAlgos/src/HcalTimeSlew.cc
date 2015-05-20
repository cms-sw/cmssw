#include "CalibCalorimetry/HcalAlgos/interface/HcalTimeSlew.h"
#include <cmath>

static const double tzero[3]= {23.960177, 13.307784, 9.109694};
static const double slope[3] = {-3.178648,  -1.556668, -1.075824 };
static const double tmax[3] = {16.00, 10.00, 6.25 };
static const double cap = 6.0;
static const double tspar0[2] = {10.2627, 9.27638};
static const double tspar1[2] = {-2.41281,-2.05585};

double HcalTimeSlew::delay(double fC, BiasSetting bias) {
  double rawDelay=tzero[bias]+slope[bias]*log(fC);
  return (rawDelay<0)?(0):((rawDelay>tmax[bias])?(tmax[bias]):(rawDelay));			   
}

double HcalTimeSlew::delay(double fC, ParaSource source, BiasSetting bias, double par0, double par1) {

  if (source==TestStand) {
    return HcalTimeSlew::delay(fC, bias);
  }
  else if (source==InputPars) {
    return std::min(cap, par0 + par1*log(fC));
  }
  else if (source==Data || source==MC){
    return std::min(cap,tspar0[source-1]+tspar1[source-1]*log(fC));
  }
  return 0;
}
