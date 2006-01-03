#ifndef DQM_HCALALGOUTILS_H
#define DQM_HCALALGOUTILS_H

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

void getLinearizedADC(const HcalQIEShape& shape,
		      const HcalQIECoder* coder,
		      int bins,int capid,
		      float& lo,
		      float& hi){
  
  float low = coder->charge(shape,0,capid); 
  float high = coder->charge(shape,bins-1,capid);
  //  printf("Coder: lo: %f, hi: %f, bins: %d, capid: %d\n",low,high,bins,capid);
  float step = (high-low)/(bins-1);
  low -= step/2.0; high += step/2.0;
  lo = low; hi = high;
  return;
}

float maxDiff(float one, float two, float three, float four){
  float max=-1000; float min = 1000;
  if(one>max) max = one;      if(one<min) min = one;
  if(two>max) max = two;      if(two<min) min = two;
  if(three>max) max = three;  if(three<min) min = three;
  if(four>max) max = four;    if(four<min) min = four;
  return fabs(max-min);
}

#endif
