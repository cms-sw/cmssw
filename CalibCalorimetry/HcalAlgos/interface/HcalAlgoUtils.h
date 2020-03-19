#ifndef DQM_HCALALGOUTILS_H
#define DQM_HCALALGOUTILS_H

#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

void getLinearizedADC(const HcalQIEShape& shape, const HcalQIECoder* coder, int bins, int capid, float& lo, float& hi);

float maxDiff(float one, float two, float three, float four);

#endif
