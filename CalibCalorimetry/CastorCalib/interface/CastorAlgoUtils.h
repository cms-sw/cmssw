#ifndef DQM_CASTORALGOUTILS_H
#define DQM_CASTORALGOUTILS_H

#include "CalibFormats/CastorObjects/interface/CastorDbService.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"

void getLinearizedADC(const CastorQIEShape& shape,
		      const CastorQIECoder* coder,
		      int bins,int capid,
		      float& lo,
		      float& hi);

float maxDiff(float one, float two, float three, float four);


#endif
