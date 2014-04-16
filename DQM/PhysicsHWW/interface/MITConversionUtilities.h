#ifndef MITCONVERSIONUTILITIES_H
#define MITCONVERSIONUTILITIES_H

#include <stdint.h>
#include <vector>
#include "DQM/PhysicsHWW/interface/HWW.h"

namespace HWWFunctions {

bool isMITConversion(HWW&, unsigned int elidx, 
		     int nWrongHitsMax, 
		     float probMin,
		     float dlMin,
		     bool matchCTF,
		     bool requireArbitratedMerged);

}

#endif
