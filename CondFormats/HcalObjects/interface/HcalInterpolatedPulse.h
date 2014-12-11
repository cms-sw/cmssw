#ifndef CondFormats_HcalObjects_HcalInterpolatedPulse_h_
#define CondFormats_HcalObjects_HcalInterpolatedPulse_h_

#include "CondFormats/HcalObjects/interface/InterpolatedPulse.h"

// Use some number which is sufficient to simulate at least 13
// 25 ns time slices with 0.25 ns step (need to get at least
// 3 ts ahead of the 10 time slices digitized)
typedef InterpolatedPulse<1500U> HcalInterpolatedPulse;

#endif // CondFormats_HcalObjects_HcalInterpolatedPulse_h_
