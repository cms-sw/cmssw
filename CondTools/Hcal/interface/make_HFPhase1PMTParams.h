#ifndef CONDTOOLS_HCAL_MAKE_HFPHASE1PMTPARAMS_H_
#define CONDTOOLS_HCAL_MAKE_HFPHASE1PMTPARAMS_H_

#include <memory>
#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_data();

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_mc();

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_dummy();

std::unique_ptr<HFPhase1PMTParams> make_HFPhase1PMTParams_test();

#endif // CONDTOOLS_HCAL_MAKE_HFPHASE1PMTPARAMS_H_
