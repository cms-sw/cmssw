// Original Author:  Jan Kašpar

#ifndef CondFormats_CTPPSReadoutObjects_LHCInterpolatedOpticalFunctionsSetCollection_h
#define CondFormats_CTPPSReadoutObjects_LHCInterpolatedOpticalFunctionsSetCollection_h

#include "CondFormats/CTPPSReadoutObjects/interface/LHCInterpolatedOpticalFunctionsSet.h"

#include <unordered_map>

class LHCInterpolatedOpticalFunctionsSetCollection
    : public std::unordered_map<unsigned int, LHCInterpolatedOpticalFunctionsSet> {};

#endif
