// Original Author:  Jan Kašpar

#ifndef CondFormats_PPSObjects_LHCInterpolatedOpticalFunctionsSetCollection_h
#define CondFormats_PPSObjects_LHCInterpolatedOpticalFunctionsSetCollection_h

#include "CondFormats/PPSObjects/interface/LHCInterpolatedOpticalFunctionsSet.h"

#include <unordered_map>

class LHCInterpolatedOpticalFunctionsSetCollection
    : public std::unordered_map<unsigned int, LHCInterpolatedOpticalFunctionsSet> {};

#endif
