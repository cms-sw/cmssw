// Original Author:  Jan Kašpar

#ifndef CondFormats_PPSObjects_LHCOpticalFunctionsSetCollection_h
#define CondFormats_PPSObjects_LHCOpticalFunctionsSetCollection_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/PPSObjects/interface/LHCOpticalFunctionsSet.h"

#include <map>
#include <unordered_map>

/**
 \brief Collection of optical functions for two crossing angle values and various scoring planes.
 * map: crossing angle --> (map: RP id --> optical functions)
**/
class LHCOpticalFunctionsSetCollection
    : public std::map<double, std::unordered_map<unsigned int, LHCOpticalFunctionsSet>> {
private:
  COND_SERIALIZABLE;
};

#endif
