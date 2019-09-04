#ifndef Geometry_TrackerNumberingBuilder_ExtractStringFromDDD_H
#define Geometry_TrackerNumberingBuilder_ExtractStringFromDDD_H

#include "FWCore/ParameterSet/interface/types.h"
#include <string>

/**
 * Helper function to extract a string from a SpecPar; only returns the 
 * first one and complains if more than 1 is found.
 */
template <class FilteredView>
class ExtractStringFromDDD {
public:
  static std::string getString(const std::string &, FilteredView *);
};

#endif
