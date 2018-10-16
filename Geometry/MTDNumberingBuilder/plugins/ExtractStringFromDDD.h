#ifndef Geometry_MTDNumberingBuilder_ExtractStringFromDDD_H
#define Geometry_MTDNumberingBuilder_ExtractStringFromDDD_H

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "FWCore/ParameterSet/interface/types.h"
#include <string>

class DDFilteredView;

/**
 * Helper function to extract a string from a SpecPar; only returns the 
 * first one and complains if more than 1 is found.
 */
class ExtractStringFromDDD{
 public:
  static std::string getString(std::string const &,DDFilteredView*);
};

#endif
