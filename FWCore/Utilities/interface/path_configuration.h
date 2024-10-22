#ifndef FWCore_Utilities_path_configuration_h
#define FWCore_Utilities_path_configuration_h
// -*- C++ -*-
//
// Package  :     FWCore/Utilities
// namespace:     path_configuration
//
/**\class path_configuration path_configuration.h "FWCore/Utilities/interface/path_configuration.h"

 Description: Functions used to understand Path configurations

 Usage:
    <usage>

*/
//
// Original Author:  Christopher Jones
//         Created:  Wed, 30 Mar 2022 14:59:29 GMT
//

// system include files
#include <vector>
#include <string>
#include <unordered_set>

// user include files

// forward declarations

namespace edm::path_configuration {
  ///The label is an indicator of a path scheduling construct and not an actual module.
  using SchedulingConstructLabelSet = std::unordered_set<std::string, std::hash<std::string>, std::equal_to<>>;
  SchedulingConstructLabelSet const& schedulingConstructLabels();

  //Takes the Parameter associated to a given Path and converts it to the list of modules
  // in the same order as the Path's position bits
  std::vector<std::string> configurationToModuleBitPosition(std::vector<std::string>);

  //removes any scheduling tokens from the module's label
  std::string removeSchedulingTokensFromModuleLabel(std::string iLabel);
}  // namespace edm::path_configuration

#endif
