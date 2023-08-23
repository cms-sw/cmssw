// -*- C++ -*-
//
// Package:     FWCore/Utilities
// namespace  :     path_configuration
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Wed, 30 Mar 2022 15:04:35 GMT
//

// system include files
#include <algorithm>
#include <array>

// user include files
#include "FWCore/Utilities/interface/path_configuration.h"

namespace edm::path_configuration {

  SchedulingConstructLabelSet const& schedulingConstructLabels() {
    static const SchedulingConstructLabelSet s_set({{"@"}, {"#"}});
    return s_set;
  }

  //Takes the Parameter associated to a given Path and converts it to the list of modules
  // in the same order as the Path's position bits
  std::vector<std::string> configurationToModuleBitPosition(std::vector<std::string> iConfig) {
    auto const& labelsToRemove = schedulingConstructLabels();
    iConfig.erase(
        std::remove_if(iConfig.begin(),
                       iConfig.end(),
                       [&labelsToRemove](auto const& n) { return labelsToRemove.find(n) != labelsToRemove.end(); }),
        iConfig.end());
    return iConfig;
  }

  std::string removeSchedulingTokensFromModuleLabel(std::string iLabel) {
    constexpr std::array<char, 4> s_tokens = {{'!', '-', '+', '|'}};
    if (not iLabel.empty()) {
      for (auto t : s_tokens) {
        if (t == iLabel[0]) {
          iLabel.erase(0, 1);
          break;
        }
      }
    }
    return iLabel;
  }

}  // namespace edm::path_configuration
