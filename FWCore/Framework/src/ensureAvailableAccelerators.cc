#include "FWCore/Framework/interface/ensureAvailableAccelerators.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <vector>

namespace edm {
  void ensureAvailableAccelerators(edm::ParameterSet const& parameterSet) {
    ParameterSet const& optionsPset(parameterSet.getUntrackedParameterSet("options"));
    auto accelerators = optionsPset.getUntrackedParameter<std::vector<std::string>>("accelerators");
    if (not accelerators.empty()) {
      auto const& availableAccelerators =
          parameterSet.getUntrackedParameter<std::vector<std::string>>("@available_accelerators");
      std::sort(accelerators.begin(), accelerators.end());
      std::vector<std::string> unavailableAccelerators;
      std::set_difference(accelerators.begin(),
                          accelerators.end(),
                          availableAccelerators.begin(),
                          availableAccelerators.end(),
                          std::back_inserter(unavailableAccelerators));
      if (not unavailableAccelerators.empty()) {
        Exception ex(errors::UnavailableAccelerator);
        ex << "Compute accelerators ";
        bool first = true;
        for (auto const& acc : unavailableAccelerators) {
          if (not first) {
            ex << ", ";
          } else {
            first = true;
          }
          ex << acc;
        }
        ex << " were requested but are not available in this system.";
        throw ex;
      }
    }
  }
}  // namespace edm
