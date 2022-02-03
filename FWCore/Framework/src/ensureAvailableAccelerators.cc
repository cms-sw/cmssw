#include "FWCore/Framework/interface/ensureAvailableAccelerators.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <vector>

namespace edm {
  void ensureAvailableAccelerators(edm::ParameterSet const& parameterSet) {
    ParameterSet const& optionsPset(parameterSet.getUntrackedParameterSet("options"));
    auto accelerators = optionsPset.getUntrackedParameter<std::vector<std::string>>("accelerators");
    if (accelerators.empty()) {
      Exception ex(errors::UnavailableAccelerator);
      ex << "The system has no compute accelerators that match the patterns specified in "
            "process.options.accelerators.\nThe following compute accelerators are available:\n";
      auto const& availableAccelerators =
          parameterSet.getUntrackedParameter<std::vector<std::string>>("@available_accelerators");
      for (auto const& acc : availableAccelerators) {
        ex << " " << acc << "\n";
      }
      throw ex;
    }
  }
}  // namespace edm
