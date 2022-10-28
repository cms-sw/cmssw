#include "FWCore/Framework/interface/ensureAvailableAccelerators.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <algorithm>
#include <vector>

namespace edm {
  void ensureAvailableAccelerators(edm::ParameterSet const& parameterSet) {
    auto const& selectedAccelerators =
        parameterSet.getUntrackedParameter<std::vector<std::string>>("@selected_accelerators");
    if (selectedAccelerators.empty()) {
      Exception ex(errors::UnavailableAccelerator);
      ex << "The system has no compute accelerators that match the patterns specified in "
            "process.options.accelerators:\n";
      ParameterSet const& optionsPset(parameterSet.getUntrackedParameterSet("options"));
      auto const& patterns = optionsPset.getUntrackedParameter<std::vector<std::string>>("accelerators");
      for (auto const& pat : patterns) {
        ex << " " << pat << "\n";
      }
      ex << "\nThe following compute accelerators are available:\n";
      auto const& availableAccelerators =
          parameterSet.getUntrackedParameter<std::vector<std::string>>("@available_accelerators");
      for (auto const& acc : availableAccelerators) {
        ex << " " << acc << "\n";
      }

      throw ex;
    }
  }
}  // namespace edm
