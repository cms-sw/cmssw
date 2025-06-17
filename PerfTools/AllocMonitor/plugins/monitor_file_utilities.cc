#include "monitor_file_utilities.h"
#include "FWCore/Utilities/interface/OStreamColumn.h"

#include <vector>
#include <cassert>
#include <algorithm>

namespace {
  std::string const space{"  "};
}

namespace edm::moduleAlloc::monitor_file_utilities {
  void moduleIdToLabel(std::ostream& oStream,
                       std::vector<std::string> const& iModuleLabels,
                       char moduleIdSymbol,
                       std::string const& iIDHeader,
                       std::string const& iLabelHeader) {
    std::size_t const width{std::to_string(iModuleLabels.size()).size()};
    OStreamColumn col0{iIDHeader, width};
    std::string const& lastCol = iLabelHeader;

    oStream << "\n#  " << col0 << space << lastCol << '\n';
    oStream << "#  " << std::string(col0.width() + space.size() + lastCol.size(), '-') << '\n';

    for (std::size_t i{}; i < iModuleLabels.size(); ++i) {
      auto const& label = iModuleLabels[i];
      if (not label.empty()) {
        oStream << '#' << moduleIdSymbol << ' ' << std::setw(width) << std::left << col0(i) << space << std::left
                << label << '\n';
      }
    }
    oStream << '\n';
  }

  void moduleIdToLabelAndType(std::ostream& oStream,
                              std::vector<std::string> const& iModuleLabels,
                              std::vector<std::string> const& iModuleTypes,
                              char moduleIdSymbol,
                              std::string const& iIDHeader,
                              std::string const& iLabelHeader,
                              std::string const& iTypeHeader) {
    assert(iModuleLabels.size() == iModuleTypes.size());
    std::size_t const width{std::to_string(iModuleLabels.size()).size()};
    OStreamColumn col0{iIDHeader, width};
    auto itMax = std::max_element(iModuleLabels.begin(), iModuleLabels.end(), [](auto const& iL, auto const& iR) {
      return iL.size() < iR.size();
    });
    OStreamColumn labelCol{iLabelHeader, itMax->size()};
    std::string const& typeCol = iTypeHeader;

    oStream << "\n#  " << col0 << space << labelCol << space << typeCol << '\n';
    oStream << "#  " << std::string(col0.width() + space.size() + labelCol.width() + space.size() + typeCol.size(), '-')
            << '\n';

    for (std::size_t i{}; i < iModuleLabels.size(); ++i) {
      auto const& label = iModuleLabels[i];
      auto const& type = iModuleTypes[i];
      if (not label.empty()) {
        oStream << '#' << moduleIdSymbol << ' ' << std::setw(width) << std::left << col0(i) << space
                << std::setw(itMax->size()) << std::left << labelCol(label) << space << std::left << type << '\n';
      }
    }
    oStream << '\n';
  }
}  // namespace edm::moduleAlloc::monitor_file_utilities
