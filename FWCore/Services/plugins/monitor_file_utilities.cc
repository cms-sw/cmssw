#include "monitor_file_utilities.h"
#include "FWCore/Utilities/interface/OStreamColumn.h"

#include <vector>

namespace {
  std::string const space{"  "};
}

namespace edm::service::monitor_file_utilities {
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

  void moduleIdToLabel(std::ostream& oStream,
                       std::vector<std::pair<std::string, std::string>> const& iModuleLabels,
                       char moduleIdSymbol,
                       std::string const& iIDHeader,
                       std::string const& iLabelHeader,
                       std::string const& iTypeHeader) {
    std::size_t const width{std::to_string(iModuleLabels.size()).size()};
    OStreamColumn col0{iIDHeader, width};
    std::size_t labelWidth = 0;
    std::size_t typeWidth = 0;
    for (auto const& labelType : iModuleLabels) {
      if (labelWidth < labelType.first.size()) {
        labelWidth = labelType.first.size();
      }
      if (typeWidth < labelType.second.size()) {
        typeWidth = labelType.second.size();
      }
    }
    OStreamColumn labelCol{iLabelHeader, labelWidth};
    OStreamColumn typeCol{iTypeHeader, typeWidth};

    oStream << "\n#  " << col0 << space << std::left << labelCol << space << std::left << typeCol << '\n';
    oStream << "#  "
            << std::string(col0.width() + space.size() + labelCol.width() + space.size() + typeCol.width(), '-')
            << '\n';

    for (std::size_t i{}; i < iModuleLabels.size(); ++i) {
      auto const& label = iModuleLabels[i];
      if (not label.first.empty()) {
        oStream << '#' << moduleIdSymbol << ' ' << std::left << col0(i) << space << std::left << labelCol(label.first)
                << space << std::left << label.second << '\n';
      }
    }
    oStream << '\n';
  }
}  // namespace edm::service::monitor_file_utilities
