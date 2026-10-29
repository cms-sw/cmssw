#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Services/plugins/ProcSmaps.h"

#include <cstdint>
#include <sstream>

namespace edm::service {
  class LibraryMemoryReport {
  public:
    LibraryMemoryReport(edm::ParameterSet const&, edm::ActivityRegistry&);

    void postModuleConstruction(edm::ModuleDescription const&) const;
    void postEndJob() const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    bool allModules_;
  };

  LibraryMemoryReport::LibraryMemoryReport(edm::ParameterSet const& iPS, edm::ActivityRegistry& registry)
      : allModules_(iPS.getUntrackedParameter<bool>("allModules")) {
    registry.watchPostModuleConstruction(this, &LibraryMemoryReport::postModuleConstruction);
    registry.watchPostEndJob(this, &LibraryMemoryReport::postEndJob);
  }

  void LibraryMemoryReport::postModuleConstruction(edm::ModuleDescription const& module) const {
    auto const smaps = readProcSmaps();
    std::uint64_t totalSizeKB = 0;
    std::uint64_t totalRssKB = 0;
    std::uint64_t totalPssKB = 0;
    std::uint64_t totalExecutableSizeKB = 0;
    std::ostringstream report;
    report << "LibraryMemoryReport moduleLabel=\"" << module.moduleLabel() << "\" moduleType=\"" << module.moduleName()
           << "\"";

    for (auto const& library : smaps.libraries) {
      if (allModules_) {
        report << "\n library path=\"" << library.path << "\" sizeKB=" << library.sizeKB << " rssKB=" << library.rssKB
               << " pssKB=" << library.pssKB << " executableSizeKB=" << library.executableSizeKB;
      }
      totalSizeKB += library.sizeKB;
      totalRssKB += library.rssKB;
      totalPssKB += library.pssKB;
      totalExecutableSizeKB += library.executableSizeKB;
    }

    report << "\n summary libraries=" << smaps.libraries.size() << " sizeKB=" << totalSizeKB << " rssKB=" << totalRssKB
           << " pssKB=" << totalPssKB << " executableSizeKB=" << totalExecutableSizeKB;
    edm::LogAbsolute("LibraryMemoryReport") << report.str();
  }

  void LibraryMemoryReport::postEndJob() const {
    auto const smaps = readProcSmaps();
    std::uint64_t totalSizeKB = 0;
    std::uint64_t totalRssKB = 0;
    std::uint64_t totalPssKB = 0;
    std::uint64_t totalExecutableSizeKB = 0;
    std::ostringstream report;
    report << "LibraryMemoryReport summary";
    for (auto const& library : smaps.libraries) {
      report << "\n library path=\"" << library.path << "\" sizeKB=" << library.sizeKB << " rssKB=" << library.rssKB
             << " pssKB=" << library.pssKB << " executableSizeKB=" << library.executableSizeKB;
      totalSizeKB += library.sizeKB;
      totalRssKB += library.rssKB;
      totalPssKB += library.pssKB;
      totalExecutableSizeKB += library.executableSizeKB;
    }
    report << "\nLibraryMemoryReport summary libraries=" << smaps.libraries.size() << " sizeKB=" << totalSizeKB
           << " rssKB=" << totalRssKB << " pssKB=" << totalPssKB
           << " executableSizeKB=" << totalExecutableSizeKB;
    edm::LogAbsolute("LibraryMemoryReport") << report.str();
  }

  void LibraryMemoryReport::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.addUntracked<bool>("allModules", false)
        ->setComment(
            "If true the individual memory report will be printed for each module, otherwise only the summary will be "
            "printed.");
    descriptions.add("LibraryMemoryReport", description);
  }

}  // namespace  edm::service

using edm::service::LibraryMemoryReport;
DEFINE_FWK_SERVICE(LibraryMemoryReport);
