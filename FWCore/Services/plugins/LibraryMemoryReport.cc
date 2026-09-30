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
    LibraryMemoryReport(ParameterSet const&, ActivityRegistry& registry) {
      registry.watchPostModuleConstruction(this, &LibraryMemoryReport::postModuleConstruction);
    }

    static void fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription description;
      descriptions.add("LibraryMemoryReport", description);
    }

  private:
    void postModuleConstruction(ModuleDescription const& module) const {
      auto const smaps = readProcSmaps();
      std::uint64_t totalSizeKB = 0;
      std::uint64_t totalRssKB = 0;
      std::uint64_t totalPssKB = 0;
      std::uint64_t totalExecutableSizeKB = 0;
      std::ostringstream report;
      report << "LibraryMemoryReport moduleLabel=\"" << module.moduleLabel() << "\" moduleType=\""
             << module.moduleName() << "\" libraries=" << smaps.libraries.size();

      for (auto const& library : smaps.libraries) {
        report << "\n library path=\"" << library.path << "\" sizeKB=" << library.sizeKB
               << " rssKB=" << library.rssKB << " pssKB=" << library.pssKB
               << " executableSizeKB=" << library.executableSizeKB;
        totalSizeKB += library.sizeKB;
        totalRssKB += library.rssKB;
        totalPssKB += library.pssKB;
        totalExecutableSizeKB += library.executableSizeKB;
      }

      report << "\n summary libraries=" << smaps.libraries.size() << " sizeKB=" << totalSizeKB
             << " rssKB=" << totalRssKB << " pssKB=" << totalPssKB
             << " executableSizeKB=" << totalExecutableSizeKB;
      LogAbsolute("LibraryMemoryReport") << report.str();
    }
  };
}  // namespace edm::service

using edm::service::LibraryMemoryReport;
DEFINE_FWK_SERVICE(LibraryMemoryReport);
