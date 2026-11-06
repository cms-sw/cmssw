#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ComponentDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "ProcSmaps.h"

#include <algorithm>
#include <fstream>
#include <string>

namespace edm::service {
  class LibraryMemoryReport {
  public:
    LibraryMemoryReport(edm::ParameterSet const&, edm::ActivityRegistry&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // Prints a summary of the memory used by the loaded shared libraries.
    // If reportLibraryPaths is true the memory used by each individual library
    // is written to a text file, named after signalName with the spaces replaced
    // by underscores, in the current working directory.
    void report(std::string const& signalName) const;

    bool allModules_;
    bool collectLibraryPaths_;
  };

  LibraryMemoryReport::LibraryMemoryReport(edm::ParameterSet const& iPS, edm::ActivityRegistry& registry)
      : allModules_(iPS.getUntrackedParameter<bool>("allModules")),
        collectLibraryPaths_(iPS.getUntrackedParameter<bool>("collectLibraryPaths")) {
    registry.watchPostServicesConstruction([this]() { report("postServicesConstruction"); });
    registry.watchBeginProcessing([this]() { report("beginProcessing"); });
    registry.watchPostEndJob([this]() { report("postEndJob"); });

    // These signals lead to a printout for each module, source, and ESModule
    if (allModules_) {
      registry.watchPostModuleConstruction([this](edm::ModuleDescription const& module) {
        report("postModuleConstruction moduleLabel=\"" + module.moduleLabel() + "\" moduleType=\"" +
                   module.moduleName() + "\"");
      });
      registry.watchPostSourceConstruction([this](edm::ModuleDescription const& module) {
        report("postSourceConstruction moduleLabel=\"" + module.moduleLabel() + "\" moduleType=\"" +
                   module.moduleName() + "\"");
      });
      registry.watchPostESModuleConstruction([this](edm::eventsetup::ComponentDescription const& description) {
        report("postESModuleConstruction moduleLabel=\"" + description.label_ + "\" moduleType=\"" +
                   description.type_ + "\"");
      });
    }
  }

  void LibraryMemoryReport::report(std::string const& signalName) const {
    auto const smapsOrNone = readProcSmaps(collectLibraryPaths_);
    if (not smapsOrNone) {
      edm::LogAbsolute("LibraryMemoryReport")
          << "LibraryMemoryReport " << signalName << ": failed to read /proc/self/smaps";
      return;
    }
    auto const& smaps = *smapsOrNone;
    auto const soIndex = static_cast<unsigned int>(SmapsSection::kSharedObject);
    edm::LogAbsolute report("LibraryMemoryReport");
    report << "LibraryMemoryReport " << signalName << ":";
    if (collectLibraryPaths_) {
      report << " libraries=" << smaps.libraries.size();
    }
    report << " VSIZE " << smaps.sectionVSize_[soIndex] << " kB"
           << " (executable " << smaps.sharedObjectExecutableSize_ << " kB RSS " << smaps.sectionRss_[soIndex]
           << " kB PSS " << smaps.sharedObjectPss_ << " kB)";
    if (collectLibraryPaths_) {
      std::string fileName = signalName;
      std::replace(fileName.begin(), fileName.end(), ' ', '_');
      fileName.erase(std::remove(fileName.begin(), fileName.end(), '\"'), fileName.end());
      fileName += ".txt";
      std::ofstream file(fileName);
      if (not file) {
        edm::LogWarning("LibraryMemoryReport") << "Failed to open file " << fileName << " for writing";
        return;
      }
      file << "LibraryMemoryReport " << signalName << ": libraries=" << smaps.libraries.size() << "\n";
      for (auto const& library : smaps.libraries) {
        file << "library name=\"" << library.name << "\" VSIZE " << library.sizeKB << " kB"
             << " RSS " << library.rssKB << " kB PSS " << library.pssKB << " kB"
             << " executableSize " << library.executableSizeKB << " kB\n";
      }
    }
  }

  void LibraryMemoryReport::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.addUntracked<bool>("collectLibraryPaths", false)
        ->setComment(
            "If true the memory used by each individual library is written to a text file (in the current working "
            "directory) at postServicesConstruction, beginProcessing and postEndJob.");
    description.addUntracked<bool>("allModules", false)
        ->setComment(
            "If true the memory report will be printed after the construction of each module, source, and ESModule "
            "with only the sum of all libraries.");
    descriptions.add("LibraryMemoryReport", description);
  }

}  // namespace  edm::service

using edm::service::LibraryMemoryReport;
DEFINE_FWK_SERVICE(LibraryMemoryReport);
