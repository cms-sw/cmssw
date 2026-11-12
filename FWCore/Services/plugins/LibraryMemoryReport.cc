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
#include <format>

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
        std::string msg = std::format(
            "postModuleConstruction moduleLabel=\"{}\" moduleType=\"{}\"", module.moduleLabel(), module.moduleName());
        report(msg);
      });
      registry.watchPostSourceConstruction([this](edm::ModuleDescription const& module) {
        std::string msg = std::format(
            "postSourceConstruction moduleLabel=\"{}\" moduleType=\"{}\"", module.moduleLabel(), module.moduleName());
        report(msg);
      });
      registry.watchPostESModuleConstruction([this](edm::eventsetup::ComponentDescription const& description) {
        std::string msg = std::format(
            "postESModuleConstruction moduleLabel=\"{}\" moduleType=\"{}\"", description.label_, description.type_);
        report(msg);
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
    std::string msg = std::format("VSIZE {} kB (executable {} kB) RSS {} kB PSS {} kB",
                                  smaps.sectionVSize_[soIndex],
                                  smaps.sectionExecutableSize_[soIndex],
                                  smaps.sectionRss_[soIndex],
                                  smaps.sectionPss_[soIndex]);
    report << " " << msg;
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
      std::string msg = std::format("LibraryMemoryReport {} libraries={}", signalName, smaps.libraries.size());
      file << msg << "\n";
      for (auto const& library : smaps.libraries) {
        std::string msg = std::format("library name {} VSIZE {} kB RSS {} kB PSS {} kB executableSize {} kB",
                                      library.name,
                                      library.vsizeKB_,
                                      library.rssKB_,
                                      library.pssKB_,
                                      library.executableSizeKB_);
        file << msg << "\n";
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
