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
  class MemoryMappedFilesReport {
  public:
    MemoryMappedFilesReport(edm::ParameterSet const&, edm::ActivityRegistry&);

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    // Prints a summary of the memory used by memory mapped files.
    // If reportMemoryMappedFilesPaths is true the memory used by each individual path
    // is written to a text file, named after signalName with the spaces replaced
    // by underscores, in the current working directory.
    void report(std::string const& signalName) const;

    bool allModules_;
    bool collectPaths_;
  };

  MemoryMappedFilesReport::MemoryMappedFilesReport(edm::ParameterSet const& iPS, edm::ActivityRegistry& registry)
      : allModules_(iPS.getUntrackedParameter<bool>("allModules")),
        collectPaths_(iPS.getUntrackedParameter<bool>("collectPaths")) {
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

  void MemoryMappedFilesReport::report(std::string const& signalName) const {
    auto const smapsOrNone = readProcSmaps(collectPaths_);
    if (not smapsOrNone) {
      edm::LogAbsolute("MemoryMappedFilesReport")
          << "MemoryMappedFilesReport " << signalName << ": failed to read /proc/self/smaps";
      return;
    }
    auto const& smaps = *smapsOrNone;
    auto const soIndex = static_cast<unsigned int>(SmapsSection::kSharedObject);
    edm::LogAbsolute report("MemoryMappedFilesReport");
    report << "MemoryMappedFilesReport " << signalName << ":";
    if (collectPaths_) {
      report << " paths=" << smaps.paths.size();
    }
    std::string msg = std::format("VSIZE {} kB (executable {} kB) RSS {} kB PSS {} kB",
                                  smaps.sectionVSize_[soIndex],
                                  smaps.sectionExecutableSize_[soIndex],
                                  smaps.sectionRss_[soIndex],
                                  smaps.sectionPss_[soIndex]);
    report << " " << msg;
    if (collectPaths_) {
      std::string fileName = signalName;
      std::replace(fileName.begin(), fileName.end(), ' ', '_');
      fileName.erase(std::remove(fileName.begin(), fileName.end(), '\"'), fileName.end());
      fileName += ".txt";
      std::ofstream file(fileName);
      if (not file) {
        edm::LogWarning("MemoryMappedFilesReport") << "Failed to open file " << fileName << " for writing";
        return;
      }
      std::string msg = std::format("MemoryMappedFilesReport {} paths={}", signalName, smaps.paths.size());
      file << msg << "\n";

      auto paths = smaps.paths;
      std::sort(paths.begin(), paths.end(), [](auto const& lhs, auto const& rhs) { return lhs.rssKB_ > rhs.rssKB_; });
      for (auto const& path : paths) {
        std::string msg = std::format("path {} VSIZE {} kB RSS {} kB PSS {} kB executableSize {} kB",
                                      path.path,
                                      path.vsizeKB_,
                                      path.rssKB_,
                                      path.pssKB_,
                                      path.executableSizeKB_);
        file << msg << "\n";
      }
    }
  }

  void MemoryMappedFilesReport::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription description;
    description.addUntracked<bool>("collectPaths", false)
        ->setComment(
            "If true the memory used by each individual memory mapped file path is written to a text file (in the "
            "current working "
            "directory) at postServicesConstruction, beginProcessing and postEndJob.");
    description.addUntracked<bool>("allModules", false)
        ->setComment(
            "If true the memory report will be printed after the construction of each module, source, and ESModule "
            "with only the sum of all paths.");
    descriptions.add("MemoryMappedFilesReport", description);
  }

}  // namespace  edm::service

using edm::service::MemoryMappedFilesReport;
DEFINE_FWK_SERVICE(MemoryMappedFilesReport);
