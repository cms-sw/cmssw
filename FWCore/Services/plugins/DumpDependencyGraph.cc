/*
 * Service to dump the module dependency graph of a process as a .json
 */

#include <algorithm>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ModuleConsumesInfo.h"
#include "FWCore/ServiceRegistry/interface/PathsAndConsumesOfModulesBase.h"
#include "FWCore/ServiceRegistry/interface/ProcessContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/BranchType.h"

using json = nlohmann::json;

namespace {
  // the label the framework gives to the unnamed input source
  std::string const kSourceLabel = "source";

  std::string const kEmptyLabel = "@EmptyLabel@";

  // Get the value of the "@module_edm_type" parameter (e.g. "Source",
  // "EDProducer", etc.) every module carries in its ParameterSet.
  std::string moduleType(edm::ModuleDescription const& module) {
    auto const* pset = edm::pset::Registry::instance()->getMapped(module.parameterSetID());
    if (pset and pset->existsAs<std::string>("@module_edm_type")) {
      return pset->getParameter<std::string>("@module_edm_type");
    }
    return "Unknown";
  }

  // Map each EDAlias label to the labels of the modules it aliases
  std::unordered_map<std::string, std::vector<std::string>> aliasTargets(edm::ParameterSetID const& processPSetID) {
    std::unordered_map<std::string, std::vector<std::string>> targets;
    auto const* processPSet = edm::pset::Registry::instance()->getMapped(processPSetID);

    if (not processPSet or not processPSet->existsAs<std::vector<std::string>>("@all_aliases")) {
      // There are no aliases
      return targets;
    }
    for (std::string const& alias : processPSet->getParameter<std::vector<std::string>>("@all_aliases")) {
      targets[alias] = processPSet->getParameterSet(alias).getParameterNamesForType<edm::VParameterSet>();
    }
    return targets;
  }

  // the labels of a list of modules, in order, as a JSON array
  json moduleLabels(std::vector<edm::ModuleDescription const*> const& modules) {
    json labels = json::array();
    for (edm::ModuleDescription const* module : modules) {
      labels.push_back(module->moduleLabel());
    }
    return labels;
  }
}  // namespace

class DumpDependencyGraph {
public:
  DumpDependencyGraph(edm::ParameterSet const& pset, edm::ActivityRegistry& registry)
      : fileName_(pset.getUntrackedParameter<std::string>("fileName")) {
    registry.watchPreSourceConstruction([this](edm::ModuleDescription const& module) {
      modules_[module.moduleLabel()] = {{"class", module.moduleName()}, {"type", "Source"}};
    });
    registry.watchLookupInitializationComplete(this, &DumpDependencyGraph::lookupInitializationComplete);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.setComment(
        "Dumps the module dependency graph of a process as JSON."
        "\nThe document is keyed by module label and has the following fields:"
        "\n - process: the process name"
        "\n - modules: {label: {class, type, consumes, consumesNonEvent, consumesUnresolved}}"
        "\n - paths / endpaths: {name: [labels in schedule order]}"
        "\nEach module's three 'consume' lists are omitted when empty, and hold:"
        "\n - consumes: event-level dependencies on modules of this process"
        "\n - consumesNonEvent: the same, for the non-event transitions (run, "
        " lumi and process block), excluding modules already listed in 'consumes'."
        "\n - consumesUnresolved: declared labels, on any transition, whose data product"
        " cannot come from this process, and thus has to be read from the input");
    desc.addUntracked<std::string>("fileName", "dependency_graph.json");
    descriptions.add("DumpDependencyGraph", desc);
  }

  void lookupInitializationComplete(edm::PathsAndConsumesOfModulesBase const& pathsAndConsumes,
                                    edm::ProcessContext const& context) {
    // record the class and the type of all the modules in the process
    for (edm::ModuleDescription const* module : pathsAndConsumes.allModules()) {
      auto& entry = modules_[module->moduleLabel()];
      entry["class"] = module->moduleName();
      entry["type"] = moduleType(*module);
    }

    std::string const& processName = context.processName();
    auto const aliases = aliasTargets(context.parameterSetID());

    for (edm::ModuleDescription const* consumer : pathsAndConsumes.allModules()) {
      std::set<std::string> consumed;
      std::set<std::string> consumesNonEvent;
      std::set<std::string> unresolved;

      // collect the dependencies already resolved by the framework
      for (edm::ModuleDescription const* produced :
           pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id(), edm::InEvent)) {
        // Event depencencies
        consumed.insert(produced->moduleLabel());
      }
      for (edm::BranchType branchType : {edm::InLumi, edm::InRun, edm::InProcess}) {
        for (edm::ModuleDescription const* produced :
             pathsAndConsumes.modulesWhoseProductsAreConsumedBy(consumer->id(), branchType)) {
          // Non event-only dependencies
          consumesNonEvent.insert(produced->moduleLabel());
        }
      }

      // Data products from a prior process or from Source. Look for those in
      // the declared consumes().
      for (edm::ModuleConsumesInfo const& info : pathsAndConsumes.moduleConsumesInfos(consumer->id())) {
        std::string label{info.label()};
        if (label.empty() or label == kEmptyLabel) {
          continue;
        }

        std::set<std::string>& target = (info.branchType() == edm::InEvent) ? consumed : consumesNonEvent;

        if (info.skipCurrentProcess() or (not info.process().empty() and info.process() != processName)) {
          // If products come from the earleir process
          // (info.skipCurrentProcess()) or the process name is something
          // different from this process' name.

          unresolved.insert(label);
        } else if (auto alias = aliases.find(label); alias != aliases.end()) {
          if (std::find(alias->second.begin(), alias->second.end(), kSourceLabel) != alias->second.end()) {
            // is the Source among the modules this alias stands for?
            target.insert(kSourceLabel);
          }
        } else if (label == kSourceLabel) {
          // the Source is not reported by modulesWhoseProductsAreConsumedBy()
          target.insert(kSourceLabel);
        } else if (not modules_.contains(label)) {
          // no module of this process makes this data product.
          unresolved.insert(label);
        }
      }

      // Remove self-dependencies (modules consuming their own products)
      consumed.erase(consumer->moduleLabel());
      consumesNonEvent.erase(consumer->moduleLabel());

      // Don't repeat in consumesNonEvent a label already reported in consumed
      // This way consumesNotEvent declares the edges that exist *only* because
      // of a non-event transition
      for (std::string const& label : consumed) {
        consumesNonEvent.erase(label);
      }

      // write to the json object
      json& entry = modules_[consumer->moduleLabel()];
      if (not consumed.empty()) {
        entry["consumes"] = consumed;
      }
      if (not consumesNonEvent.empty()) {
        entry["consumesNonEvent"] = consumesNonEvent;
      }
      if (not unresolved.empty()) {
        entry["consumesUnresolved"] = unresolved;
      }
    }

    // save the modules scheduled on each Path and EndPath, in schedule order.
    json paths = json::object();
    for (unsigned int i = 0; i < pathsAndConsumes.paths().size(); ++i) {
      paths[pathsAndConsumes.paths()[i]] = moduleLabels(pathsAndConsumes.modulesOnPath(i));
    }
    json endpaths = json::object();
    for (unsigned int i = 0; i < pathsAndConsumes.endPaths().size(); ++i) {
      endpaths[pathsAndConsumes.endPaths()[i]] = moduleLabels(pathsAndConsumes.modulesOnEndPath(i));
    }

    // write out the dependency graph
    json out;
    out["process"] = context.processName();
    out["modules"] = std::move(modules_);
    out["paths"] = std::move(paths);
    out["endpaths"] = std::move(endpaths);

    std::ofstream file(fileName_);
    file << out.dump();
  }

private:
  std::string const fileName_;

  json modules_ = json::object();
};

// define as a framework service
DEFINE_FWK_SERVICE(DumpDependencyGraph);
